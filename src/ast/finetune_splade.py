import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from sentence_transformers import SparseEncoder
import argparse
import scipy.sparse as sp

def parse_args():
    parser = argparse.ArgumentParser(description="Asymmetric contrastive fine-tuning for SPLADE ToT retrieval")
    parser.add_argument("--model_id",         type=str,   default="naver/splade-v3")
    parser.add_argument("--cache_prefix",     type=str,   default="./dataset/ast/train/splade/doc-cache")
    parser.add_argument("--dataset_path",     type=str,   default="./dataset/ast/train/splade/splade_dataset.jsonl")
    parser.add_argument("--output_dir",       type=str,   default="./models/splade_query_encoder_ft")
    parser.add_argument("--final_output_dir", type=str,   default="./models/splade_query_encoder_ft_final")
    parser.add_argument("--temperature",      type=float, default=3.0)
    parser.add_argument("--learning_rate",    type=float, default=0.00005)
    parser.add_argument("--num_negatives",    type=int,   default=8)
    parser.add_argument("--batch_size",       type=int,   default=512) 
    parser.add_argument("--num_epochs",       type=int,   default=5)
    parser.add_argument("--alpha",            type=float, default=0.05) 
    parser.add_argument("--q_lambda",         type=float, default=0.010)
    return parser.parse_args()

class ToTAsymmetricDataset(Dataset):
    def __init__(self, samples, num_hard_negatives=16):
        self.samples = samples
        self.num_hard_negatives = num_hard_negatives
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        query, pos_id, neg_ids_pool, q_idx = self.samples[idx]
        
        if len(neg_ids_pool) > self.num_hard_negatives:
            sampled_negs = random.sample(neg_ids_pool, self.num_hard_negatives)
        else:
            sampled_negs = neg_ids_pool
            
        return query, pos_id, sampled_negs, q_idx

def get_collator(tokenizer):
    def collate_fn(batch):
        queries = [item[0] for item in batch]
        pos_ids = [item[1] for item in batch]
        neg_ids = [neg for item in batch for neg in item[2]]
        q_indices = [item[3] for item in batch]

        encodings = tokenizer(queries, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "pos_ids": torch.tensor(pos_ids, dtype=torch.long),
            "neg_ids": torch.tensor(neg_ids, dtype=torch.long),
            "q_indices": torch.tensor(q_indices, dtype=torch.long),
        }
    return collate_fn

def prepare_caches(data_path, cache_prefix, model_id):
    """Now utilizing SparseEncoder natively for massively simplified caching."""
    tensor_path = f"{cache_prefix}_tensors.pt"
    map_path = f"{cache_prefix}_map.json"
    q_tensor_path = f"{cache_prefix}_query_tensors.pt"

    if os.path.exists(tensor_path) and os.path.exists(map_path) and os.path.exists(q_tensor_path):
        with open(map_path, 'r', encoding='utf-8') as f:
            doc_to_id = json.load(f)
        return doc_to_id, torch.load(tensor_path, map_location='cpu'), torch.load(q_tensor_path, map_location='cpu')

    unique_docs = set()
    queries = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                unique_docs.add(item['passage'])
                unique_docs.update(item['mined_negatives'])
                queries.append(item['SPLADE_query'])
    
    doc_list = list(unique_docs)
    doc_to_id = {doc: idx for idx, doc in enumerate(doc_list)}

    print(f"Loading {model_id} via SparseEncoder to compute caches...")
    # SparseEncoder handles everything, including bfloat16 casting
    model = SparseEncoder(model_id, model_kwargs={"torch_dtype": torch.bfloat16})
    
    print("Encoding Documents...")
    # Convert_to_tensor ensures we get the sparse vectors directly
    doc_embs = model.encode_document(doc_list, batch_size=64, show_progress_bar=True, convert_to_tensor=True).cpu()
    
    print("Encoding Queries...")
    query_embs = model.encode_query(queries, batch_size=64, show_progress_bar=True, convert_to_tensor=True).cpu()

    torch.save(doc_embs, tensor_path)
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(doc_to_id, f)
    torch.save(query_embs, q_tensor_path)
    
    del model
    torch.cuda.empty_cache()
    
    return doc_to_id, doc_embs, query_embs

class AsymmetricSPLADETrainer(Trainer):
    def __init__(self, frozen_doc_embs, frozen_query_embs, alpha=0.0, temperature=0.05, q_lambda=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        def pt_to_scipy_csr(pt_tensor):
            pt_tensor = pt_tensor.cpu().coalesce()
            indices = pt_tensor.indices().numpy()
            
            # Cast bfloat16 to float32 so NumPy can understand it
            values = pt_tensor.values().to(torch.float32).numpy()
            
            return sp.coo_matrix((values, (indices[0], indices[1])), shape=pt_tensor.shape).tocsr()

        self.frozen_doc_embs = pt_to_scipy_csr(frozen_doc_embs)
        self.frozen_query_embs = pt_to_scipy_csr(frozen_query_embs)
        self.alpha = alpha
        self.temperature = temperature
        self.q_lambda = q_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_ids = inputs.pop("pos_ids")
        neg_ids = inputs.pop("neg_ids")
        q_indices = inputs.pop("q_indices")

        # 1. Forward Pass
        # By passing the dict directly, SparseEncoder applies the MLM head 
        # and SpladePooling automatically, returning the final vectors in 'sentence_embedding'
        st_outputs = model(inputs)
        q_embs = st_outputs["sentence_embedding"]
        B = q_embs.size(0)

        # 2. FLOPs Sparsity Regularization (Crucial for SPLADE)
        # Even with the native model, we must regularize the query embeddings during our custom loss step
        q_flops_loss = torch.sum(torch.mean(torch.abs(q_embs), dim=0) ** 2)
        # Calculate Non-Zeros (NNZ) - SPLADE uses ReLU, so count elements > 0
        nnz = (q_embs > 1e-4).sum(dim=-1).float().mean()

        # Print periodically to avoid spamming the console
        if getattr(self.state, "global_step", 0) % 20 == 0:
            print(f"\nStep: {self.state.global_step} | Avg Query NNZ: {nnz.item():.2f} | FLOPs Loss: {q_flops_loss.item():.4f}")

        # 3. Retrieve Cached Embeddings
        doc_indices_gpu = torch.cat([pos_ids, neg_ids])
        doc_indices_np = torch.cat([pos_ids, neg_ids]).cpu().numpy()
        q_indices_np = q_indices.cpu().numpy()

        # Slice sparse with Scipy -> convert mini-batch to dense -> cast to PyTorch GPU tensor
        d_embs = torch.from_numpy(
            self.frozen_doc_embs[doc_indices_np].toarray()
        ).to(q_embs.device, dtype=q_embs.dtype)
        
        ref_embs = torch.from_numpy(
            self.frozen_query_embs[q_indices_np].toarray()
        ).to(q_embs.device, dtype=q_embs.dtype)

        # 4. MNRL Loss Calculation
        scores = torch.matmul(q_embs, d_embs.transpose(0, 1)) / self.temperature

        # Mask out in-batch duplicates
        duplicate_mask = (doc_indices_gpu.unsqueeze(0) == pos_ids.unsqueeze(1))
        duplicate_mask[torch.arange(B, device=q_embs.device), torch.arange(B, device=q_embs.device)] = False
        scores.masked_fill_(duplicate_mask, -1e4)
        labels = torch.arange(B, device=q_embs.device)
        
        mnrl_loss = F.cross_entropy(scores, labels)
        
        # 5. Anchor Loss (Self-Distillation)
        anchor_loss = F.mse_loss(q_embs, ref_embs, reduction='none').sum(dim=-1).mean()
        
        # Total Loss Assembly
        loss = mnrl_loss + (self.alpha * anchor_loss) + (self.q_lambda * q_flops_loss)

        return (loss, st_outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

def train(args):
    doc_to_id, frozen_doc_embs, frozen_query_embs = prepare_caches(args.dataset_path, args.cache_prefix, args.model_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = 'right' 

    # Load the model via SparseEncoder so it retains the correct pooling architecture
    query_encoder = SparseEncoder(args.model_id, model_kwargs={"torch_dtype": torch.bfloat16})
    query_encoder.train()
    for param in query_encoder.parameters():
        param.requires_grad = True

    # Setup dataset
    all_samples = []
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                query = item['SPLADE_query']
                pos_id = doc_to_id[item['passage']]
                neg_ids = [doc_to_id[neg] for neg in item['mined_negatives']]
                all_samples.append((query, pos_id, neg_ids, idx))

    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.9)
    train_dataset = ToTAsymmetricDataset(all_samples[:split_idx], num_hard_negatives=args.num_negatives)
    eval_dataset = ToTAsymmetricDataset(all_samples[split_idx:], num_hard_negatives=args.num_negatives)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        logging_steps=10,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        save_safetensors=False
    )

    trainer = AsymmetricSPLADETrainer(
        frozen_doc_embs=frozen_doc_embs,
        frozen_query_embs=frozen_query_embs,
        alpha=args.alpha,
        temperature=args.temperature,
        q_lambda=args.q_lambda,
        model=query_encoder,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_collator(tokenizer)
    )

    trainer.train()
    
    print(f"Saving the fine-tuned SPLADE query encoder to {args.final_output_dir}...")
    # SparseEncoder supports standard save methods
    inner_model = query_encoder[0].auto_model   # the bare MLM model
    inner_tokenizer = query_encoder[0].tokenizer

    inner_model.save_pretrained(args.final_output_dir)
    inner_tokenizer.save_pretrained(args.final_output_dir)
    print(f"Saved HuggingFace-format model to {args.final_output_dir}")

if __name__ == "__main__":
    args = parse_args()
    train(args)