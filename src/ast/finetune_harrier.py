import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Asymmetric contrastive fine-tuning for ToT retrieval")
    parser.add_argument("--model_id",         type=str,   default="microsoft/harrier-oss-v1-0.6b")
    parser.add_argument("--cache_prefix",     type=str,   default="./dataset/ast/train/harrier/doc-cache")
    parser.add_argument("--dataset_path",     type=str,   default="./dataset/ast/train/harrier/harrier_dataset.jsonl")
    parser.add_argument("--output_dir",       type=str,   default="./models/harrier_query_encoder_ft")
    parser.add_argument("--final_output_dir", type=str,   default="./models/harrier_query_encoder_ft_final")
    parser.add_argument("--temperature",      type=float, default=0.05)
    parser.add_argument("--learning_rate",    type=float, default=0.00001)
    parser.add_argument("--num_negatives",    type=int,   default=8)
    parser.add_argument("--batch_size",       type=int,   default=1024)
    parser.add_argument("--num_epochs",       type=int,   default=5)
    parser.add_argument("--alpha",            type=float, default=0.0)
    parser.add_argument("--gpu",              type=str,   default="1")
    return parser.parse_args()

class ToTAsymmetricDataset(Dataset):
    def __init__(self, samples, num_hard_negatives=4):
        self.samples = samples
        self.num_hard_negatives = num_hard_negatives
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        query, pos_id, neg_ids_pool, q_idx = self.samples[idx]
        
        # Dynamically sample a subset of negatives for this specific step
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
    """Uses sentence-transformers to massively simplify encoding frozen docs and queries."""
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
                queries.append("Instruct: Given a noisy Tip-of-the-Tongue web query, retrieve relevant passages that answer the query\nQuery: " + item['DENSE_query'])
    
    doc_list = list(unique_docs)
    doc_to_id = {doc: idx for idx, doc in enumerate(doc_list)}

    print("Loading SentenceTransformer to compute caches...")
    st_model = SentenceTransformer(model_id, model_kwargs={"torch_dtype": torch.bfloat16})
    
    print("Encoding Documents...")
    doc_embs = st_model.encode(doc_list, batch_size=64, show_progress_bar=True, convert_to_tensor=True).cpu()
    print("Encoding Queries...")
    query_embs = st_model.encode(queries, batch_size=64, show_progress_bar=True, convert_to_tensor=True).cpu()

    torch.save(doc_embs, tensor_path)
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(doc_to_id, f)
    torch.save(query_embs, q_tensor_path)
    
    del st_model
    torch.cuda.empty_cache()
    
    return doc_to_id, doc_embs, query_embs

class AsymmetricTrainer(Trainer):
    def __init__(self, frozen_doc_embs, frozen_query_embs, alpha=0.0, temperature=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep caches on CPU to save VRAM
        self.frozen_doc_embs = frozen_doc_embs
        self.frozen_query_embs = frozen_query_embs
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs already contains input_ids and attention_mask from your collator
        pos_ids = inputs.pop("pos_ids")
        neg_ids = inputs.pop("neg_ids")
        q_indices = inputs.pop("q_indices")

        # --- THE MAGIC HAPPENS HERE ---
        # Passing the dict directly triggers the ST pipeline:
        # Transformer -> Last Token Extraction -> L2 Normalization
        st_outputs = model(inputs)
        q_embs = st_outputs["sentence_embedding"]
        B = q_embs.size(0)

        # Lazily move caches to GPU on first call
        if self.frozen_doc_embs.device != q_embs.device:
            self.frozen_doc_embs = self.frozen_doc_embs.to(q_embs.device, dtype=q_embs.dtype)
            self.frozen_query_embs = self.frozen_query_embs.to(q_embs.device, dtype=q_embs.dtype)

        doc_indices = torch.cat([pos_ids, neg_ids])

        d_embs = self.frozen_doc_embs[doc_indices]
        ref_embs = self.frozen_query_embs[q_indices]

        # Loss Calculation
        scores = torch.matmul(q_embs, d_embs.transpose(0, 1)) / self.temperature

        # 3. MASK OUT DUPLICATES (In-batch collisions)
        # Find where any document in the pool matches the query's positive document ID
        duplicate_mask = (doc_indices.unsqueeze(0) == pos_ids.unsqueeze(1))
        duplicate_mask[torch.arange(B, device=q_embs.device), torch.arange(B, device=q_embs.device)] = False
        scores.masked_fill_(duplicate_mask, -1e4)
        labels = torch.arange(q_embs.size(0), device=q_embs.device)
        
        mnrl_loss = F.cross_entropy(scores, labels)
        anchor_loss = F.mse_loss(q_embs, ref_embs)
        
        loss = mnrl_loss + (self.alpha * anchor_loss)

        return (loss, st_outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override to handle SentenceTransformer's positional `input` argument.
        The default Trainer unpacks inputs as **kwargs, which breaks ST's forward().
        We mirror compute_loss: pop auxiliary keys, pass the remainder as a dict.
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss = self.compute_loss(model, inputs)

        # Trainer expects (loss, logits, labels) — return None for logits/labels
        # since we have no token-level predictions to report
        return (loss.detach(), None, None)

def train(args):
    doc_to_id, frozen_doc_embs, frozen_query_embs = prepare_caches(args.dataset_path, args.cache_prefix, args.model_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = 'right' 

    #Setup model from training
    query_encoder = SentenceTransformer(args.model_id, model_kwargs={"torch_dtype": torch.bfloat16})
    query_encoder.train()
    for param in query_encoder.parameters():
        param.requires_grad = True

    #Setup dataset
    all_samples = []
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                query = "Instruct: Given a noisy Tip-of-the-Tongue web query, retrieve relevant passages that answer the query\nQuery: " + item['DENSE_query']
                pos_id = doc_to_id[item['passage']]
                neg_ids = [doc_to_id[neg] for neg in item['mined_negatives']]
                all_samples.append((query, pos_id, neg_ids, idx))

    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.9)
    train_dataset = ToTAsymmetricDataset(all_samples[:split_idx])
    eval_dataset = ToTAsymmetricDataset(all_samples[split_idx:])

    #Train using HF trainer
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
        remove_unused_columns=False 
    )

    trainer = AsymmetricTrainer(
        frozen_doc_embs=frozen_doc_embs,
        frozen_query_embs=frozen_query_embs,
        alpha=args.alpha,
        temperature=args.temperature,
        model=query_encoder,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_collator(tokenizer)
    )

    trainer.train()
    
    print(f"Saving the full fine-tuned SentenceTransformer model to {args.final_output_dir}...")
    query_encoder.save(args.final_output_dir)

if __name__ == "__main__":
    args = parse_args()
    train(args)