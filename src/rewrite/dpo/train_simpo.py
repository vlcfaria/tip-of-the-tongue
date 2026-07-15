import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #same order as nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import re
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl.experimental.cpo import CPOConfig, CPOTrainer
from ..utils import extract_thinking, extract_query
from ..consts import get_user_prompt, get_system_prompt
from peft import LoraConfig

def parse_generation(text: str):
    """Extracts the think and query blocks safely."""
    return extract_thinking(text), extract_query(text)

def prepare_simpo_dataset(jsonl_path: str, tokenizer):
    df = pd.read_json(jsonl_path, lines=True)
    formatted_data = {"prompt": [], "chosen": [], "rejected": []}
    
    for _, row in df.iterrows():
        chosen_think, chosen_query = parse_generation(row['chosen'])
        # Extract the rejected think block as well
        rejected_think, rejected_query = parse_generation(row['rejected']) 
        
        prompt_messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": get_user_prompt(row['query'], row['retriever'])}
        ]
        
        # The prompt should ONLY contain the system and user messages
        prompt_str = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        formatted_data["prompt"].append(prompt_str)
        
        # The completions contain both the reasoning and the final query
        formatted_data["chosen"].append(f"<think>\n{chosen_think}\n</think>\n<query>\n{chosen_query}\n</query>")
        formatted_data["rejected"].append(f"<think>\n{rejected_think}\n</think>\n<query>\n{rejected_query}\n</query>")
        
    return Dataset.from_dict(formatted_data)


def main():
    model_id = "./models/tot_sft_merged_model"
    dataset_path = "queries/dpo-train/dpo_alignment_dataset.jsonl"
    
    print("[INFO] Loading Tokenizer & Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in bfloat16 for modern GPU efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("[INFO] Formatting Dataset & Masking Thoughts...")
    full_dataset = prepare_simpo_dataset(dataset_path, tokenizer)
    
    dataset_splits = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_splits["train"]
    eval_dataset = dataset_splits["test"]

    print("[INFO] Calculating optimal sequence length...")
    
    def calculate_lengths(example):
        chosen_len = len(tokenizer(example["prompt"] + example["chosen"], add_special_tokens=False)["input_ids"])
        rejected_len = len(tokenizer(example["prompt"] + example["rejected"], add_special_tokens=False)["input_ids"])
        
        return {
            "max_seq_length": max(chosen_len, rejected_len)
        }
    
    length_stats = train_dataset.map(calculate_lengths, num_proc=4)

    max_length = max(length_stats['max_seq_length']) + 50

    print('[INFO] Max length of', max_length)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    training_args = CPOConfig(
        output_dir="./models/simpo-lora-query-rewriter",
        loss_type="simpo",
        max_length=max_length,
        cpo_alpha=0.0,      
        beta=2.0,           
        simpo_gamma=0.5,
        learning_rate=2e-5, 
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=2,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        per_device_eval_batch_size=4,
        save_strategy="epoch",
        bf16=True, 
        gradient_checkpointing=True, 
    )

    print("[INFO] Initializing SimPOTrainer with LoRA...")
    trainer = CPOTrainer(
        model=model,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("[INFO] Starting LoRA SimPO Alignment...")
    trainer.train()
    
    print("[INFO] Saving Final Model...")
    trainer.save_model("./models/simpo-query-rewriter-final")

if __name__ == "__main__":
    main()