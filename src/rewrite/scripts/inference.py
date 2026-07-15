import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from consts import get_system_prompt, get_user_prompt
import pandas as pd

# 1. Paths
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "./tot_sft_model_final"

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def generate_rewrite(query, retriever):
    # Recreate the exact prompt structure from training
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user",   "content": get_user_prompt(query, retriever)}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3000,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return result

if __name__ == "__main__":
    #Load some queries from the dev dataset
    df = pd.read_json('./queries/2026/dev/queries-dev-en.jsonl', lines=True)

    test_queries = list(df['query'][:3])
    
    # Replace this with whatever your actual retriever names are in your dataset
    test_retriever = "BM25" 
    
    print("\n--- Starting Tests ---")
    for q in test_queries:
        print(f"\nOriginal Query: {q}")
        rewrite = generate_rewrite(q, test_retriever)
        print(f"Model Rewrite:  {rewrite}")