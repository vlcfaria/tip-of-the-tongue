import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "./tot_sft_model_final"
output_path = "./tot_sft_merged_model"

print("Loading base model...", base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu", # Doing this on CPU saves VRAM, just takes a bit of RAM
)

print("Loading adapter and merging...", adapter_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
# This is the magic command that fuses the weights
merged_model = model.merge_and_unload()

print("Saving merged model...", output_path)
merged_model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.save_pretrained(output_path)

print(f"Done! Standalone model saved to {output_path}")