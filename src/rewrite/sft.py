import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #same order as nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from consts import get_system_prompt, get_user_prompt

# 1. Define the model and load it
# If on an A100, use torch.bfloat16. If on an A2, you'd need bitsandbytes for 4-bit loading.
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # LLaMA needs a pad token defined
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto", # Automatically places the model on your GPU
    attn_implementation="flash_attention_2"
)

# 2. Configure LoRA
# This is the "parameter efficient" part. We freeze the base model and only train a small adapter.
peft_config = LoraConfig(
    r=32, # The "rank" of the adapter. 16 or 32 is standard for rewriting tasks.
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Target all linear layers for better instruction following
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 3. Load and Format your Dataset
# Assuming your 7,500 examples are in a JSONL file with 'original_query' and 'golden_rewrite' columns
full_dataset = load_dataset("json", data_files="golden-rewrites-final.jsonl", split="train")
split_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)

train_data = split_dataset["train"]
eval_data = split_dataset["test"]

def format_to_prompt_completion(batch):
    prompts = []
    completions = []
    
    for i in range(len(batch['query'])):
        for retriever, rewrite in batch['rewrites'][i].items():
            if not rewrite: 
                continue
                
            messages = [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user",   "content": get_user_prompt(batch['query'][i], retriever)}
            ]
            
            # This generates the prompt and adds the <|start_header_id|>assistant<|end_header_id|> tags
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # The completion is just your golden rewrite + Llama 3's specific conversational end token
            completion = f"{rewrite}<|eot_id|>"
            
            prompts.append(prompt)
            completions.append(completion)
            
    return {"prompt": prompts, "completion": completions}

# Apply the mapping
processed_train = train_data.map(format_to_prompt_completion, batched=True, remove_columns=train_data.column_names)
processed_eval = eval_data.map(format_to_prompt_completion, batched=True, remove_columns=eval_data.column_names)


# 4. Set Training Arguments
# With 7,500 examples, 3 epochs is the sweet spot.
training_args = SFTConfig(
    output_dir="./tot_sft_model",
    per_device_train_batch_size=8, # Increase to 8 or 16 if you are on an A100, 4 if in an A2
    gradient_accumulation_steps=8, # Simulates a larger batch size -> smoother with 8, lower if vram too high
    learning_rate=1e-4,
    logging_steps=50,
    max_length=2500,

    # --- EVALUATION ADDITIONS ---
    eval_strategy="steps", # 'evaluation_strategy' is deprecated in newer HF versions
    eval_steps=100,        # Evaluate every 100 steps
    save_strategy="steps", # Save checkpoints aligned with evaluations
    save_steps=100,
    load_best_model_at_end=True, # Automatically load the best checkpoint when done
    metric_for_best_model="eval_loss",
    # ----------------------------

    max_steps=-1,
    num_train_epochs=3, 
    optim="paged_adamw_8bit", # Memory efficient optimizer
    fp16=False,
    bf16=True, # Use bfloat16 for modern GPUs like A100
    report_to="none", # Set to "wandb" if you use Weights & Biases for tracking
    warmup_ratio=0.1,
    weight_decay=0.05,
    lr_scheduler_type="cosine",

    completion_only_loss=True
)

# 5. Initialize the Trainer and Start!
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_train,
    eval_dataset=processed_eval,
    peft_config=peft_config,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Starting training...")
trainer.train()

# 6. Save the final adapter weights
trainer.model.save_pretrained("./tot_sft_model_final")
tokenizer.save_pretrained("./tot_sft_model_final")
print("Training complete and adapter saved!")