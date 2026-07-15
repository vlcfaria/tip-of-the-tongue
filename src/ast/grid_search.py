import subprocess
from itertools import product
import json
import os

GRID = {
    "temperature":   [0.02, 0.05, 0.1],
    "learning_rate": [1e-6, 5e-6, 1e-5],
    "num_negatives": [4, 8, 16],
    "num_epochs": [3,5,10],
    "batch_size": [512, 1024],
}

# Fixed params
BASE_ARGS = {
    "model_id":      "microsoft/harrier-oss-v1-0.6b",
    "dataset_path":  "./dataset/ast/train/harrier/harrier_dataset.jsonl",
    "cache_prefix":  "./dataset/ast/train/harrier/doc-cache",
    "alpha":         0.0,
    "gpu":           "1",
}

DONE_LOG = "results/grid-search/completed_runs.txt"
os.makedirs("results/grid-search", exist_ok=True)

if os.path.exists(DONE_LOG):
    with open(DONE_LOG) as f:
        completed = set(f.read().splitlines())
else:
    completed = set()

configs = [
    dict(zip(GRID.keys(), values))
    for values in product(*GRID.values())
]

print(f"Starting grid search: {len(configs)} runs total")

for i, cfg in enumerate(configs):
    run_name = (
        f"temp{cfg['temperature']}"
        f"_lr{cfg['learning_rate']}"
        f"_neg{cfg['num_negatives']}"
        f"_ep{cfg['num_epochs']}"
        f"_batch{cfg['batch_size']}"
    )

    print(f"\n[{i+1}/{len(configs)}] Run: {run_name}")

    if run_name in completed:
        print(f"[{i+1}/{len(configs)}] Skipping {run_name} (already done)")
        continue

    print(f"\n[{i+1}/{len(configs)}] Run: {run_name}")

    #Finetune
    train_cmd = [
        "python", "src/ast/finetune_harrier.py",
        "--temperature",      str(cfg["temperature"]),
        "--learning_rate",    str(cfg["learning_rate"]),
        "--num_negatives",    str(cfg["num_negatives"]),
        #"--output_dir",       output_dir,
        #"--final_output_dir", final_output_dir,
        "--model_id",         BASE_ARGS["model_id"],
        "--dataset_path",     BASE_ARGS["dataset_path"],
        "--cache_prefix",     BASE_ARGS["cache_prefix"],
        "--num_epochs",       str(cfg["num_epochs"]),
        "--batch_size",       str(cfg["batch_size"]),
        "--alpha",            str(BASE_ARGS["alpha"]),
        "--gpu",              BASE_ARGS["gpu"],
    ]
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed for {run_name} (exit code {e.returncode}), skipping retrieval and eval.")
        continue  # jumps straight to next config if this crashed

    #Retrieve
    retriever_cmd = [
        "experiments/retrievers/dense-harrier-medium.sh", "rewritten-sft-ast", "DENSE_query"
    ]
    subprocess.run(retriever_cmd, check=True)

    # Evaluate
    eval_cmd = [
        "python", "results/eval.py",
        "--run_name",   run_name,
        "--output_dir", 'results/grid-search',
        "--cfg",        json.dumps(cfg),
    ]
    subprocess.run(eval_cmd, check=True)

    with open(DONE_LOG, "a") as f:
        f.write(run_name + "\n")


print("\nGrid search complete.")