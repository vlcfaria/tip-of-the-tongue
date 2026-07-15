import subprocess
from itertools import product
import json
import os

GRID = {
    "temperature":   [1.0, 2.0, 3.0],
    "learning_rate": [1e-5, 3e-5, 5e-5],
    "alpha":         [0.0, 0.05, 0.1, 0.25],
    "q_lambda":      [0.0, 0.005, 0.01],
    "num_negatives": [8],
    "num_epochs":    [5],
}

RESULTS_DIR = "results/grid-search-splade"
DONE_LOG    = f"{RESULTS_DIR}/completed_runs.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)

completed = set()
if os.path.exists(DONE_LOG):
    with open(DONE_LOG) as f:
        completed = set(f.read().splitlines())

configs = [
    dict(zip(GRID.keys(), values))
    for values in product(*GRID.values())
]
print(f"Starting grid search: {len(configs)} total configs")

for i, cfg in enumerate(configs):
    run_name = (
        f"temp{cfg['temperature']}"
        f"_lr{cfg['learning_rate']}"
        f"_neg{cfg['num_negatives']}"
        f"_ep{cfg['num_epochs']}"
        f"_alpha{cfg['alpha']}"
        f"_lambda{cfg['q_lambda']}"
    )

    if run_name in completed:
        print(f"[{i+1}/{len(configs)}] Skipping {run_name} (already done)")
        continue

    print(f"\n[{i+1}/{len(configs)}] Run: {run_name}")

    train_cmd = [
        "python", "src/ast/finetune_splade.py",
        "--temperature",      str(cfg["temperature"]),
        "--learning_rate",    str(cfg["learning_rate"]),
        "--num_negatives",    str(cfg["num_negatives"]),
        "--num_epochs",       str(cfg["num_epochs"]),
        "--alpha",            str(cfg["alpha"]),
        "--q_lambda",         str(cfg["q_lambda"]),
    ]
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  Training failed (exit {e.returncode}), skipping retrieval and eval.")
        continue

    retriever_cmd = [
        "experiments/retrievers/splade-passage.sh", "rewritten-sft-ast", "SPLADE_query"
    ]
    try:
        subprocess.run(retriever_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  Retrieval failed (exit {e.returncode}), skipping eval.")
        continue

    eval_cmd = [
        "python", "results/eval.py",
        "--run_name",   run_name,
        "--output_dir", RESULTS_DIR,
        "--cfg",        json.dumps(cfg),
    ]
    try:
        subprocess.run(eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  Eval failed (exit {e.returncode}).")
        continue

    with open(DONE_LOG, "a") as f:
        f.write(run_name + "\n")

print("\nGrid search complete.")