import itertools
import os
import subprocess
import sys
import time
import pandas as pd


def run_orchestrator():
    # --- CONFIGURATION ---
    MAX_CONCURRENT_WORKERS = 4  # Adjust based on your CPU cores and RAM limits
    INDEX_PATH = "./indexes/bm25"  # Replace with your index directory
    TOPICS_PATH = "queries/sft-train/rewritten-queries-sft.jsonl"
    QRELS_PATH = "queries/sft-train/qrels.txt"
    OUTPUT_CSV = "bm25_grid_search_results.csv"
    METRIC = "recall_1000"
    EXPECTED_FOLDS = 5

    b_values = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    k1_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    k3_values = [0, 5, 8, 20, 100, 1000]
    # ---------------------

    # 1. Initialize empty CSV or run recovery logic
    completed_combos = set()
    if not os.path.exists(OUTPUT_CSV):
        df_empty = pd.DataFrame(columns=["b", "k1", "k3", "fold", METRIC])
        df_empty.to_csv(OUTPUT_CSV, index=False)
    else:
        print("[Orchestrator] Existing CSV found. Running recovery logic...")
        df = pd.read_csv(OUTPUT_CSV)
        
        # Round the hyperparameter columns to prevent float precision mismatch issues
        df_rounded = df.copy()
        df_rounded[["b", "k1", "k3"]] = df_rounded[["b", "k1", "k3"]].round(4)
        
        # Group by hyperparameters to count how many folds were completed
        combo_counts = df_rounded.groupby(["b", "k1", "k3"]).size()
        
        # Identify combinations that have all folds finished
        complete_indices = combo_counts[combo_counts >= EXPECTED_FOLDS].index
        completed_combos = set(complete_indices)
        
        # Filter the DataFrame to ONLY include completely finished runs and overwrite CSV
        df_cleaned = df[
            df_rounded.set_index(["b", "k1", "k3"]).index.isin(complete_indices)
        ]
        df_cleaned.to_csv(OUTPUT_CSV, index=False)
        print(f"[Orchestrator] Recovery complete: kept {len(completed_combos)} fully finished combinations. Cleared partials.")

    # 2. Generate all hyperparameter combinations and filter out the completed ones
    all_combinations = list(itertools.product(b_values, k1_values, k3_values))
    
    # Match the rounding applied to the DataFrame for a clean comparison
    combinations = [
        combo for combo in all_combinations 
        if tuple(round(v, 4) for v in combo) not in completed_combos
    ]
    
    total_combos = len(combinations)
    print(
        f"Orchestrator: Found {total_combos} combinations remaining to run using up to {MAX_CONCURRENT_WORKERS} parallel processes."
    )

    active_processes = []
    combo_idx = 0

    while combo_idx < total_combos or active_processes:
        # Spawn new workers up to the concurrent cap limit
        while (
            len(active_processes) < MAX_CONCURRENT_WORKERS
            and combo_idx < total_combos
        ):
            b, k1, k3 = combinations[combo_idx]
            combo_idx += 1

            print(
                f"[Orchestrator] Launching worker [{combo_idx}/{total_combos}] -> b={b}, k1={k1}, k3={k3}"
            )

            # Build terminal command invocation
            cmd = [
                sys.executable,
                "src/classical/bm25-tuning/bm25_worker.py",
                "--index_path",
                INDEX_PATH,
                "--topics_path",
                TOPICS_PATH,
                "--qrels_path",
                QRELS_PATH,
                "--b",
                str(b),
                "--k1",
                str(k1),
                "--k3",
                str(k3),
                "--metric",
                METRIC,
                "--output_csv",
                OUTPUT_CSV,
            ]

            # Fire off process completely isolated from current JVM context
            proc = subprocess.Popen(
                cmd, stdout=sys.stdout, stderr=subprocess.STDOUT
            )
            active_processes.append((proc, (b, k1, k3)))

        # Poll running tasks to clean up dead processes and free slots
        for active_item in active_processes[:]:
            proc, combo = active_item
            return_code = proc.poll()

            if return_code is not None:  # Process finished execution
                if return_code != 0:
                    _, stderr = proc.communicate()
                    print(
                        f"[ALERT] Worker for combo {combo} failed with exit code {return_code}."
                    )
                    print(f"Error Log:\n{stderr.decode('utf-8')}")
                active_processes.remove(active_item)

        time.sleep(0.5)  # Throttle polling loop slightly to yield CPU

    print("\n[Orchestrator] Grid search complete across all configurations.")


if __name__ == "__main__":
    run_orchestrator()