import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_qrels(qrels_path: str) -> set:
    """Loads TREC qrels file (no header: qid Q0 docno rel)."""
    qrels = set()
    with open(qrels_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                # Store as tuple string pairs (query_id, docno) for O(1) lookups
                if int(parts[3]) > 0:  # Only grab relevant documents
                    qrels.add((parts[0], parts[2]))
    return qrels

def compute_rewards_for_run(run_path: str, qrels: set, tau: float = 1.0) -> pd.DataFrame:
    """Processes a single retriever TREC run file and calculates composite rewards."""
    # Load TREC run: qid,Q,docno,rank,score,run_id
    df = pd.read_csv(run_path)
    
    # Cast qid to string to ensure safe parsing of virtual IDs
    df['qid'] = df['qid'].astype(str)
    
    # Unpack virtual query IDs (query_id$candidate_id)
    df[['query_id', 'candidate_id']] = df['qid'].str.split('$', expand=True)
    df['candidate_id'] = df['candidate_id'].astype(int)
    
    # Step 1: Identify Golden vs Negative documents
    # Creates a boolean mask matching our loaded qrels tuples
    df['is_rel'] = [
        (row.query_id, str(row.docno)) in qrels 
        for row in df.itertuples(index=False)
    ]
    
    def local_softmax(group):
        scores = group['score']
        std_dev = scores.std() + 1e-9 # Prevent div by zero
        z_scores = (scores - scores.mean()) / std_dev
        
        exp_scores = np.exp(z_scores / tau)
        group['prob'] = exp_scores / exp_scores.sum()
        return group

    df = df.groupby('qid', group_keys=False).apply(local_softmax)
    
    # Step 3: Compute Metrics per Virtual Query Vector
    rewards = []
    for qid, group in df.groupby('qid'):
        query_id = group['query_id'].iloc[0]
        candidate_id = group['candidate_id'].iloc[0]
        
        rel_docs = group[group['is_rel']]
        neg_docs = group[~group['is_rel']]
        
        # Defaults if golden passage is missed in top-K candidate pool
        smooth_mrr = 0.0
        cmi_score = -10.0  # Safe static penalty for missing the target entirely
        
        if not rel_docs.empty:
            # Handle Smooth MRR: checking if rank is 0-indexed or 1-indexed
            # The pattern provided shows rank 0 for top 1, so we add 2 to prevent division by zero
            top_rel_rank = rel_docs['rank'].min()
            smooth_mrr = 1.0 / np.log2(top_rel_rank + 2)
            
            # Handle Contrastive Mutual Information (CMI)
            p_gold = rel_docs['prob'].max()
            p_neg_mean = neg_docs['prob'].mean() if not neg_docs.empty else 1e-9
            cmi_score = np.log(p_gold + 1e-9) - np.log(p_neg_mean + 1e-9)
            
        # Composite reward calculation
        total_reward = 0.1 * cmi_score + smooth_mrr
        
        rewards.append({
            'query_id': query_id,
            'candidate_id': candidate_id,
            'reward': total_reward,
            'smooth_mrr': smooth_mrr,
            'cmi': cmi_score
        })
        
    return pd.DataFrame(rewards)

def build_dpo_pairs(rewards_df: pd.DataFrame, generations_df: pd.DataFrame, response_df: pd.DataFrame, original_queries_df: pd.DataFrame, retriever_name: str):
    """Pairs the rewards data with the raw strings to form (prompt, chosen, rejected) triplets."""
    dpo_pairs = []
    generations_df[['query_id', 'candidate_id']] = generations_df['query_id'].str.split('$', expand=True)
    
    generations_df = generations_df.copy()
    rewards_df = rewards_df.copy()
    
    #Explicitly align types across both dataframes
    rewards_df['query_id'] = rewards_df['query_id'].astype(str)
    generations_df['query_id'] = generations_df['query_id'].astype(str)
    rewards_df['candidate_id'] = rewards_df['candidate_id'].astype(int)
    generations_df['candidate_id'] = generations_df['candidate_id'].astype(int)
    
    # Group by the true baseline query to compare alternative generations side-by-side
    for query_id, group in rewards_df.groupby('query_id'):
        if len(group) < 2:
            continue
            
        # Extract best performing and worst performing generation variations
        chosen_row = group.loc[group['reward'].idxmax()]
        rejected_row = group.loc[group['reward'].idxmin()]
        
        # Verify a structural margin exists to avoid training on ties
        if chosen_row['reward'] <= rejected_row['reward']:
            continue
            
        # Match back to the raw source strings from your generation files
        gen_subset = generations_df[generations_df['query_id'] == query_id]
        if gen_subset.empty:
            continue
            
        # Capture the original raw query template text
        raw_query_text = gen_subset[f'{retriever_name}_query'].iloc[0]
        
        # CRITICAL FIX 2: Safe integer lookup comparison
        chosen_gen = response_df.loc[int(query_id),int(chosen_row['candidate_id']),retriever_name]
        rejected_gen = response_df.loc[int(query_id),int(rejected_row['candidate_id']),retriever_name]
        
        if chosen_gen.empty or rejected_gen.empty:
            # This will no longer trigger due to type mismatches
            print(f"[WARN] Empty generation match for query_id={query_id}")
            continue

        dpo_pairs.append({
            "query_id": int(query_id),
            "retriever": retriever_name,
            "query": original_queries_df.loc[int(query_id)]['query'],
            "chosen": chosen_gen['raw_response'],
            "rejected": rejected_gen['raw_response'],
            "metadata": {
                "chosen_reward": float(chosen_row['reward']),
                "rejected_reward": float(rejected_row['reward']),
                "margin": float(chosen_row['reward'] - rejected_row['reward'])
            }
        })
        
    return dpo_pairs

def main():
    qrels_path = "queries/dpo-train/qrels.txt"
    original_queries_jsonl = "queries/dpo-train/queries.jsonl"
    generations_jsonl = "queries/dpo-train/dpo-train-queries-sft.jsonl"
    raw_response_jsonl = "queries/dpo-train/dpo-train-queries-sft-temp.jsonl"

    retriever_runs = {
        "BM25": "results/rewrite-sft-ast/bm25/base-bm25-dpo-train-run.csv",
        "SPLADE": "results/rewrite-sft-ast/splade-passage/splade-passage-dpo-train-run-max.csv",
        "DENSE": "results/rewrite-sft-ast/dense-harrier-medium/dense-harrier-medium-dpo-train-run-max.csv"
    }
    
    print("[INFO] Loading Qrels...")
    qrels = load_qrels(qrels_path)
    
    print("[INFO] Loading Raw Language Model Generations...")
    generations_df = pd.read_json('queries/dpo-train/dpo-train-queries-sft.jsonl', lines=True)
    raw_response_df = pd.read_json(raw_response_jsonl, lines=True).set_index(['query_id', 'candidate_id', 'retriever'])
    original_queries_df = pd.read_json(original_queries_jsonl, lines=True).set_index('query_id')
    
    all_dpo_pairs = []
    
    for retriever, run_file in retriever_runs.items():
        if not Path(run_file).exists():
            print(f"[WARN] Run file for {retriever} not found at {run_file}. Skipping...")
            continue

        filtered_generations_df = generations_df[generations_df[f'{retriever}_query'].str.strip() != '']
            
        print(f"[INFO] Computing rewards for {retriever}, across {len(filtered_generations_df)} queries")
        # Tune tau parameter per retriever based on typical score spread scales
        tau_val = 10.0 if retriever == "SPLADE" else 1.0
        
        rewards_df = compute_rewards_for_run(run_file, qrels, tau=tau_val)
        
        print(f"[INFO] Constructing DPO pairs for {retriever}...")
        pairs = build_dpo_pairs(rewards_df, filtered_generations_df, raw_response_df, original_queries_df, retriever)
        all_dpo_pairs.extend(pairs)
        print(f"[SUCCESS] Created {len(pairs)} pairs for {retriever}.")

    # Save final synchronized training asset out to disk
    output_path = "queries/dpo-train/dpo_alignment_dataset.jsonl"
    with open(output_path, 'w') as f:
        for pair in all_dpo_pairs:
            f.write(json.dumps(pair) + '\n')
            
    print(f"\n[COMPLETE] Combined dataset saved with {len(all_dpo_pairs)} pairs total at: {output_path}")

if __name__ == "__main__":
    main()