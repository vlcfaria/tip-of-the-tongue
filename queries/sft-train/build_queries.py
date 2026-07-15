import pandas as pd
import os

TOTAL_QUERIES = 2500

real_files = [
    'queries/dev1/dev1-2025-queries.jsonl', 
    'queries/dev2/dev2-2025-queries.jsonl', 
    'queries/train-2025/train-2025-queries.jsonl'
]
qrels_files = [
    'queries/dev1/dev1-2025-qrel.txt', 
    'queries/dev2/dev2-2025-qrel.txt', 
    'queries/train-2025/train-2025-qrel.txt'
]

synthetic_file = 'queries/2026/train/queries-train-en.jsonl'
synthetic_qrel = 'queries/2026/train/qrels-train-en.txt'

output_file = 'queries/sft-train/queries.jsonl'
output_qrel_file = 'queries/sft-train/qrels.txt'

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 1. Load the REAL queries and keep them completely intact
real_dfs = []
for fp in real_files:
    df = pd.read_json(fp, lines=True)
    
    if 'id' in df.columns and 'query_id' not in df.columns:
        df = df.rename(columns={'id': 'query_id'})
        
    df['original_query_id'] = df['query_id'].astype(str).str.strip()
    # Normalize ID to numeric to clear leading zero discrepancies
    df['merge_id'] = pd.to_numeric(df['original_query_id'], errors='coerce')
    df['source_dataset'] = os.path.basename(fp).replace('.jsonl', '')
    real_dfs.append(df)

real_queries = pd.concat(real_dfs, ignore_index=True)
num_real_queries = len(real_queries)

# 2. Load the SYNTHETIC queries
synth_df = pd.read_json(synthetic_file, lines=True)

if 'id' in synth_df.columns and 'query_id' not in synth_df.columns:
    synth_df = synth_df.rename(columns={'id': 'query_id'})
    
synth_df['original_query_id'] = synth_df['query_id'].astype(str).str.strip()
synth_df['merge_id'] = pd.to_numeric(synth_df['original_query_id'], errors='coerce')
synth_df['source_dataset'] = os.path.basename(synthetic_file).replace('.jsonl', '')

# 3. Downsample ONLY the synthetic queries to hit the 2,500 target
num_synth_needed = TOTAL_QUERIES - num_real_queries

if num_synth_needed > 0:
    synth_sampled = synth_df.sample(n=num_synth_needed, random_state=42).reset_index(drop=True)
else:
    synth_sampled = pd.DataFrame(columns=synth_df.columns)

# 4. Combine real and sampled synthetic queries
all_queries = pd.concat([real_queries, synth_sampled], ignore_index=True)

# Assign clean numeric IDs starting from 1
all_queries['query_id'] = (all_queries.index + 1).astype(str)

# Save combined queries
all_queries.to_json(output_file, orient='records', lines=True)
print(f"✅ Saved {len(all_queries)} queries (Real: {num_real_queries}, Synthetic: {len(synth_sampled)}) to {output_file}")

# 5. Load ALL qrels
qrel_dfs = []
for fp in qrels_files + [synthetic_qrel]:
    df = pd.read_csv(fp, sep=r'\s+', names=['original_query_id', 'iteration', 'doc_id', 'relevance'], dtype={'original_query_id': str})
    
    df['original_query_id'] = df['original_query_id'].str.strip()
    # Normalize ID to numeric to match the queries dataframe exactly
    df['merge_id'] = pd.to_numeric(df['original_query_id'], errors='coerce')
    
    is_synthetic = "2026" in fp
    df['source_dataset'] = os.path.basename(synthetic_file).replace('.jsonl', '') if is_synthetic \
                           else os.path.basename(fp).replace('-qrel.txt', '-queries')
    qrel_dfs.append(df)

all_qrels = pd.concat(qrel_dfs, ignore_index=True)

# 6. Map old IDs using the normalized 'merge_id' and 'source_dataset'
merged_qrels = pd.merge(
    all_qrels, 
    all_queries[['query_id', 'merge_id', 'source_dataset']], 
    on=['merge_id', 'source_dataset'], 
    how='inner'
)

# Reorder columns to the standard TREC format (query_id iteration doc_id relevance)
final_qrels = merged_qrels[['query_id', 'iteration', 'doc_id', 'relevance']]

# Save qrels file
final_qrels.to_csv(output_qrel_file, sep=' ', header=False, index=False)
print(f"✅ Saved {len(final_qrels)} mapped qrels to {output_qrel_file}")

corpus_df = pd.read_json('dataset/trec-tot-2025-corpus.jsonl', lines=True)
corpus_df = corpus_df.rename(columns={'id': 'doc_id'})
with_qrel = all_queries.merge(merged_qrels, how='inner', on='query_id', suffixes=('', '_right'))
queries_with_title = with_qrel.merge(corpus_df, how='inner', on='doc_id', suffixes=('', '_right'))

queries_with_title[['query_id', 'query', 'original_query_id', 'merge_id', 'source_dataset', 'title']].to_json(output_file, orient='records', lines=True)