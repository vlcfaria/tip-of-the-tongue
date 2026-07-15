#For the DPO dataset, we take the train-2026 dataset queries which were not select on the sft one, and then generate

import pandas as pd
import os

synthetic_file = 'queries/2026/train/queries-train-en.jsonl'
synthetic_qrel = 'queries/2026/train/qrels-train-en.txt'

sft_dataset = 'queries/sft-train/queries.jsonl'

output_file = 'queries/dpo-train/queries.jsonl'
output_qrel_file = 'queries/dpo-train/qrels.txt'

os.makedirs(os.path.dirname(output_file), exist_ok=True)

#Load 2026 dataset
synth_df = pd.read_json(synthetic_file, lines=True)

sft_df = pd.read_json(sft_dataset, lines=True)

sft_df = sft_df[sft_df['source_dataset'] == 'queries-train-en']

queries = synth_df[~synth_df['query_id'].isin(sft_df['query_id'])]

qrels = pd.read_csv(synthetic_qrel, header=None, sep= ' ', names=['query_id', 'zero', 'docid', 'one'])
qrels = qrels[qrels['query_id'].isin(queries['query_id'])]

print(qrels)

queries.to_json(output_file, lines=True, orient='records')
qrels.to_csv(output_qrel_file, header=False, index=False, sep=' ')