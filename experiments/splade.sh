#!/bin/bash

#First call creating the index

echo "Creating index and running the 2024-test for SPLADE..."
python3 ./src/classical/Splade.py 2024-test ./indexes/SPLADE-pisa-cleaned/ ./queries/2024/test-partial/queries.jsonl --corpus_path ./dataset/corpus_cleaned.tsv --out_dir ./results/splade

echo "Running remaining queries"
python3 ./src/classical/Splade.py 2024-train ./indexes/SPLADE-pisa-cleaned/ ./queries/2024/train/queries.jsonl --out_dir ./results/splade

python3 ./src/classical/Splade.py 2023-test ./indexes/SPLADE-pisa-cleaned/ ./queries/2023/test/queries.jsonl --out_dir ./results/splade

python3 ./src/classical/Splade.py 2023-train ./indexes/SPLADE-pisa-cleaned/ ./queries/2023/train/queries.jsonl --out_dir ./results/splade
