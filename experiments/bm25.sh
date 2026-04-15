#!/bin/bash

#First call creating the index

echo "Creating index and running the 2024-test for BM25..."
python3 ./src/classical/BaseBM25.py 2024-test ./indexes/bm25-cleaned/ ./queries/2024/test-partial/queries.jsonl --corpus_path ./dataset/corpus_cleaned.tsv --out_dir ./results/bm25

echo "Running remaining queries"
python3 ./src/classical/BaseBM25.py 2024-train ./indexes/bm25-cleaned/ ./queries/2024/train/queries.jsonl --out_dir ./results/bm25

python3 ./src/classical/BaseBM25.py 2023-test ./indexes/bm25-cleaned/ ./queries/2023/test/queries.jsonl --out_dir ./results/bm25

python3 ./src/classical/BaseBM25.py 2023-train ./indexes/bm25-cleaned/ ./queries/2023/train/queries.jsonl --out_dir ./results/bm25