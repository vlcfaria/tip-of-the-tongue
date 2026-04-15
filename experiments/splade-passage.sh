#!/bin/bash

#First call creating the index

#This will fail since multithreading fails on pisaindex + terrier for some reason, but it will build the index
python3 ./src/classical/Splade-Passage.py 2024-test ./indexes/SPLADE-pisa-passage/ ./queries/2024/test-partial/queries.jsonl --corpus_path ./dataset/corpus_colbert_overlap.tsv --out_dir ./results/splade-passage

#Run queries
python src/classical/Splade-Passage.py ./indexes/SPLADE-pisa-passage/ ./results/splade-passage/ train-2023 test-2023 train-2024 test-2024  --queries_path ./queries/2023/train/queries.jsonl ./queries/2023/test/queries.jsonl ./queries/2024/train/queries.jsonl ./queries/2024/test-partial/queries.jsonl --doc_id_map_path ./dataset/idx_to_pid_splade_overlap.json 