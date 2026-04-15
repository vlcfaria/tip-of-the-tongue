#!/bin/bash

echo "Converting corpus to .tsv"
python3 ./src/utils/convert_to_colbert.tsv

echo "Converting queries to .tsv"
python3 ./src/utils/convert_queries_to_tsv.py ./queries/

echo "Creating index for colBERT & searching"
python src/classical/ColBERT.py ./indexes/colbert/ ./results/colbert/ train-2023 test-2023 train-2024 test-2024 --queries_paths ./queries/2023/train/queries.tsv ./queries/2023/test/queries.tsv ./queries/2024/train/queries.tsv ./queries/2024/test-partial/queries.tsv  --doc_id_map_path ./dataset/idx_to_pid.json --corpus_dir ./dataset/corpus_colbert.tsv