#!/bin/bash
source ./experiments/retrievers/config.sh


OUT_DIR="${BASE_OUT_DIR}/bm25-passage"
INDEX_DIR="${BASE_INDEX_DIR}/bm25-passage"
CORPUS_PATH="./dataset/harrier-oss-600/cleaned-corpus-harrier-600-tokenized-overlap.tsv"

#Create index
echo "Creating index and running experiments: ${EXP_NAMES[*]}"

python ./src/classical/BaseBM25.py \
    "${INDEX_DIR}/" \
    "${OUT_DIR}/" \
    "${EXP_NAMES[@]}" \
    --queries_path "${EXP_QUERIES[@]}" \
    --corpus_path "${CORPUS_PATH}" \
    --query_key "${QUERY_KEY}" \

echo "Done."