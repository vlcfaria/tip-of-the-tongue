#!/bin/bash

# Source the configuration file (assuming it is saved as config.sh)
source ./experiments/retrievers/config.sh

HARRIER_CORPUS_PATH="./dataset/harrier-small/cleaned-corpus-harrier-small-tokenized-overlap.tsv"
HARRIER_INDEX_DIR="${BASE_INDEX_DIR}/dense-harrier-small"
HARRIER_OUT_DIR="${BASE_OUT_DIR}/dense-harrier-small"
DOC_MAP_PATH="./dataset/harrier-small/cleaned-idx-to-pid-harrier-small-overlap.json"

#Create index
echo "Creating index and running experiments: ${EXP_NAMES[*]}"

python ./src/classical/DenseRetrieval.py \
    "${HARRIER_INDEX_DIR}/" \
    "${HARRIER_OUT_DIR}/" \
    "${EXP_NAMES[@]}" \
    --queries_path "${EXP_QUERIES[@]}" \
    --doc_id_map_path "${DOC_MAP_PATH}" \
    --corpus_path "${HARRIER_CORPUS_PATH}" \
    --query_key "${QUERY_KEY}" \
    --model microsoft/harrier-oss-v1-270m \
    --experiment_name dense-retriever \

echo "Done."