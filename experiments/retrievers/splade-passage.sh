#!/bin/bash

# Source the configuration file (assuming it is saved as config.sh)
source ./experiments/retrievers/config.sh

SPLADE_CORPUS_PATH="./dataset/splade/cleaned-corpus-splade-tokenized-overlap.tsv"
SPLADE_INDEX_DIR="${BASE_INDEX_DIR}/SPLADE-pisa-passage"
SPLADE_OUT_DIR="${BASE_OUT_DIR}/splade-passage"
DOC_MAP_PATH="./dataset/splade/cleaned-idx-to-pid-splade-overlap.json" # Update if this changed for 2025

case "$QUERY_SET" in
    *"-ast")
        MODEL_NAME="models/splade_query_encoder_ft_final"
        ;;
    *)
        MODEL_NAME="naver/splade-v3"
        ;;
esac

echo "Running experiments: ${EXP_NAMES[*]} with model ${MODEL_NAME}"

python ./src/classical/Splade-Passage.py \
    "${SPLADE_INDEX_DIR}/" \
    "${SPLADE_OUT_DIR}/" \
    "${EXP_NAMES[@]}" \
    --queries_path "${EXP_QUERIES[@]}" \
    --doc_id_map_path "${DOC_MAP_PATH}" \
    --query_key "${QUERY_KEY}" \
    --model "${MODEL_NAME}" \

echo "Done."