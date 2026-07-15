#!/bin/bash

# Source the configuration file (assuming it is saved as config.sh)
source ./experiments/retrievers/config.sh

BGE_CORPUS_PATH="./dataset/harrier-oss-600/cleaned-corpus-harrier-600-tokenized-overlap.tsv"
BGE_INDEX_DIR="${BASE_INDEX_DIR}/dense-harrier-medium"
BGE_OUT_DIR="${BASE_OUT_DIR}/dense-harrier-medium"
DOC_MAP_PATH="./dataset/harrier-oss-600/cleaned-idx-to-pid-harrier-600-overlap.json"

case "$QUERY_SET" in
    *"-ast")
        MODEL_NAME="models/harrier_query_encoder_ft_final"
        ;;
    *)
        MODEL_NAME="microsoft/harrier-oss-v1-0.6b"
        ;;
esac

echo "Creating index and running experiments: ${EXP_NAMES[*]} with model name: ${MODEL_NAME}"

python ./src/classical/DenseRetrieval.py \
    "${BGE_INDEX_DIR}/" \
    "${BGE_OUT_DIR}/" \
    "${EXP_NAMES[@]}" \
    --queries_path "${EXP_QUERIES[@]}" \
    --doc_id_map_path "${DOC_MAP_PATH}" \
    --corpus_path "${BGE_CORPUS_PATH}" \
    --query_key "${QUERY_KEY}" \
    --model "${MODEL_NAME}" \
    --experiment_name dense-harrier-medium \

echo "Done."