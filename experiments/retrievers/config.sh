#!/bin/bash

BASE_INDEX_DIR="./indexes"
BASE_OUT_DIR="./results"
BASE_CORPUS_PATH="./dataset/trec-tot-2025-corpus.jsonl"

declare -A EXPERIMENTS_JSONL=(
    #["dev1"]="./queries/dev1/dev1-2025-queries.jsonl"
    #["dev2"]="./queries/dev2/dev2-2025-queries.jsonl"
    #["dev3"]="./queries/dev3/dev3-2025-queries.jsonl"
    #["train-2025"]="./queries/train-2025/train-2025-queries.jsonl"
    ["test-2025"]="./queries/test-2025/test-2025-queries.jsonl"
    ["train-2026"]="./queries/2026/train/queries-train-en.jsonl"
    #["dev-2026"]="./queries/2026/dev/queries-dev-en.jsonl"
)

declare -A REWRITTEN_EXPERIMENTS_JSONL=(
    #["dev1"]="./queries/dev1/rewritten-queries.jsonl"
    #["dev2"]="./queries/dev2/rewritten-queries.jsonl"
    #["dev3"]="./queries/dev3/rewritten-queries.jsonl"
    #["train-2025"]="./queries/train-2025/rewritten-queries.jsonl"
    ["test-2025"]="./queries/test-2025/rewritten-queries.jsonl"
)

declare -A REWRITTEN_EXPERIMENTS_SFT_JSONL=(
    #["dev1"]="./queries/dev1/rewritten-queries-sft.jsonl"
    #["dev2"]="./queries/dev2/rewritten-queries-sft.jsonl"
    #["dev3"]="./queries/dev3/rewritten-queries-sft.jsonl"
    #["train-2025"]="./queries/train-2025/rewritten-queries-sft.jsonl"
    #["test-2025"]="./queries/test-2025/rewritten-queries-sft.jsonl"
    #["sft-train"]="./queries/sft-train/rewritten-queries-sft.jsonl"
    #["dev-2026"]="./queries/2026/dev/rewritten-queries-sft.jsonl"
    ["dpo-train"]="./queries/dpo-train/dpo-train-queries-sft.jsonl"
)

declare -A REWRITTEN_EXPERIMENTS_SIMPO_JSONL=(
    #["dev1"]="./queries/dev1/rewritten-queries-simpo.jsonl"
    #["dev2"]="./queries/dev2/rewritten-queries-simpo.jsonl"
    #["dev3"]="./queries/dev3/rewritten-queries-simpo.jsonl"
    #["train-2025"]="./queries/train-2025/rewritten-queries-simpo.jsonl"
    ["test-2025"]="./queries/test-2025/rewritten-queries-simpo.jsonl"
    #["sft-train"]="./queries/simpo-train/rewritten-queries-simpo.jsonl"
    ["dev-2026"]="./queries/2026/dev/rewritten-queries-simpo.jsonl"
    #["dpo-train"]="./queries/dpo-train/dpo-train-queries-simpo.jsonl"
)

declare -A EXPERIMENTS_TSV=(
    ["dev1"]="./queries/dev1/dev1-2025-queries.tsv"
    ["dev2"]="./queries/dev2/dev2-2025-queries.tsv"
    ["dev3"]="./queries/dev3/dev3-2025-queries.tsv"
    ["train-2025"]="./queries/train-2025/train-2025-queries.tsv"
    ["test-2025"]="./queries/test-2025/test-2025-queries.tsv"
)

QUERY_SET="$1"
# The :- syntax means "if $2 is empty or unset, use 'QUERIES' instead"
QUERY_KEY="${2:-queries}" 

# 2. Configure variables based on the first parameter
case "$QUERY_SET" in
    "standard")
        declare -n EXPERIMENTS="EXPERIMENTS_JSONL"
        BASE_OUT_DIR="${BASE_OUT_DIR}/raw-query"
        ;;
    "rewritten-baseline")
        declare -n EXPERIMENTS="REWRITTEN_EXPERIMENTS_JSONL"
        BASE_OUT_DIR="${BASE_OUT_DIR}/rewrite-baseline"
        ;;
    "rewritten-sft")
        declare -n EXPERIMENTS="REWRITTEN_EXPERIMENTS_SFT_JSONL"
        BASE_OUT_DIR="${BASE_OUT_DIR}/rewrite-sft"
        ;;
    "rewritten-sft-ast")
        declare -n EXPERIMENTS="REWRITTEN_EXPERIMENTS_SFT_JSONL"
        BASE_OUT_DIR="${BASE_OUT_DIR}/rewrite-sft-ast"
        ;;
    "rewritten-simpo-ast")
        declare -n EXPERIMENTS="REWRITTEN_EXPERIMENTS_SIMPO_JSONL"
        BASE_OUT_DIR="${BASE_OUT_DIR}/rewrite-simpo-ast"
        ;;
    *)
        echo "Error: Invalid parameter."
        echo "Usage: $0 {standard|rewritten-baseline|rewritten-sft|rewritten-sft-ast|rewritten-simpo-ast} [query_key]"
        exit 1
        ;;
esac

echo "Running for $QUERY_SET queries..."
echo "Output path set to: $BASE_OUT_DIR"
echo "Query key set to: $QUERY_KEY"

EXP_NAMES=()
EXP_QUERIES=()

# Extract keys and values from the associative array into ordered standard arrays
for exp in "${!EXPERIMENTS[@]}"; do
    EXP_NAMES+=("$exp")
    EXP_QUERIES+=("${EXPERIMENTS[$exp]}")
done