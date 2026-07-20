#!/bin/bash

source ./experiments/retrievers/config.sh

RERANKER_OUT_DIR="${BASE_OUT_DIR}/reranker-doc"
BM25_DIR="${BASE_OUT_DIR}/bm25"
DENSE_DIR="${BASE_OUT_DIR}/dense-harrier-medium"
SPLADE_DIR="${BASE_OUT_DIR}/splade-passage"
DOC_CORPUS="./dataset/corpus-cleaned-bm25.jsonl"

WEAK_MODEL="Qwen/Qwen3-Reranker-0.6B"
STRONG_MODEL="Qwen/Qwen3-Reranker-4B"
WEAK_MAX_LENGTH=2048
WEAK_BATCH_SIZE=128
STRONG_MAX_LENGTH=8192
STRONG_BATCH_SIZE=8
TOP_K=1000        # candidates into stage 1 (from RRF)
SURVIVORS=50     # candidates into stage 2 (top-N survivors from stage 1)

mkdir -p "${RERANKER_OUT_DIR}"

for i in "${!EXP_NAMES[@]}"; do
    exp="${EXP_NAMES[$i]}"
    queries="${EXP_QUERIES[$i]}"

    weak_output="${RERANKER_OUT_DIR}/ce-reranker-doc-${exp}-weak-run.csv"
    final_output="${RERANKER_OUT_DIR}/ce-reranker-doc-${exp}-run.csv"

    #--- Stage 1: weak rerank on RRF-fused top-1000 ---
    #Produces ${weak_output} (TREC run, weak-CE scored, full 1000 per query) and
    #${weak_output%.csv}-rrf.csv (RRF-fused baseline).
    if [ ! -f "${weak_output}" ]; then
        echo "Stage 1 (weak): ${exp}"
        python ./src/classical/Reranker.py \
            --mode document \
            --queries_path "${queries}" \
            --query_key DENSE_query \
            --output "${weak_output}" \
            --run_name "ce-reranker-doc-${exp}-weak" \
            --top_k ${TOP_K} \
            --batch_size ${WEAK_BATCH_SIZE} \
            --max_length ${WEAK_MAX_LENGTH} \
            --rrf_k 60 \
            --doc_runs \
                "${BM25_DIR}/base-bm25-${exp}-run.csv" \
                "${DENSE_DIR}/dense-harrier-medium-${exp}-run-top3.csv" \
                "${SPLADE_DIR}/splade-passage-${exp}-run-top3.csv" \
            --doc_corpus "${DOC_CORPUS}" \
            --model "${WEAK_MODEL}" \
            --model_type qwen_yesno
    else
        echo "Stage 1 (weak): cached, skipping (${weak_output})"
    fi

    #--- Stage 2: strong rerank on weak's top-100 survivors ---
    #Reads ${weak_output} as a single "retriever". Single-element RRF degenerates
    #to pass-through (preserves weak-CE rank order). _load_prepassages caps at
    #top_k=100 per query -> reads weak run's top-100 by rank.
    echo "Stage 2 (strong): ${exp}"
    python ./src/classical/Reranker.py \
        --mode document \
        --queries_path "${queries}" \
        --query_key DENSE_query \
        --output "${final_output}" \
        --run_name "ce-reranker-doc-${exp}" \
        --top_k ${SURVIVORS} \
        --batch_size ${STRONG_BATCH_SIZE} \
        --max_length ${STRONG_MAX_LENGTH} \
        --rrf_k 60 \
        --doc_runs "${weak_output}" \
        --doc_corpus "${DOC_CORPUS}" \
        --model "${STRONG_MODEL}" \
        --model_type qwen_yesno

done

echo "Done."
