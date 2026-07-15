import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
import json
import torch
import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .filter_dataset import filter_passage
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
TREC_RUN_PATH  = './results/rewrite-sft/bm25-passage/base-bm25-sft-train-run.csv'
#TREC_RUN_PATH  = './sample.csv'
QRELS_PATH     = './queries/sft-train/qrels.txt'
QUERIES_PATH   = './queries/sft-train/rewritten-queries-sft.jsonl'
MAPPING_PATH   = './dataset/harrier-oss-600/cleaned-idx-to-pid-harrier-600-overlap.json'
TSV_PATH       = './dataset/harrier-oss-600/cleaned-corpus-harrier-600-tokenized-overlap.tsv'
OUTPUT_CSV     = './queries/sft-train/negatives/bm25-passage.csv'
FILTERED_PASSAGE_OUTPUT = './dataset/ast/bm25-filtered-passages.jsonl'
QUERY_KEY   = 'DENSE_query'

# ── Tunables ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 128
MAX_LENGTH     = 800


def run_streaming_reranker():
    # 1. Load ground-truth Qrels to map queries to their positive documents
    print("Loading target validation structures...")
    queries = pd.read_json(QUERIES_PATH, lines=True)
    qrels   = pd.read_csv(
        QRELS_PATH, sep=" ", header=None,
        names=["query_id", "zero", "doc_id", "one"],
    )
    qrels = queries.merge(qrels, how="inner", on="query_id").set_index("doc_id", drop=False)
    qrels["doc_id"] = qrels["doc_id"].astype(str)
    qrels["query_id"] = qrels["query_id"].astype(str)
    true_docs = qrels.set_index("query_id")["doc_id"].to_dict()
    qrels = qrels.set_index('query_id', drop=False)

    # 2. Load the input run file
    trec_run = pd.read_csv(TREC_RUN_PATH, sep=",")
    
    # 3. Load the flat JSON passage-to-document tracking array
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        idx_to_pid = json.load(f)

    # 4. First Pass: Parse candidates and filter out target positive documents
    print("Evaluating candidate passages and purging true positives...")
    candidates = [] # Elements: (qid, query_text, passage_id, run_id)
    needed_passage_ids = set()

    for _, row in trec_run.iterrows():
        qid = str(row["qid"])
        p_id = int(row["docno"])
        
        if p_id >= len(idx_to_pid):
            continue
            
        parent_doc = str(idx_to_pid[p_id])
        true_doc = true_docs.get(qid)

        # Pure Negative Enforcement: Skip if passage belongs to the golden doc
        if parent_doc == true_doc:
            continue
        
        candidates.append({
            "qid": qid,
            "query": qrels.loc[qid][QUERY_KEY],
            "passage_id": str(p_id),
            "run_id": f"{row['run_id']}-nemotron-reranked"
        })
        needed_passage_ids.add(str(p_id))
    
    print(candidates[:3])

    # 5. Second Pass: Stream the heavy TSV to pull required candidate texts
    print(f"Streaming text rows from TSV for {len(needed_passage_ids)} target candidates...")
    passages = filter_passage(needed_passage_ids, TSV_PATH, FILTERED_PASSAGE_OUTPUT)
    passages['docno'] = passages['docno'].astype(str)
    passages = passages.set_index('docno', drop=False)

    print(passages)

    # Attach loaded text strings to our tracking elements
    for item in candidates:
        item["passage_text"] = passages.loc[item["passage_id"]]

    # 6. Initialize Nemotron-1B Model on the GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "nvidia/llama-nemotron-rerank-1b-v2"
    print(f"Initializing {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right" # Protect query template strings from clipping

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    # 7. Execute Pointwise Inference
    print(f"Computing scores over {len(candidates)} records...")
    scored_elements = []

    for i in tqdm(range(0, len(candidates), BATCH_SIZE), desc="Inference Batches"):
        batch = candidates[i : i + BATCH_SIZE]
        
        texts = [
            f"question:{item['query']} \n \n passage:{item['passage_text']}" 
            for item in batch
        ]

        batch_dict = tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**batch_dict)
            scores = outputs.logits.squeeze(-1).float().cpu().numpy()
            if scores.ndim == 0:
                scores = [scores.item()]

        for item, score in zip(batch, scores):
            item["score"] = float(score)
            scored_elements.append(item)

    # 8. Sort and Assign Ranks Per Query Group
    print("Regrouping collections to calculate updated rank positions...")
    df_scored = pd.DataFrame(scored_elements)
    
    # Sort descending by calculated score within each distinct query group
    df_scored = df_scored.sort_values(by=["qid", "score"], ascending=[True, False])
    df_scored["rank"] = df_scored.groupby("qid").cumcount()

    # 9. Output to clean TREC CSV Format
    output_df = pd.DataFrame({
        "qid": df_scored["qid"],
        "Q": 'Q0',
        "docno": df_scored["passage_id"],
        "rank": df_scored["rank"],
        "score": df_scored["score"],
        "run_id": df_scored["run_id"]
    })

    output_df.to_csv(OUTPUT_CSV, index=False, sep=",")
    print(f"Processing successful! Output run file saved at: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_streaming_reranker()