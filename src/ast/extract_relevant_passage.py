import os
# Ensure this is set before importing torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import concurrent.futures

import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import tqdm

from src.utils.DocumentChunker import DocumentChunker
from .filter_dataset import filter_corpus

# ── Paths ────────────────────────────────────────────────────────────────────
CORPUS_PATH          = './dataset/corpus-cleaned-sections.jsonl'
FILTERED_CORPUS_PATH = './dataset/ast/filtered-corpus.jsonl'
QUERIES_PATH         = './queries/sft-train/rewritten-queries-sft.jsonl'
QRELS_PATH           = './queries/sft-train/qrels.txt'
OUTPUT_PATH          = './queries/sft-train/golden-passages-sft.jsonl'

# ── Tunables ─────────────────────────────────────────────────────────────────
NUM_CHUNK_PROCS  = 24   # CPU workers for the chunking phase
MAX_PASSAGES     = None # None → keep every ranked passage; set an int to cap


# ── CPU worker: chunk one document ───────────────────────────────────────────

def chunk_row_worker(args):
    """
    Runs in a subprocess. Re-instantiates the tokenizer/chunker per process
    to avoid pickling issues with HuggingFace tokenizers.
    """
    doc_id, query_id, query_text, title, sections = args

    try:
        chunk_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/harrier-oss-v1-270m", use_fast=True, local_files_only=True
        )
    except Exception:
        # Fallback to online if it hasn't cached yet (only happens on the first few hits)
        print("no cache fallback")
        chunk_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/harrier-oss-v1-270m", use_fast=True
        )
    chunker = DocumentChunker(tokenizer=chunk_tokenizer)
    passages = chunker.chunk_document(title, sections)

    return [
        {
            "query_id": query_id,
            "query":    query_text,
            "doc_id":   doc_id,
            "passage":  passage,
            "p_index":  i,
        }
        for i, passage in enumerate(passages)
    ]


# ── Grouping Helper ──────────────────────────────────────────────────────────

def group_pairs_by_query(flat_pairs: list[dict]) -> dict[str, list[dict]]:
    """Group flat (query_id, doc_id, query, passage) records by query_id."""
    groups: dict[str, list[dict]] = {}
    for item in flat_pairs:
        groups.setdefault(item["query_id"], []).append(item)
    return groups


# ── GPU scoring phase ─────────────────────────────────────────────────────────

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def score_all_pairs(
    flat_pairs: list[dict],
    tokenizer,
    model,
    device: str,
    batch_size: int = 32, # Adjust based on your VRAM
    max_length: int = 1024
) -> list[dict]:
    """
    Process pairs in standard pointwise batches. 
    Nemotron evaluates each (query, passage) pair independently.
    """
    scored_all: list[dict] = []
    
    # Process in standard pointwise batches across the entire dataset
    for i in tqdm.tqdm(range(0, len(flat_pairs), batch_size), desc="GPU Reranking (Pointwise)"):
        batch_items = flat_pairs[i : i + batch_size]
        
        # Nemotron specifically expects this prompt template format
        texts = [
            f"question:{item['query']} \n \n passage:{item['passage']}" 
            for item in batch_items
        ]
        
        # Tokenize and pad the batch
        batch_dict = tokenizer(
            texts, 
            padding=True, 
            truncation=True,
            max_length=max_length, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**batch_dict)
            
            # The model outputs a single raw logit for relevance per pair
            scores = outputs.logits.squeeze(-1).float().cpu().numpy()
            
            # Handle edge case where batch_size=1 makes squeeze() return a scalar
            if scores.ndim == 0:
                scores = [scores.item()]
                
        for item, score in zip(batch_items, scores):
            scored_all.append({**item, "score": float(score)})
            
    return scored_all


# ── Aggregate scores → per-(query, doc) ranked lists ─────────────────────────

def aggregate(scored_pairs: list[dict], max_passages: int | None) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for item in scored_pairs:
        key = (item["query_id"], item["doc_id"])
        groups.setdefault(key, []).append(item)

    for key, items in groups.items():
        items.sort(key=lambda x: x["score"], reverse=True)
        if max_passages is not None:
            groups[key] = items[:max_passages]

    return groups


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load queries + qrels
    print("Loading queries and target document IDs...")
    queries = pd.read_json(QUERIES_PATH, lines=True)
    qrels   = pd.read_csv(
        QRELS_PATH, sep=" ", header=None,
        names=["query_id", "zero", "doc_id", "one"],
    )
    
    merged = queries.merge(qrels, how="inner", on="query_id").set_index("doc_id", drop=False)
    merged["doc_id"] = merged["doc_id"].astype(str)
    merged = merged.drop_duplicates(subset=["doc_id"])
    
    target_doc_ids = set(merged["doc_id"])
    print(f"Unique doc IDs: {len(target_doc_ids)} | Rows: {len(merged)}")

    # 2. Filter corpus
    filtered_corpus_df = filter_corpus(target_doc_ids, CORPUS_PATH, FILTERED_CORPUS_PATH)
    filtered_corpus_df["id"] = filtered_corpus_df["id"].astype(str)
    corpus_dict = filtered_corpus_df.set_index("id").to_dict(orient="index")

    tasks = [
        (
            row["doc_id"],
            row["query_id"],
            row["DENSE_query"],
            corpus_dict[row["doc_id"]].get("title", ""),
            corpus_dict[row["doc_id"]].get("sections", []),
        )
        for _, row in merged.iterrows()
        if row["doc_id"] in corpus_dict
    ]

    print(f"Phase 1 - Chunking {len(tasks)} documents on {NUM_CHUNK_PROCS} CPU cores...")
    flat_pairs: list[dict] = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CHUNK_PROCS) as pool:
        for chunk_list in tqdm.tqdm(pool.map(chunk_row_worker, tasks), total=len(tasks), desc="Chunking"):
            flat_pairs.extend(chunk_list)

    print(f"Total (query, passage) pairs to score: {len(flat_pairs):,}")

    # 4. Load tokenizer and model on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 2 - Loading Nemotron-1B on {device}...")
    
    model_name = "nvidia/llama-nemotron-rerank-1b-v2"
    
    # Nemotron requires left-padding configuration
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load as a Sequence Classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
        
    model = model.to(device).eval()

    # 5. GPU: Score pairs in standard batches
    # Make sure you pass the tokenizer and device now
    scored_pairs = score_all_pairs(
        flat_pairs=flat_pairs, 
        tokenizer=tokenizer, 
        model=model, 
        device=device,
        batch_size=32 # Drop to 16 if 32 OOMs depending on max_length
    )

    # 6. Aggregate into per-(query, doc) ranked lists
    print("Aggregating and ranking passages per (query, doc) pair...")
    ranked = aggregate(scored_pairs, MAX_PASSAGES)

    # 7. Write output
    print(f"Writing results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for _, row in merged.iterrows():
            doc_id   = row["doc_id"]
            query_id = row["query_id"]

            if doc_id not in corpus_dict:
                continue

            ranked_passages = ranked.get((query_id, doc_id), [])
            golden_passages = [
                {"passage": p["passage"], "score": p["score"]}
                for p in ranked_passages
            ]

            record = {
                "query_id":       query_id,
                "query":          row["DENSE_query"],
                "doc_id":         doc_id,
                "golden_passages": golden_passages,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Done. Golden passages saved to {OUTPUT_PATH}")