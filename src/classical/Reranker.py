import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import argparse


class Reranker:
    def __init__(self, model_name="nvidia/llama-nemotron-rerank-1b-v2", batch_size=128, max_length=1024, device="cuda", pool="max", pool_k=3):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pool = pool
        self.pool_k = pool_k

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(self.device).eval()

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _load_queries(self, queries_path, query_key):
        "Reads queries.jsonl and builds pairs of query_id <-> query, from `query_key`"
    
        df = pd.read_json(queries_path, lines=True, dtype={'query_id': 'str'})
        return dict(zip(df['query_id'], df[query_key]))

    def _load_prepassages(self, csv_path, top_k):
        "Loads prepassage results: raw chunked passages from documents"

        df = pd.read_csv(csv_path, dtype={'qid': 'str', 'docno': 'str'})
        per_query = defaultdict(list)
        for _, row in df.iterrows():
            if len(per_query[row['qid']]) < top_k:
                per_query[row['qid']].append(row['docno'])
        return dict(per_query)

    def _load_mapping(self, mapping_path):
        "Load passage -> document mappings"

        with open(mapping_path, 'r') as f:
            return json.load(f)

    def _stream_passage_texts(self, tsv_path, needed_ids):
        "Get actual passage texts"

        texts = {}
        with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
            for row in csv.reader(f, delimiter='\t'):
                if row[0] in needed_ids:
                    texts[row[0]] = row[1]
                    if len(texts) == len(needed_ids):
                        break
        return texts

    def _cross_encode(self, batch):
        "Applies the cross encoder to a batch of texts"

        texts = [f"question:{p['query']} \n \n passage:{p['text']}" for p in batch]
        batch_dict = self.tokenizer(texts, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            scores = self.model(**batch_dict).logits.view(-1).cpu().tolist()
        return scores

    def _pool(self, per_query_scored, mapping):
        "Pool passages -> documents"

        if self.pool == "top3":
            per_doc = defaultdict(list)
            for item in per_query_scored:
                pid = int(item['passage_id'])
                doc_id = str(mapping[pid]) if pid < len(mapping) else str(pid)
                per_doc[doc_id].append(item['ce_score'])
            doc_scores = {
                doc_id: sum(sorted(scores, reverse=True)[:self.pool_k])
                for doc_id, scores in per_doc.items()
            }
        else:  # max
            doc_scores = {}
            for item in per_query_scored:
                pid = int(item['passage_id'])
                doc_id = str(mapping[pid]) if pid < len(mapping) else str(pid)
                s = item['ce_score']
                if s > doc_scores.get(doc_id, float('-inf')):
                    doc_scores[doc_id] = s
        return dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True))

    def rerank_retriever(self, prepassage_csv, mapping_path, tsv_path, query_lookup, top_k):
        # Get query -> passagenos mapping
        per_query = self._load_prepassages(prepassage_csv, top_k)

        all_needed = set()
        for pids in per_query.values(): #Determine which passages to fetch
            all_needed.update(pids)
        #Fetch + load mapping
        passage_texts = self._stream_passage_texts(tsv_path, all_needed)
        mapping = self._load_mapping(mapping_path)

        #Build qid -> (qid, query, passage_id, text) mapping
        pairs_by_qid = {}
        for qid, pids in tqdm(per_query.items(), desc="Building pairs"):
            if qid not in query_lookup:
                continue
            pairs_by_qid[qid] = [
                {"qid": qid, "query": query_lookup[qid], "passage_id": pid, "text": passage_texts.get(pid, "")}
                for pid in pids if pid in passage_texts
            ]

        #Flatten pairs + batch + score
        all_pairs = [p for pairs in pairs_by_qid.values() for p in pairs]
        all_scores = []
        for i in tqdm(range(0, len(all_pairs), self.batch_size), desc="Cross-encoding"):
            batch = all_pairs[i:i + self.batch_size]
            scores = self._cross_encode(batch)
            for item, score in zip(batch, scores):
                all_scores.append({**item, "ce_score": float(score)})

        #Route new scored pairs
        scored_by_qid = defaultdict(list)
        for item in all_scores:
            scored_by_qid[item['qid']].append(item)

        #Return scored passages + mapping (pooling applied later)
        return dict(scored_by_qid), mapping

    @staticmethod
    def merge(*retriever_scores):
        all_qids = set()
        for r in retriever_scores:
            all_qids.update(r.keys())

        merged = {}
        for qid in all_qids:
            doc_scores = {}
            for r in retriever_scores:
                for doc_id, score in r.get(qid, {}).items():
                    if score > doc_scores.get(doc_id, float('-inf')):
                        doc_scores[doc_id] = score
            merged[qid] = dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True))
        return merged

    @staticmethod
    def save_passages(scored_by_qid, output_path, run_name):
        "Cache CE-scored passages (pool-agnostic) to CSV"
        rows = []
        for qid, items in scored_by_qid.items():
            for item in items:
                rows.append({"qid": qid, "passage_id": item['passage_id'], "ce_score": item['ce_score']})
        pd.DataFrame(rows).to_csv(output_path, index=False, columns=['qid', 'passage_id', 'ce_score'])

    @staticmethod
    def load_passages(csv_path):
        "Load cached CE-scored passages into qid -> [items] lookup"
        df = pd.read_csv(csv_path, dtype={'qid': 'str', 'passage_id': 'str'})
        scored = defaultdict(list)
        for _, row in df.iterrows():
            scored[row['qid']].append({"passage_id": row['passage_id'], "ce_score": float(row['ce_score'])})
        return dict(scored)

    @staticmethod
    def save_trec(ranking, output_path, run_name, top_n=1000):
        rows = []
        for qid, docs in ranking.items():
            for rank, (doc_id, score) in enumerate(docs.items()):
                if rank >= top_n:
                    break
                rows.append({"qid": qid, "Q": "Q0", "docno": doc_id, "rank": rank, "score": score, "run_id": run_name})
        pd.DataFrame(rows).to_csv(output_path, index=False,
                                  columns=['qid', 'Q', 'docno', 'rank', 'score', 'run_id'])

    @staticmethod
    def load_trec(csv_path):
        df = pd.read_csv(csv_path, dtype={'qid': 'str', 'docno': 'str'})
        ranking = {}
        for _, row in df.iterrows():
            ranking.setdefault(row['qid'], {})[row['docno']] = row['score']
        return ranking


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-encoder reranker with multi-retriever score merging")
    parser.add_argument('--queries_path', required=True, help='JSONL with queries')
    parser.add_argument('--query_key', default='DENSE_query', help='Query field for CE input')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--run_name', required=True, help='TREC run_id')
    parser.add_argument('--top_k', type=int, default=1000, help='Passages per query per retriever')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pool', choices=['max', 'top3'], default='max',
                        help='Passage -> document pooling: "max" (best passage) or "top3" (sum of top-k passages)')
    parser.add_argument('--pool_k', type=int, default=3, help='k for top-k sum pooling (only used with --pool top3)')
    parser.add_argument('--passage_runs', nargs='+', required=True, help='Prepassage CSV files')
    parser.add_argument('--mappings', nargs='+', required=True, help='idx_to_pid JSON files')
    parser.add_argument('--corpora', nargs='+', required=True, help='Passage TSV files')
    parser.add_argument('--names', nargs='+', default=None, help='Retriever labels')
    args = parser.parse_args()

    n = len(args.passage_runs)
    assert len(args.mappings) == n == len(args.corpora), \
        "--passage_runs, --mappings, --corpora must have equal length"

    names = args.names if args.names else [f"r{i}" for i in range(n)]

    reranker = Reranker(batch_size=args.batch_size, max_length=1024, pool=args.pool, pool_k=args.pool_k)
    query_lookup = reranker._load_queries(args.queries_path, args.query_key)

    all_pooled = []
    for i in range(n):
        base = args.output.replace(".csv", "")
        per_run = f"{base}-{names[i]}.csv"
        run_id = f"{args.run_name}-{names[i]}"

        if os.path.exists(per_run):
            print(f"\n=== {names[i]}: cached passages, loading from {per_run} ===")
            scored_by_qid = reranker.load_passages(per_run)
            mapping = reranker._load_mapping(args.mappings[i])
        else:
            print(f"\n=== Reranking: {names[i]} ===")
            scored_by_qid, mapping = reranker.rerank_retriever(
                args.passage_runs[i], args.mappings[i], args.corpora[i],
                query_lookup, args.top_k
            )
            reranker.save_passages(scored_by_qid, per_run, run_id)
            print(f"  -> {per_run}")

        #Pool passages -> documents (applied post-cache, so pool method is cheap to change)
        pooled = {qid: reranker._pool(items, mapping) for qid, items in scored_by_qid.items()}
        all_pooled.append(pooled)

    print("\n=== Merging ===")
    merged = reranker.merge(*all_pooled) if len(all_pooled) > 1 else all_pooled[0]
    reranker.save_trec(merged, args.output, args.run_name)
    print(f"Saved to {args.output}")
