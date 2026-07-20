import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import csv
import json
import math
import torch
import pandas as pd
import pyterrier_alpha as pta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
from collections import defaultdict
from tqdm import tqdm
import argparse


class Reranker:
    def __init__(self, model_name="nvidia/llama-nemotron-rerank-1b-v2", model_type="seq_cls",
                 batch_size=128, max_length=1024, device="cuda", pool="max", pool_k=3):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pool = pool
        self.pool_k = pool_k
        self.model_type = model_type

        if model_type == "qwen_yesno":
            #Qwen3-Reranker ships its own chat/prompt template + yes/no logit-diff scoring.
            #Task-specific instruction (model card recommends tailoring it for ~1-5% gain).
            self.model = CrossEncoder(
                model_name, max_length=max_length, device=device,
                trust_remote_code=True,
                automodel_args={
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                },
                tokenizer_kwargs={"padding_side": "left"},
                prompts={"tot": "Given a user's description of a document they partially remember, retrieve the matching document passage"},
                default_prompt_name="tot",
            )
            if self.model.tokenizer.pad_token is None:
                self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.model.tokenizer.pad_token_id
        elif model_type == "cross_encoder":
            #Generic sentence-transformers CrossEncoder (e.g. BGE-reranker-base).
            #Same scoring path as qwen_yesno (self.model.predict) but no chat template.
            self.model = CrossEncoder(
                model_name, max_length=max_length, device=device,
                trust_remote_code=True,
                automodel_args={
                    "torch_dtype": torch.bfloat16,
                    #"attn_implementation": "flash_attention_2",
                },
                tokenizer_kwargs={"padding_side": "left"},
            )
            if self.model.tokenizer.pad_token is None:
                self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.model.tokenizer.pad_token_id
        elif model_type == "seq_cls":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
            self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
            ).to(self.device).eval()
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
        else:
            raise ValueError(f"unknown model_type {model_type!r} (expected 'seq_cls', 'qwen_yesno', or 'cross_encoder')")

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

    def _stream_doc_texts(self, jsonl_path, needed_ids):
        "Stream a TREC-TOT corpus JSONL, keeping only doc IDs in `needed_ids`."

        needed = {str(x) for x in needed_ids}
        texts = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                did = str(obj['id'])
                if did in needed:
                    title = obj.get('title') or ''
                    body = obj.get('text') or ''
                    texts[did] = f"{title}\n\n{body}" if title else body
                    if len(texts) == len(needed):
                        break
        missing = needed - set(texts)
        if missing:
            print(f"[warn] _stream_doc_texts: {len(missing)} doc IDs not found in corpus (first few: {list(missing)[:5]})")
        return texts

    @staticmethod
    def _rrf(per_query_runs, k=60, num_results=1000):
        "Reciprocal Rank Fusion via pyterrier_alpha.fusion.rr_fusion."

        dfs = []
        for r in per_query_runs:
            rows = []
            for qid, docnos in r.items():
                for rank, docno in enumerate(docnos):
                    rows.append({
                        "qid": str(qid),
                        "query": str(qid),
                        "docno": str(docno),
                        "rank": rank,
                        "score": float(num_results - rank),  #monotonic; RRF only uses rank
                    })
            dfs.append(pd.DataFrame(rows))

        fused_df = pta.fusion.rr_fusion(*dfs, num_results=num_results, k=k)

        fused = {}
        for _, row in fused_df.iterrows():
            qid = str(row["qid"])
            docno = str(row["docno"])
            fused.setdefault(qid, {})[docno] = float(row["score"])

        return {
            qid: dict(sorted(docs.items(), key=lambda x: x[1], reverse=True))
            for qid, docs in fused.items()
        }

    def _cross_encode(self, batch, retriever_tag=None):
        "Applies the cross encoder to a batch of (query, passage) pairs"

        if self.model_type in ("qwen_yesno", "cross_encoder"):
            pairs = [(p['query'], p['text']) for p in batch]
            scores = self.model.predict(pairs, batch_size=self.batch_size,
                                         activation_fn=torch.nn.Identity(), convert_to_numpy=True)
            return scores.tolist()

        # seq_cls (nemotron) path
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
            if self.model_type == "qwen_yesno":
                doc_scores = {
                    doc_id: sum(sorted(scores, reverse=True)[:self.pool_k])
                    for doc_id, scores in per_doc.items()
                }
            else:
                doc_scores = {
                    doc_id: sum(1.0 / (1.0 + math.exp(-s)) for s in sorted(scores, reverse=True)[:self.pool_k])
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

    def rerank_retriever(self, prepassage_csv, mapping_path, tsv_path, query_lookup, top_k, retriever_tag=None):
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
            scores = self._cross_encode(batch, retriever_tag=retriever_tag)
            for item, score in zip(batch, scores):
                all_scores.append({**item, "ce_score": float(score)})

        #Route new scored pairs
        scored_by_qid = defaultdict(list)
        for item in all_scores:
            scored_by_qid[item['qid']].append(item)

        #Return scored passages + mapping (pooling applied later)
        return dict(scored_by_qid), mapping

    def rerank_documents(self, doc_run_csvs, corpus_path, query_lookup, top_k,
                         rrf_k=60, rrf_output=None, rrf_run_name="rrf"):
        "Document-level reranker: RRF-fuse doc-level TREC runs, then cross-encode (query, full_doc)."

        per_retriever = [self._load_prepassages(p, top_k) for p in doc_run_csvs]

        fused = self._rrf(per_retriever, k=rrf_k, num_results=top_k)

        if rrf_output:
            os.makedirs(os.path.dirname(rrf_output) or ".", exist_ok=True)
            self.save_trec(fused, rrf_output, rrf_run_name)
            print(f"  RRF-fused run saved to {rrf_output}")

        all_needed = set()
        for docnos in fused.values():
            all_needed.update(docnos)

        doc_texts = self._stream_doc_texts(corpus_path, all_needed)

        pairs_by_qid = {}
        for qid, docnos in tqdm(fused.items(), desc="Building doc pairs"):
            if qid not in query_lookup:
                continue
            pairs_by_qid[qid] = [
                {
                    "qid": qid,
                    "query": query_lookup[qid],
                    "doc_id": did,
                    "text": doc_texts.get(did, ""),
                    "text_label": "document",
                }
                for did in docnos if did in doc_texts
            ]

        #Flatten + batch + score.
        all_pairs = [p for pairs in pairs_by_qid.values() for p in pairs]
        #Length-bucketed batching: sort by text length descending so each batch
        #is internally length-homogeneous. Eliminates padding waste (a 12k-token
        #doc in a random batch forces all 8 sequences to pad to 12k). Longest
        #first also fails fast on OOM. The output set is unchanged — only the
        #iteration order changes, so downstream aggregation still works.
        all_pairs.sort(key=lambda p: len(p['text']), reverse=True)
        all_scores = []
        for i in tqdm(range(0, len(all_pairs), self.batch_size), desc="Cross-encoding"):
            batch = all_pairs[i:i + self.batch_size]
            scores = self._cross_encode(batch)
            for item, score in zip(batch, scores):
                all_scores.append({**item, "ce_score": float(score)})

        ranked = {}
        for item in all_scores:
            ranked.setdefault(item['qid'], {})[item['doc_id']] = item['ce_score']

        return {
            qid: dict(sorted(docs.items(), key=lambda x: x[1], reverse=True))
            for qid, docs in ranked.items()
        }

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
        "Cache CE-scored passages (pool-agnostic) in TREC run format: qid Q0 docno rank score run_id"
        rows = []
        for qid, items in scored_by_qid.items():
            ranked = sorted(items, key=lambda x: x['ce_score'], reverse=True)
            for rank, item in enumerate(ranked):
                rows.append({
                    "qid": qid, "Q": "Q0", "docno": item['passage_id'],
                    "rank": rank, "score": item['ce_score'], "run_id": run_name,
                })
        pd.DataFrame(rows).to_csv(
            output_path, index=False,
            columns=['qid', 'Q', 'docno', 'rank', 'score', 'run_id'],
        )

    @staticmethod
    def load_passages(csv_path):
        "Load cached CE-scored passages (TREC format) into qid -> [items] lookup"
        df = pd.read_csv(csv_path, dtype={'qid': 'str', 'docno': 'str'})
        scored = defaultdict(list)
        for _, row in df.iterrows():
            scored[row['qid']].append({"passage_id": row['docno'], "ce_score": float(row['score'])})
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
    parser.add_argument('--mode', choices=['passage', 'document'], default='passage',
                        help="'passage' (default, legacy) reranks passages then pools; "
                             "'document' RRF-fuses doc-level runs and cross-encodes (query, full_doc) directly.")
    parser.add_argument('--queries_path', required=True, help='JSONL with queries')
    parser.add_argument('--query_key', default='DENSE_query', help='Query field for CE input')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--run_name', required=True, help='TREC run_id')
    parser.add_argument('--model', default='nvidia/llama-nemotron-rerank-1b-v2',
                        help='Cross-encoder checkpoint (HF id or local path)')
    parser.add_argument('--model_type', choices=['seq_cls', 'qwen_yesno', 'cross_encoder'], default='seq_cls',
                        help="'seq_cls' (nemotron, scalar head), 'qwen_yesno' (Qwen3-Reranker via sentence-transformers), "
                             "or 'cross_encoder' (generic sentence-transformers CrossEncoder, e.g. BGE-reranker-base)")
    parser.add_argument('--top_k', type=int, default=1000, help='Passages/documents per query per retriever')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Max sequence length for the cross-encoder. Use 8192+ for --mode document.')
    parser.add_argument('--pool', choices=['max', 'top3'], default='max',
                        help='Passage -> document pooling (passage mode only): "max" or "top3"')
    parser.add_argument('--pool_k', type=int, default=3, help='k for top-k sum pooling (passage mode only)')
    #Passage-mode inputs
    parser.add_argument('--passage_runs', nargs='+', default=None, help='Prepassage CSV files (passage mode)')
    parser.add_argument('--mappings', nargs='+', default=None, help='idx_to_pid JSON files (passage mode)')
    parser.add_argument('--corpora', nargs='+', default=None, help='Passage TSV files (passage mode)')
    parser.add_argument('--names', nargs='+', default=None, help='Retriever labels (passage mode)')
    #Document-mode inputs
    parser.add_argument('--doc_runs', nargs='+', default=None,
                        help='Document-level TREC run CSVs to RRF-fuse then rerank (document mode)')
    parser.add_argument('--doc_corpus', default=None,
                        help='Document-level corpus JSONL with {id, title, text} (document mode)')
    parser.add_argument('--rrf_k', type=int, default=60, help='RRF k (document mode)')
    args = parser.parse_args()

    reranker = Reranker(model_name=args.model, model_type=args.model_type,
                     batch_size=args.batch_size, max_length=args.max_length,
                     pool=args.pool, pool_k=args.pool_k)
    query_lookup = reranker._load_queries(args.queries_path, args.query_key)

    if args.mode == "document":
        if not args.doc_runs or not args.doc_corpus:
            parser.error("--mode document requires --doc_runs and --doc_corpus")

        print("\n=== Document-level rerank ===")
        rrf_output = args.output.replace(".csv", "-rrf.csv")
        rrf_run_name = f"{args.run_name}-rrf"

        final = reranker.rerank_documents(
            args.doc_runs, args.doc_corpus, query_lookup, args.top_k,
            rrf_k=args.rrf_k, rrf_output=rrf_output, rrf_run_name=rrf_run_name,
        )
        reranker.save_trec(final, args.output, args.run_name)
        print(f"Saved to {args.output}")
        #End of document-mode path.
        raise SystemExit(0)

    #--- passage mode (legacy) ---
    if not args.passage_runs or not args.mappings or not args.corpora:
        parser.error("--mode passage requires --passage_runs, --mappings, --corpora")

    n = len(args.passage_runs)
    assert len(args.mappings) == n == len(args.corpora), \
        "--passage_runs, --mappings, --corpora must have equal length"

    names = args.names if args.names else [f"r{i}" for i in range(n)]

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
                query_lookup, args.top_k, retriever_tag=names[i]
            )
            reranker.save_passages(scored_by_qid, per_run, run_id)
            print(f"  -> {per_run}")

        #Pool passages -> documents (applied post-cache, so pool method is cheap to change)
        pooled = {qid: reranker._pool(items, mapping) for qid, items in scored_by_qid.items()}
        all_pooled.append(pooled)

    print("\n=== Merging ===")
    merged = reranker.merge(*all_pooled) if len(all_pooled) > 1 else all_pooled[0]
    merged_output = args.output.replace(".csv", "-merged.csv")
    reranker.save_trec(merged, merged_output, args.run_name)
    print(f"Saved to {merged_output}")
