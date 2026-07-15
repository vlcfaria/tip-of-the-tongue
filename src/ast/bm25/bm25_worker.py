import argparse
import itertools
import json
import os
import fcntl  # Works on Unix-based systems to prevent CSV write collisions
import numpy as np
import pandas as pd
import pyterrier as pt

def evaluate_single_combo(args):
    # 1. Load Index
    index = pt.IndexFactory.of(f"{args.index_path}/data.properties")

    # 2. Parse Qrels
    qrels = pd.read_csv(
        args.qrels_path,
        usecols=[0, 2, 3],
        names=["qid", "docno", "label"],
        header=None,
        sep=" ",
        dtype={"qid": str, "docno": str, "label": "int64"},
    )

    # 3. Parse Topics
    with open(args.topics_path, "r") as inp:
        qids, queries = [], []
        for line in inp:
            obj = json.loads(line)
            qids.append(obj["query_id"])
            queries.append(obj["BM25_query"])

    topics = pd.DataFrame({"qid": qids, "query": queries})
    topics = topics.astype({"qid": str, "query": str})

    # 4. Generate Folds (reproducible seed matching the orchestrator)
    np.random.seed(42)
    unique_qids = topics["qid"].unique()
    np.random.shuffle(unique_qids)
    folds_qids = np.array_split(unique_qids, args.num_folds)

    # 5. Build Pipeline for this specific worker execution
    retriever = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=args.num_docs,
        verbose=False,
        threads=6,
        controls={
            "bm25.b": args.b,
            "bm25.k_1": args.k1,
            "bm25.k_3": args.k3,
        },
    )
    pipeline = pt.rewrite.tokenise() >> retriever

    # 6. Evaluate over folds
    for fold_idx, test_qids in enumerate(folds_qids):
        fold_topics = topics[topics["qid"].isin(test_qids)]
        fold_qrels = qrels[qrels["qid"].isin(test_qids)]

        if fold_topics.empty or fold_qrels.empty:
            continue

        try:
            res = pipeline.transform(fold_topics)
            eval_res = pt.Evaluate(res, fold_qrels, metrics=[args.metric])
            score = eval_res.get(args.metric, 0.0)

            row = pd.DataFrame(
                [
                    {
                        "b": args.b,
                        "k1": args.k1,
                        "k3": args.k3,
                        "fold": fold_idx,
                        args.metric: score,
                    }
                ]
            )

            # 7. Safe Multi-Process Append using file locking
            with open(args.output_csv, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Acquire exclusive lock
                row.to_csv(f, header=False, index=False)
                fcntl.flock(f, fcntl.LOCK_UN)  # Release lock

        except Exception as e:
            print(
                f"Error on Combo b={args.b}, k1={args.k1}, k3={args.k3} | Fold {fold_idx}: {e}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single BM25 configuration worker."
    )
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, default="")
    parser.add_argument("--topics_path", type=str, required=True)
    parser.add_argument("--qrels_path", type=str, required=True)
    parser.add_argument("--b", type=float, required=True)
    parser.add_argument("--k1", type=float, required=True)
    parser.add_argument("--k3", type=float, required=True)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--metric", type=str, default="recall_1000")
    parser.add_argument("--num_docs", type=int, default=1000)
    parser.add_argument("--output_csv", type=str, default="results.csv")

    args = parser.parse_args()
    evaluate_single_combo(args)