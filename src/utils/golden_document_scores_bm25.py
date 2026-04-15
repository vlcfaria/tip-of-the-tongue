import pyterrier as pt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import sys

QUERIES_ROOT = "./queries"
BM25_INDEX_PATH = "./indexes/bm25-cleaned-metadata-skip-duplicates"
CORPUS_PATH = "./corpus.jsonl"
OUTPUT_CSV = "golden_doc_scores_analysis_bm25.csv"

def load_models(bm25_path):
    bm25_index = pt.IndexFactory.of(bm25_path)
    bm25_retriever = pt.rewrite.tokenise() >> pt.terrier.Retriever(bm25_index, wmodel='BM25', threads=1) >> pt.text.get_text(bm25_index, metadata='text')

    return bm25_index, bm25_retriever

def get_golden_documents(root_dir, retriever):
    all_results = []
    
    topic_files = list(Path(root_dir).glob('*/*/*.ner.jsonl'))

    for query_file in tqdm(topic_files, desc="Processing topic files"):
        
        qrel_file = (
            query_file.parent / 'qrel.txt'
        )
            
        with open(query_file, 'r') as inp:
            qids, queries = [], []
            for line in inp:
                obj = json.loads(line)
                qids.append(obj['query_id'])
                queries.append(obj['query'])
                
        topics = pd.DataFrame({'qid': qids, 'query': queries})
        queries_df = topics.astype({'qid': 'object', 'query': 'object'})

        qrels_df = pd.read_csv(qrel_file, usecols=[0, 2, 3], names=['qid', 'docno', 'label'], header=None, sep=' ',
                            dtype={'qid': 'object', 'docno': 'object', 'label': 'int64'})
                    
        queries_with_golden = pd.merge(queries_df, qrels_df, on='qid')

        print(queries_with_golden)

        all_results.append(retriever.transform(queries_with_golden))

    return pd.concat(all_results)

def main():
    model, retriever = load_models(BM25_INDEX_PATH)
    results_df = get_golden_documents(QUERIES_ROOT, retriever)

    print(results_df)
    
    results_df.to_csv(OUTPUT_CSV, sep='\t', index=False)
    
if __name__ == "__main__":
    main()