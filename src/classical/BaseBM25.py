from Experiment import Experiment, parse_arguments
import pyterrier as pt
from jsonlHandler import iter_jsonl, transform_raw
import pandas as pd
import json
from tsvHandler import tsv_corpus_generator
import numpy as np

class BM25(Experiment):
    '''Base BM25, only applying standard Terrier tokenization/stemming/stopwords and only using BM25 for ranking'''
    
    def __init__(self, index_path: str, corpus_path: str= ''):
        super().__init__(index_path, corpus_path)
        
        self.name = 'base-bm25'
        self.search_pipeline = pt.rewrite.tokenise() >> pt.terrier.Retriever(
            self.index,
            wmodel='BM25',
            threads=32, 
            verbose=True,
            controls={
                "bm25.b": 0.5,
                "bm25.k_1": 0.5,
                "bm25.k_3": 5.0,
            },
        )
    
    def get_index(self, index_path: str):
        return pt.IndexFactory.of(f"{index_path}/data.properties", memory=True)
    
    def build_index(self, index_path: str, corpus_path: str):
        #Index raw data
        if corpus_path.endswith('.jsonl'):
            gen = iter_jsonl(corpus_path, transform_raw, skip_duplicate_docnos=True)
        else:
            gen = tsv_corpus_generator(corpus_path)
        
        try:
            return (pt.IterDictIndexer(index_path, meta={'docno': 20}, blocks=True, threads=32, meta_reverse=['docno'])
                        .index(gen))
        except ValueError as e:
            print(f"Index already built: {e}")
            return self.get_index(index_path)

    def results_tests(
        self, 
        test_query_path: str, 
        out_dir: str, 
        experiment_name: str,
        num_docs: int=1000,
        query_key: str='query',
    ):
        with open(test_query_path, 'r') as inp:
            qids, queries = [], []
            for line in inp:
                obj = json.loads(line)
                qids.append(obj['query_id'])
                queries.append(obj[query_key])
                
        topics = pd.DataFrame({'qid': qids, 'query': queries})
        topics = topics.astype({'qid': str, 'query': str})

        df_ans = (self.search_pipeline % num_docs).transform(topics)
        df_ans = df_ans.sort_values(['qid', 'rank']) #Sort by rank, just in case

        formatted = df_ans[['qid', 'docno', 'rank', 'score']].copy() #Prevent warning
        formatted['Q'] = 'Q0'
        formatted['run_id'] = f'{self.name}'
  
        formatted.to_csv(f'{out_dir}/{self.name}-{experiment_name}-run.csv', index=False, columns=['qid', 'Q', 'docno', 'rank', 'score', 'run_id'])
        

#Example usage
if __name__ == '__main__':
    args = parse_arguments()
    bm = BM25(args.index_path, args.corpus_path)

    for exp_name, queries in zip(args.experiment_names, args.queries_paths):
        bm.results_tests(queries, args.out_dir, exp_name, 1000, args.query_key)