import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #same order as nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from Experiment import Experiment, parse_arguments
from tsvHandler import tsv_corpus_generator
import pyterrier as pt
import pyterrier_alpha as pta
import pyt_splade
from pyterrier_pisa import PisaIndex
from huggingface_hub import login
import argparse
from dotenv import load_dotenv
from RankingManager import RankingManager
import json
import pandas as pd
from collections import defaultdict
import csv
load_dotenv()
from typing import Literal
#login(token=os.getenv('HF_TOKEN'))

class SPLADE_PASSAGE(Experiment):
    '''SPLADE experimentation, with passage indexing instead of document indexing'''
    
    def __init__(self, index_path: str, corpus_path: str= '', query_model: str='naver/splade-v3'):
        super().__init__(index_path, corpus_path)
        
        self.name = 'splade-passage'
        self.query_model = query_model
        print("using query model:", query_model)
        self.splade = pyt_splade.Splade(device='cuda:0', model=query_model, max_length=512) #We use 512 max length for query encoding
        self.search_pipeline = None #Search pipeline is dynamic due to dynamic documents to pool
    
    def get_index(self, index_path: str):
        return PisaIndex(index_path, stemmer='none', threads=32) #Multithreading is giving segfaults when searching
    
    def build_index(self, index_path: str, corpus_path: str):
        #max token is a bit lower for indexing
        splade = pyt_splade.Splade(device='cuda:0', model='naver/splade-v3', max_length=256)
        index = PisaIndex(index_path, stemmer='none', threads=32, batch_size=3_000_000)
        idx_pipeline = splade.doc_encoder(batch_size=512) >> index.toks_indexer()
        
        return idx_pipeline.index(tsv_corpus_generator(corpus_path))
    
    def results_tests(
        self, 
        test_query_path: str, 
        out_dir: str, 
        experiment_name: str, 
        pool_fn: Literal['max', 'topk'] | None, 
        doc_id_map_path: str=None, 
        docs_to_pool: int=3000, 
        final_num_docs: int=1000,
        query_key: str='query'
    ):
        with open(test_query_path, 'r') as inp:
            qids, queries = [], []
            for line in inp:
                obj = json.loads(line)
                qids.append(obj['query_id'])
                queries.append(obj[query_key])
                
        topics = pd.DataFrame({'qid': qids, 'query': queries})
        topics = topics.astype({'qid': 'object', 'query': 'object'})

        search_pipeline = self.splade.query_encoder(batch_size=30) >> self.index.quantized(num_results=docs_to_pool, query_algorithm='block_max_maxscore', verbose=True)
        df_ans = search_pipeline.transform(topics)

        #Convert pyterrier dataframe into ranking
        ranking = RankingManager(df_ans, doc_id_map_path)

        #Save pre-passages
        base_path = f'{out_dir}/{self.name}-{experiment_name}'
        ranking.raw_passage_ranking_to_csv(base_path + '-prepassages.csv', f'{self.name}-prepassage')

        #Pool! (and save)
        if (pool_fn == 'max' or pool_fn == None):
            pooled = ranking.max_pooling(final_num_docs)
            ranking.save_rankings_as_trec_csv(pooled, base_path + f'-run-max.csv', f'{self.name}-max')
        if (pool_fn == 'topk' or pool_fn == None):
            pooled = ranking.top_k_sum_pooling(final_num_docs)
            ranking.save_rankings_as_trec_csv(pooled, base_path + f'-run-top3.csv', f'{self.name}-top3')

#Example usage
if __name__ == '__main__':
    args = parse_arguments(True)
    print(args.model)
    splade = SPLADE_PASSAGE(args.index_path, args.corpus_path, args.model)

    for exp_name, queries in zip(args.experiment_names, args.queries_paths):
        #Max pooling
        if args.pooling == 'maxp':
            splade.results_tests(queries, args.out_dir, exp_name, 'max', args.doc_id_map_path, 3_000, 1000, args.query_key)
        #top-k-sum pooling
        elif args.pooling == 'top_k':
            splade.results_tests(queries, args.out_dir, exp_name, 'topk', args.doc_id_map_path, 3_000, 1000, args.query_key)
        else:
            splade.results_tests(queries, args.out_dir, exp_name, None, args.doc_id_map_path, 3_000, 1000, args.query_key)