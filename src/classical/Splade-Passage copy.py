import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
from Experiment import Experiment
from tsvHandler import tsv_corpus_generator
import pyterrier as pt
import pyterrier_alpha as pta
import pyt_splade
from pyterrier_pisa import PisaIndex
from huggingface_hub import login
import argparse
from dotenv import load_dotenv
from pooling import max_pooling, top_k_sum_pooling
import json
import pandas as pd
from collections import defaultdict
import csv
load_dotenv()
#login(token=os.getenv('HF_TOKEN'))

def parse_arguments():
    """Creates a parser with common arguments for IR pipelines."""
    parser = argparse.ArgumentParser(
        description="A script to reproduce an Information Retrieval (IR) pipeline using Splade, with passage indexing",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True
    )

    parser.add_argument(
        'index_path',
        type=str,
        help='Path to the index directory.'
    )

    parser.add_argument(
        'out_dir',
        type=str,
        default=None,
        help='Path to the output directory.'
    )

    parser.add_argument(
        'experiment_names',
        type=str,
        nargs='+',
        help='Name of the given experiment.'
    )

    parser.add_argument(
        '--queries_paths',
        type=str,
        required=True,
        nargs='+',
        help='Path to the file containing the queries. Ordering must be the same as experiment_name'
    )

    parser.add_argument(
        '--corpus_path',
        type=str,
        default=None,
        help='(Optional) Path to the corpus to be indexed. Setting this option will index the corpus'
    )

    parser.add_argument(
        '--doc_id_map_path',
        type=str,
        default=None,
        help='(OPTIONAL) Path to list containing (converted) docid to ACTUAL docid'
    )

    parser.add_argument(
        '--pooling',
        type=str,
        default='maxp',
        choices=['maxp', 'top_k'],
        help='(OPTIONAL) Pooling method. Defaults to maxpooling'
    )

    args = parser.parse_args()

    if len(args.experiment_names) != len(args.queries_paths):
        parser.error("The number of experiment_names must match the number of --queries_paths.")
    
    return args

class SPLADE_PASSAGE(Experiment):
    '''SPLADE experimentation, with passage indexing instead of document indexing'''
    
    def __init__(self, index_path: str, corpus_path: str= ''):
        super().__init__(index_path, corpus_path)
        
        self.name = 'splade-passage'
        self.splade = pyt_splade.Splade(device='cuda:0', model='naver/splade-v3', max_length=512) #We use 512 max length for query encoding
        self.search_pipeline = None #Search pipeline is dynamic due to dynamic documents to pool
    
    def get_index(self, index_path: str):
        return PisaIndex(index_path, stemmer='none') #Multithreading is giving segfaults when searching
    
    def build_index(self, index_path: str, corpus_path: str):
        #max token is a bit lower for indexing
        splade = pyt_splade.Splade(device='cuda:0', model='naver/splade-v3', max_length=256)
        index = PisaIndex(index_path, stemmer='none', threads=16, batch_size=3_000_000)
        idx_pipeline = splade.doc_encoder(batch_size=512) >> index.toks_indexer()
        
        return idx_pipeline.index(tsv_corpus_generator(corpus_path))
    
    def results_tests(self, test_query_path: str, out_dir: str, experiment_name: str, pool_fn, doc_id_map_path: str=None, docs_to_pool: int=3000, final_num_docs: int=1000):
        with open(test_query_path, 'r') as inp:
            qids, queries = [], []
            for line in inp:
                obj = json.loads(line)
                qids.append(obj['query_id'])
                queries.append(obj['query'])
                
        topics = pd.DataFrame({'qid': qids, 'query': queries})
        topics = topics.astype({'qid': 'object', 'query': 'object'})

        search_pipeline = self.splade.query_encoder(batch_size=30) >> self.index.quantized(num_results=docs_to_pool, query_algorithm='block_max_maxscore', verbose=True)
        df_ans = search_pipeline.transform(topics)

        #Convert pyterrier dataframe into ranking
        qids_to_ranking = defaultdict(list)
        for _, row in df_ans.iterrows():
            qids_to_ranking[row['qid']].append((int(row['docno']), row['rank'], row['score']))

        for ranking in qids_to_ranking.values():
            ranking.sort(key=lambda x: x[2], reverse=True)

        #Also convert docids, if needed
        if doc_id_map_path:
            with open(doc_id_map_path, 'r') as inp:
                doc_id_mapper = json.load(inp)
            for ranking in qids_to_ranking.values():
                for i in range(len(ranking)): #Converts all docids
                    docid, *rest = ranking[i]
                    ranking[i] = (doc_id_mapper[docid], *rest)
        
        final_ranking = pool_fn(qids_to_ranking, final_num_docs) #Convert to qid -> proper ranking dict

        with open(f'{out_dir}/{self.name}-{experiment_name}-run.csv', 'w', encoding='utf-8', newline='') as outp:
            writer = csv.writer(outp)
            writer.writerow(['qid', 'Q', 'docno', 'rank', 'score', 'run_id'])
            for qid, ranking in final_ranking.items():
                for (docid, rank, score) in ranking:
                    writer.writerow([
                        qid, 
                        'Q0',
                        docid,
                        rank,
                        score,
                        f'{self.name}-{pool_fn.__name__}',
                    ])

#Example usage
if __name__ == '__main__':
    args = parse_arguments()
    splade = SPLADE_PASSAGE(args.index_path, args.corpus_path)

    for exp_name, queries in zip(args.experiment_names, args.queries_paths):
        #Max pooling
        if args.pooling == 'maxp':
            splade.results_tests(queries, args.out_dir, exp_name, max_pooling, args.doc_id_map_path, 3000, 1000)
        #top-k-sum pooling
        if args.pooling == 'top_k':
            print('top pooling')
            splade.results_tests(queries, args.out_dir, exp_name, top_k_sum_pooling, args.doc_id_map_path, 10_000, 1000)