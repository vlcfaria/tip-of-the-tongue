import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #same order as nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from Experiment import Experiment
from tsvHandler import tsv_corpus_generator_optimized
import pyterrier as pt
import pyterrier_dr
from RankingManager import RankingManager
import json
import pandas as pd
from typing import Literal
from dotenv import load_dotenv
import torch
from Experiment import parse_arguments

load_dotenv()

class OptimizedSBertBiEncoder(pyterrier_dr.SBertBiEncoder):
    """
    Custom subclass to allow passing model_kwargs for A100 optimizations
    like Flash Attention 2 and bfloat16.
    """
    def __init__(self, model_name, model_kwargs=None, batch_size=32, text_field='text', verbose=False, device=None):
        pyterrier_dr.BiEncoder.__init__(self, batch_size=batch_size, text_field=text_field, verbose=verbose)
        
        self.model_name = model_name
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        kwargs = model_kwargs or {}
        self.model = SentenceTransformer(model_name, model_kwargs=kwargs).to(self.device).eval()
        self.config = AutoConfig.from_pretrained(model_name)

class DenseRetrieval(Experiment):
    '''Dense passage retrieval'''
    
    def __init__(self, model_name: str, index_path: str, corpus_path: str= '', name='dense-retriever'):
        a100_kwargs = {
            "attn_implementation": "flash_attention_2", 
            "torch_dtype": torch.bfloat16
        }
        self.encoder = OptimizedSBertBiEncoder(
            model_name,
            model_kwargs=a100_kwargs,
            batch_size=8192,
            verbose=True
        )
        self.encoder.model.max_seq_length = 256
        self.encoder.model = torch.compile(self.encoder.model, mode="max-autotune")
        super().__init__(index_path, corpus_path)

        self.name = name
        self.search_pipeline = None
    
    def get_index(self, index_path: str):
        return pyterrier_dr.FlexIndex(index_path)
    
    def build_index(self, index_path: str, corpus_path: str):
        index = pyterrier_dr.FlexIndex(index_path)        
        idx_pipeline = self.encoder.doc_encoder() >> index.indexer()
        
        try:
            return idx_pipeline.index(tsv_corpus_generator_optimized(corpus_path))
        except RuntimeError as e:
            print(f"Index already built: {e}. Ignoring")
            return pyterrier_dr.FlexIndex(index_path)
    
    def results_tests(self, test_query_path: str, out_dir: str, experiment_name: str, pool_fn: Literal['max', 'topk'] | None, doc_id_map_path: str=None, docs_to_pool: int=3000, final_num_docs: int=1000, query_key: str='query'):
        with open(test_query_path, 'r') as inp:
            qids, queries = [], []
            for line in inp:
                obj = json.loads(line)
                qids.append(obj['query_id'])
                queries.append(obj[query_key])
                
        topics = pd.DataFrame({'qid': qids, 'query': queries})
        topics = topics.astype({'qid': 'object', 'query': 'object'})

        instruction = "Instruct: Given a noisy Tip-of-the-Tongue web query, retrieve relevant passages that answer the query\nQuery: "
        topics['query'] = instruction + topics['query']

        search_pipeline = self.encoder.query_encoder(batch_size=2048) >> self.index.faiss_hnsw_retriever(
                                                                            neighbours=32,
                                                                            num_results=docs_to_pool,
                                                                            ef_construction=200,
                                                                            ef_search=docs_to_pool,
                                                                        )
        df_ans = search_pipeline.transform(topics)

        # Convert pyterrier dataframe into ranking
        ranking = RankingManager(df_ans, doc_id_map_path)

        # Save pre-passages
        base_path = f'{out_dir}/{self.name}-{experiment_name}'
        ranking.raw_passage_ranking_to_csv(base_path + '-prepassages.csv', f'{self.name}-prepassage')

        # Pool! (and save)
        if (pool_fn == 'max' or pool_fn == None):
            pooled = ranking.max_pooling(final_num_docs)
            ranking.save_rankings_as_trec_csv(pooled, base_path + f'-run-max.csv', f'{self.name}-max')
        if (pool_fn == 'topk' or pool_fn == None):
            pooled = ranking.top_k_sum_pooling(final_num_docs)
            ranking.save_rankings_as_trec_csv(pooled, base_path + f'-run-top3.csv', f'{self.name}-top3')

#Example usage
if __name__ == '__main__':
    args = parse_arguments(True)
    dr = DenseRetrieval(args.model, args.index_path, args.corpus_path, args.experiment_name)

    for exp_name, queries in zip(args.experiment_names, args.queries_paths):
        #Max pooling
        if args.pooling == 'maxp':
            dr.results_tests(queries, args.out_dir, exp_name, 'max', args.doc_id_map_path, 3_000, 1000, args.query_key)
        #top-k-sum pooling
        elif args.pooling == 'top_k':
            dr.results_tests(queries, args.out_dir, exp_name, 'topk', args.doc_id_map_path, 3_000, 1000, args.query_key)
        else:
            dr.results_tests(queries, args.out_dir, exp_name, None, args.doc_id_map_path, 3_000, 1000, args.query_key)