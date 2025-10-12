from Experiment import Experiment, parse_arguments
from jsonlHandler import iter_jsonl, transform_raw
import pyterrier as pt
import pyterrier_alpha as pta
import pyt_splade
from pyterrier_pisa import PisaIndex
import json
import torch

class SPLADE(Experiment):
    '''SPLADE experimentation'''
    
    def __init__(self, index_path: str, experiment_name: str, corpus_path: str= ''):
        self.splade = pyt_splade.Splade(device='cuda:2') #Defaults to model naver/splade-cocondenser-ensembledistil
        super().__init__(index_path, corpus_path)
        
        self.name = 'splade-base'
        self.experiment_name = experiment_name

        self.search_pipeline = self.splade.query_encoder() >> self.index.quantized()
    
    def get_index(self, index_path: str):
        return PisaIndex(index_path, stemmer='none')
    
    def build_index(self, index_path: str, corpus_path: str):
        #Index raw data
        index = PisaIndex(index_path, stemmer='none')

        idx_pipeline = self.splade.doc_encoder() >> index.toks_indexer()
        
        return idx_pipeline.index(iter_jsonl(corpus_path, transform_raw))

#Example usage
if __name__ == '__main__':
    args = parse_arguments()
    bm = SPLADE(args.index_path, args.experiment_name, args.corpus_path)

    #Benchmark it
    if args.qrels:
        bm.benchmark(args.queries_path, args.qrels, args.out_dir)
    else:
        bm.results_tests(args.queries_path, args.out_dir)