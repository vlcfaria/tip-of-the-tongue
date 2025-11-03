from Experiment import Experiment, parse_arguments
from tsvHandler import tsv_corpus_generator
import pyterrier as pt
import pyterrier_alpha as pta
import pyt_splade
from pyterrier_pisa import PisaIndex

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

        idx_pipeline = self.splade.doc_encoder(batch_size=3000) >> index.toks_indexer()
        
        return idx_pipeline.index(tsv_corpus_generator(corpus_path))

#Example usage
if __name__ == '__main__':
    args = parse_arguments()
    bm = SPLADE(args.index_path, args.experiment_name, args.corpus_path)

    #Benchmark it
    if args.qrels:
        bm.benchmark(args.queries_path, args.qrels, args.out_dir)
    else:
        bm.results_tests(args.queries_path, args.out_dir)