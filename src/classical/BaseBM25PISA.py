from Experiment import Experiment
import pyterrier as pt
from pyterrier_pisa import PisaIndex
from jsonlHandler import iter_jsonl, transform_raw

class BM25(Experiment):
    '''Base BM25, only applying standard Terrier tokenization/stemming/stopwords and only using BM25 for ranking'''
    
    def __init__(self, index_path: str, corpus_path: str= ''):
        super().__init__(index_path, corpus_path)
        
        self.name = 'base-bm25'
        self.search_pipeline = self.index.bm25(precompute_impact=True, verbose=True)
    
    def get_index(self, index_path: str):
        return PisaIndex(index_path, stemmer='porter2')
    
    def build_index(self, index_path: str, corpus_path: str):
        index = PisaIndex(index_path, stemmer='porter2', threads=3)

        index.index(iter_jsonl(corpus_path, transform_raw))

        return index

#Example usage
if __name__ == '__main__':
    #Make an index at `./terrier_index` using ./dataset/corpus.jsonl
    bm = BM25('./indexes/terrier-pisa')

    #Benchmark it
    bm.benchmark('./queries/2023/test/queries.jsonl', './queries/2023/test/qrel.txt')
    #bm.results_tests('./queries/2024/train/queries.jsonl', './results')