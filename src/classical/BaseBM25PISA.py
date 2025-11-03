from Experiment import Experiment
import pyterrier as pt
from pyterrier_pisa import PisaIndex
from jsonlHandler import iter_jsonl, transform_raw

def transform_toks(raw: dict[str, str]) -> dict[str,str]:
    'Transform an entity dictionary into another entity with a single text field'

    return {'docno': raw['doc_id'], 'toks': raw['toks']}

class BM25(Experiment):
    '''Base BM25, only applying standard Terrier tokenization/stemming/stopwords and only using BM25 for ranking'''
    
    def __init__(self, index_path: str, corpus_path: str= ''):
        super().__init__(index_path, corpus_path)
        
        self.name = 'base-bm25'
        self.search_pipeline = self.index.bm25(precompute_impact=True, verbose=True)
    
    def get_index(self, index_path: str):
        return PisaIndex(index_path, stemmer='porter2')
    
    def build_index(self, index_path: str, corpus_path: str):
        index = PisaIndex(index_path, stemmer='porter2', overwrite=True, batch_size=2)

        index.index(iter_jsonl(corpus_path, transform_raw))

        return index

#Example usage
if __name__ == '__main__':
    #Make an index at `./terrier_index` using ./dataset/corpus.jsonl
    bm = BM25('./indexes/testing', './dataset/single.jsonl')

    #Benchmark it
    print(bm.search_pipeline.search('cat'))