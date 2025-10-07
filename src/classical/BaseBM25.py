from Experiment import Experiment, parse_arguments
import pyterrier as pt
from jsonlHandler import iter_jsonl, transform_raw

class BM25(Experiment):
    '''Base BM25, only applying standard Terrier tokenization/stemming/stopwords and only using BM25 for ranking'''
    
    def __init__(self, index_path: str, experiment_name: str, corpus_path: str= ''):
        super().__init__(index_path, corpus_path)
        
        self.name = 'base-bm25'
        self.experiment_name = experiment_name
        self.search_pipeline = pt.rewrite.tokenise() >> pt.terrier.Retriever(self.index, wmodel='BM25', threads=8)
    
    def get_index(self, index_path: str):
        return pt.IndexFactory.of(f"{index_path}/data.properties")
    
    def build_index(self, index_path: str, corpus_path: str):
        #Index raw data
        return (pt.IterDictIndexer(index_path, meta={'docno': 20}, blocks=True, threads=8)
                    .index(iter_jsonl(corpus_path, transform_raw)))

#Example usage
if __name__ == '__main__':
    args = parse_arguments()
    bm = BM25(args.index_path, args.experiment_name, args.corpus_path)

    #Benchmark it
    if args.qrels:
        bm.benchmark(args.queries_path, args.qrels, args.out_dir)
    else:
        bm.results_tests(args.queries_path, args.out_dir)