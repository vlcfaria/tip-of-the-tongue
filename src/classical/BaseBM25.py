from Experiment import Experiment
import pyterrier as pt
from jsonlHandler import iter_jsonl, transform_raw
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A script to reproduce an Information Retrieval (IR) pipeline using PyTerrier.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'experiment_name',
        type=str,
        help='Name of the given experiment.'
    )

    parser.add_argument(
        'index_path',
        type=str,
        help='Path to the Terrier index directory.'
    )

    parser.add_argument(
        'queries_path',
        type=str,
        help='Path to the file containing the queries.'
    )

    parser.add_argument(
        '--corpus_path',
        type=str,
        default=None,
        help='(Optional) Path to the corpus that was indexed.'
    )

    parser.add_argument(
        '--qrels',
        type=str,
        default=None,
        help='Path to the qrels file for evaluation.'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='Path to the output directory. Mandatory if --qrels is not provided.'
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if not args.qrels and not args.out_dir:
        parser.error('argument --out_dir is required when --qrels is not provided.')
    
    return args

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