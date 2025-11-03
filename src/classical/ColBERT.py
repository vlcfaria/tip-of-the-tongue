from io import TextIOWrapper
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #same order as nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from Experiment import Experiment
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher, Indexer
from pooling import max_pooling, top_k_sum_pooling
import argparse
import csv
import json
import torch

def parse_arguments():
    """Creates a parser with common arguments for IR pipelines."""
    parser = argparse.ArgumentParser(
        description="A script to reproduce an Information Retrieval (IR) pipeline using the standard ColBERT library.",
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

class ColBERT(Experiment):
    '''Indexing + Ranking with ColBERT'''
    
    def __init__(
            self, 
            index_path: str,
            corpus_path: str= '',
            corpus_name: str='tot',
            checkpoint: str='colbert-ir/colbertv2.0'
        ):

        self.checkpoint = checkpoint
        self.index_path = index_path

        self.config = ColBERTConfig(
            nbits=2,
            nranks=2,
            gpus=1,
            experiment=corpus_name,
            index_root=index_path,
            root=index_path,
            doc_maxlen=256,
            query_maxlen=512,
            index_bsize=2048,
        )

        self.index_name = f"{self.config.experiment}.nbits={self.config.nbits}"

        super().__init__(index_path, corpus_path)
        self.name = 'colbert'

        with Run().context(RunConfig(root=index_path)):
            self.search_pipeline = Searcher(self.index_name, config=self.config)
        
    def get_index(self, index_path: str):
        return None #Not used here
    
    def build_index(self, index_path: str, corpus_path: str):
        #Root from `index_root` in config does not work by default here, have to set this
        with Run().context(RunConfig(root=index_path)):
            indexer = Indexer(checkpoint=self.checkpoint, config=self.config)
            indexer.index(name=self.index_name, collection=corpus_path)
        
        return None #Not used here
    
    def results_tests(self, test_query_path: str, out_dir: str, experiment_name: str, pool_fn, doc_id_map_path: str=None, docs_to_pool: int=3000, final_num_docs: int=1000):
        'Outputs the query results (up to 1000) for each test query, using the search pipeline'

        #Load queries tsv
        queries = Queries(test_query_path)
        
        #Search for more than we will actually output, since we will apply some sort of pooling
        ans = self.search_pipeline.search_all(queries, k=docs_to_pool)

        if doc_id_map_path:
            with open(doc_id_map_path, 'r') as inp:
                doc_id_mapper = json.load(inp)
            for __builtins__, ranking in ans.items():
                for i in range(len(ranking)): #Converts all docids
                    docid, *rest = ranking[i]
                    ranking[i] = (doc_id_mapper[docid], *rest)
        
        final_ranking = pool_fn(ans, final_num_docs) #Convert to qid -> proper ranking

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
                        f'{self.name}-{self.config.nbits}bits-{self.config.ncells or "default"}cells-{self.config.centroid_score_threshold or "default"}threshold-{self.config.ndocs or "default"}ndocs',
                    ])            
            
    def benchmark(self, topics_path: str, qrels_path: str, out_dir: str = ''):
        raise NotImplementedError()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    args = parse_arguments()
    colbert = ColBERT(args.index_path, args.corpus_path)

    for exp_name, queries in zip(args.experiment_names, args.queries_paths):
        #Max pooling
        if args.pooling == 'maxp':
            colbert.results_tests(queries, args.out_dir, exp_name, max_pooling, args.doc_id_map_path, 3000, 1000)
        #top-k-sum pooling
        if args.pooling == 'top_k':
            print('top pooling')
            colbert.results_tests(queries, args.out_dir, exp_name, top_k_sum_pooling, args.doc_id_map_path, 20_000, 1000)