import pandas as pd
import pyterrier as pt
import json
import argparse
import sys

def parse_arguments(passage_args:bool= False):
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

    if passage_args:
        parser.add_argument(
            '--doc_id_map_path',
            type=str,
            default=None,
            help='(OPTIONAL) Path to list containing (converted) docid to ACTUAL docid'
        )

        parser.add_argument(
            '--pooling',
            type=str,
            default=None,
            choices=['maxp', 'top_k'],
            help='(OPTIONAL) Pooling method. If not specified, tries all of them'
        )

    parser.add_argument(
        '--query_key',
        type=str,
        default='query',
        help='(OPTIONAL) Defines the key to access the query string in the input .jsonl'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='(OPTIONAL) Defines the model name to be used, if the script allows it'
    )

    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='(OPTIONAL) Custom experiment name to be used, if the script allows it'
    )

    args = parser.parse_args()

    if len(args.experiment_names) != len(args.queries_paths):
        parser.error("The number of experiment_names must match the number of --queries_paths.")
    
    return args

class Experiment():
    def __init__(self, index_path: str, corpus_path: str = '') -> None:
        self.name = 'abstract'
        self.experiment_name = ''

        if corpus_path:
            self.index = self.build_index(index_path, corpus_path)
        else:
            self.index = self.get_index(index_path)

        self.search_pipeline = None
        
    def get_index(self, index_path):
        'Gets the given index from `index_path`'
        raise NotImplementedError()

    def build_index(self, index_path, corpus_path):
        'Builds the given index in `index_path`, using the given corpus'
        raise NotImplementedError()
    
    def benchmark(self, topics_path: str, qrels_path: str, out_dir: str = ''):
        'Benchmarks the established search pipeline, according to established metrics'

        qrels = pd.read_csv(qrels_path, usecols=[0, 2, 3], names=['qid', 'docno', 'label'], header=None, sep=' ',
                            dtype={'qid': 'object', 'docno': 'object', 'label': 'int64'})

        with open(topics_path, 'r') as inp:
            qids, queries = [], []
            for line in inp:
                obj = json.loads(line)
                qids.append(obj['query_id'])
                queries.append(obj['query'])
        
        topics = pd.DataFrame({'qid': qids, 'query': queries})
        topics = topics.astype({'qid': 'object', 'query': 'object'})

        ans = pt.Experiment([self.search_pipeline], topics, qrels, 
                            eval_metrics=['ndcg_cut', 'recall'], 
                            names=[self.name])
        
        if out_dir:
            ans.to_csv(f'{out_dir}/{self.name}-{self.experiment_name}-benchmark.csv')
        else:
            print(ans)

    def results_tests(self, test_query_path: str, out_dir: str):
        'Outputs the query results (up to 100) for each test query, using the search pipeline'

        #Load queries
        with open(test_query_path, 'r') as inp:
            qids, queries = [], []
            for line in inp:
                obj = json.loads(line)
                qids.append(str(obj['query_id']))
                queries.append(obj['query'])
                
        topics = pd.DataFrame({'qid': qids, 'query': queries})
        topics = topics.astype({'qid': 'object', 'query': 'object'})

        #Transform applies the search
        df_ans = (self.search_pipeline).transform(topics)
        df_ans = df_ans.sort_values(['qid', 'rank']) #Sort by rank, just in case

        #Detailed result (full pyterrier df) -> Disabled due to .csv size -> Showed details regarding query rewriting
        #Can format this a little bit better when using query rewriting
        #df_ans.to_csv(f'{out_dir}/{self.name}-{self.experiment_name}-detailed.csv', index=False)

        #Formatted result, in TREC ordering
        formatted = df_ans[['qid', 'docno', 'rank', 'score']].copy() #Prevent warning
        formatted['Q'] = 'Q0'
        formatted['run_id'] = f'{self.name}'
        
        formatted.to_csv(f'{out_dir}/{self.name}-{self.experiment_name}-run.csv', index=False, columns=['qid', 'Q', 'docno', 'rank', 'score', 'run_id'])