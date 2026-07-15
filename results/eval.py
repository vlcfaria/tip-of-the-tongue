#analysis.ipynb in .py format for grid search/script evaluation
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #same order as nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pyterrier as pt
import pyterrier_alpha as pta
from pathlib import Path
import pandas as pd
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cfg",        type=json.loads, default="{}") 
    return parser.parse_args()

DATASETS = {
    'dev1': {
        'queries': 'queries/dev1/dev1-2025-queries.jsonl',
        'qrels': 'queries/dev1/dev1-2025-qrel.txt'
    },
    'dev2': {
        'queries': 'queries/dev2/dev2-2025-queries.jsonl',
        'qrels': 'queries/dev2/dev2-2025-qrel.txt'
    },
    'dev3': {
        'queries': 'queries/dev3/dev3-2025-queries.jsonl',
        'qrels': 'queries/dev3/dev3-2025-qrel.txt',
    },
    'test-2025': {
        'queries': 'queries/test-2025/test-2025-queries.jsonl',
        'qrels': 'queries/test-2025/test-2025-qrel.txt',
    },
    'train-2025': {
        'queries': 'queries/train-2025/train-2025-queries.jsonl',
        'qrels': 'queries/train-2025/train-2025-qrel.txt',
    },
    'train-2026': {
        'queries': 'queries/2026/train/queries-train-en.jsonl',
        'qrels': 'queries/2026/train/qrels-train-en.txt',
    },
    'dev-2026': {
        'queries': 'queries/2026/dev/queries-dev-en.jsonl',
        'qrels': 'queries/2026/dev/qrels-dev-en.txt'
    }
}

DATASETS_TO_TEST = ['test-2025', 'dev-2026']

def get_result_dirs(base_dir: Path):
    return [
        #{ 'name': 'bm25', 'dir': base_dir/'bm25', 'variations': None },
        #{ 'name': 'dense-small', 'dir': base_dir/'dense-harrier-small', 'variations': ['max', 'top3'] },
        #{ 'name': 'dense-medium', 'dir': base_dir/'dense-harrier-medium', 'variations': ['max', 'top3'] },
        { 'name': 'splade', 'dir': base_dir/'splade-passage', 'variations': ['max', 'top3'] },
    ]

class ExperimentResults:
    def __init__(self, experiments, datasets):
        self.dt2files = self._infer_result_files(experiments, datasets)
        print(self.dt2files)
        self.qrels, self.queries = self._load_qrels_queries(datasets)
        self.dt2dfs = self._load_trec_csv(self.dt2files)

    def _infer_result_files(self, experiments, datasets):
        'Take in result details, output files grouped by dataset'

        dt2files = { x: {} for x in datasets }
        for xp in experiments:
            files = [f for f in Path(xp['dir']).iterdir() if f.is_file()]

            for d in datasets:
                if xp['variations'] == None:
                    dt2files[d][xp['name']] = next((f for f in files if d in f.name))
                    continue
                for var in xp['variations']:
                    dt2files[d][xp['name'] + '-' + var] = next((f for f in files if d in f.name and var in f.name))

        return dt2files
    
    def _load_qrels_queries(self, datasets):
        dt2qrel = {}
        dt2queries = {}

        for dt in datasets:            
            #load qrels
            dt2qrel[dt] = pd.read_csv(DATASETS[dt]['qrels'], usecols=[0, 2, 3], names=['qid', 'docno', 'relevance'], header=None, sep=' ',
                                dtype={'qid': 'str', 'docno': 'str', 'relevance': 'int64'})
            
            #load queries
            dt2queries[dt] = pd.read_json(DATASETS[dt]['queries'], lines=True, dtype={'query_id': 'str', 'query': 'str'}).rename(columns={'query_id': 'qid'})

        return dt2qrel, dt2queries

    def _load_trec_csv(self, dt2files: dict):
        dt2df = { x: {} for x in dt2files.keys() }

        print(dt2files)
        for name, experiments in dt2files.items():
            for exp, file in experiments.items():
                dt2df[name][exp] = pd.read_csv(file, dtype={'qid': 'string'})

            #Also do rr fusion
            dt2df[name]['fused'] = pta.fusion.rr_fusion(*[pd.merge(self.queries[name], x, on='qid') for x in dt2df[name].values()], num_results=1000)
        
        return dt2df

    def aggregate_experiment_results(self, eval_metrics):
        dt2result = { x: {} for x in self.dt2dfs.keys() }

        for dataset, experiments in self.dt2dfs.items():
            dt2result[dataset] = pt.Experiment(
                [x for x in experiments.values()],
                self.queries[dataset],
                self.qrels[dataset],
                eval_metrics=eval_metrics,
                names=[x for x in experiments.keys()],
                highlight=True,
            )
        return dt2result
    
from ir_measures import *
AGGR_METRICS = [nDCG@10, nDCG@100, nDCG@1000, Success@1, Success@10, Success@100, Success@1000, MRR]

raw_results = ExperimentResults(get_result_dirs(Path('./results/rewrite-sft-ast')), DATASETS_TO_TEST)
agg_raw = raw_results.aggregate_experiment_results(AGGR_METRICS)

def save_results(results: dict, run_name: str, cfg: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    frames = []
    for split_name, df in results.items():
        df = df.copy()
        df["split"]        = split_name
        df["run_name"]     = run_name
        for k, v in cfg.items():
            df[k] = v
        frames.append(df)
    
    combined = pd.concat(frames, ignore_index=True)
    
    out_path = os.path.join(output_dir, f"{run_name}.csv")
    combined.to_csv(out_path, index=False)
    return combined

args = parse_args()

save_results(agg_raw, args.run_name, args.cfg, args.output_dir)