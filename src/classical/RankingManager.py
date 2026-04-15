import pandas as pd
from typing import Dict, Union, List, Tuple
from collections import defaultdict
import json
import csv

class RankingManager():
    def __init__(self, rankings: Union[pd.DataFrame, Dict[str, List[Tuple[str,int,float]]]], doc_id_map_path: str = None):
        if isinstance(rankings, pd.DataFrame): #If it came from terrier, convert
            qids_to_ranking = defaultdict(list)
            for _, row in rankings.iterrows():
                qids_to_ranking[row['qid']].append((int(row['docno']), row['rank'], row['score']))
            self.rankings = qids_to_ranking
        else:
            self.rankings = rankings

        #Sort, just to make sure
        for _, ranking in self.rankings.items():
            ranking.sort(key=lambda x: x[2], reverse=True)

        self.doc_id_map_path = doc_id_map_path

    def max_pooling(self, num_docs: int):
        '''
        Apply max pooling. `qid_to_raking` is a dictionary mapping a qid to a formed ranking, which is a list of tuples (docid, rank, score). 
        Returns another qid to ranking with the pooled result
        '''
        ans = {}
        converted = self._convert_passages_to_documents()
        for qid, ranking in converted.items():
            scored = set()
            final_ranking = []
            for (docid, _, score) in ranking:
                if docid not in scored:
                    final_ranking.append([
                        docid, 
                        len(scored), #also mind the rank
                        score,
                    ])

                    scored.add(docid)
                    if len(scored) == num_docs: break
            ans[qid] = final_ranking
        return ans

    def top_k_sum_pooling(self, num_docs: int, k=3, cutoff: Union[int, None]= None):
        '''Top-k-sum-pooling. `qid_to_raking` is a dictionary mapping a qid to a formed ranking, which is a list of tuples (docid, rank, score). Returns another qid to ranking with the pooled result'''
        ans = {}
        converted = self._convert_passages_to_documents()
        for qid, ranking in converted.items():
            docid_count = {}
            docid_sum = {}

            for (docid, rank, score) in ranking:
                if cutoff and rank > cutoff: break #Reached cutoff
                if docid_count.get(docid, 0) < k: #Still up until k
                    docid_count[docid] = docid_count.get(docid,0) + 1
                    docid_sum[docid] = docid_sum.get(docid,0) + score
            
            sorted_sums = sorted(docid_sum.items(), key=lambda item: item[1], reverse=True)
            ans[qid] = [(docid,rank,score) for rank, (docid, score) in enumerate(sorted_sums[:num_docs])]

        return ans

    def _convert_passages_to_documents(self):
        '''Converts passages into the original document, for pooling'''
        converted_rankings = {}
        print(self.doc_id_map_path)
        if self.doc_id_map_path:
            with open(self.doc_id_map_path, 'r') as inp:
                doc_id_mapper = json.load(inp)
            for qid, ranking in self.rankings.items():
                converted_ranks = []
                for i in range(len(ranking)): #Converts all docids
                    docid, *rest = ranking[i]
                    converted_ranks.append((doc_id_mapper[docid], *rest))
                converted_rankings[qid] = converted_ranks
            return converted_rankings
        
        print("warning: no doc_id_to_map_path unspecified, running on raw passages")

    def raw_passage_ranking_to_csv(self, path: str, experiment_name: str):
        with open(path, 'w', encoding='utf-8', newline='') as outp:
            writer = csv.writer(outp)
            writer.writerow(['qid', 'Q', 'docno', 'rank', 'score', 'run_id'])
            for qid, ranking in self.rankings.items():
                for (docid, rank, score) in ranking:
                    writer.writerow([
                        qid, 
                        'Q0', #Same format as TREC
                        docid,
                        rank,
                        score,
                        experiment_name,
                    ])

    def save_rankings_as_trec_csv(self, rankings: Dict[str, List[Tuple[str,int,float]]], path: str, experiment_name):
        '''Saves a ranking into a TREC csv'''
        with open(path, 'w', encoding='utf-8', newline='') as outp:
            writer = csv.writer(outp)
            writer.writerow(['qid', 'Q', 'docno', 'rank', 'score', 'run_id'])
            for qid, ranking in rankings.items():
                for (docid, rank, score) in ranking:
                    writer.writerow([
                        qid, 
                        'Q0',
                        docid,
                        rank,
                        score,
                        experiment_name,
                    ])

#CLI use, takes in a saved ranking and repools it with top-k (for now)
import argparse

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(
            description="Auxiliary script to rerank using top-k passages",
            formatter_class=argparse.RawTextHelpFormatter,
            add_help=True
        )

        parser.add_argument(
            'csv_path',
            type=str,
            help='Path to passage ranking.'
        )

        parser.add_argument(
            '--doc_id_map_path',
            type=str,
            default=None,
            help='(OPTIONAL) Path to list containing (converted) docid to ACTUAL docid'
        )

        parser.add_argument(
            '--cutoff',
            type=int,
            default=None,
            help='(OPTIONAL) Cutoff to apply on top-k pooling. Defaults to no cutoff'
        )

        return parser.parse_args()

    args = parse_args()

    df = pd.read_csv(args.csv_path)

    ranking = RankingManager(df, args.doc_id_map_path)
    
    top3 = ranking.top_k_sum_pooling(1000, cutoff=args.cutoff)

    ranking.save_rankings_as_trec_csv(top3, './ranking_output.csv', df['run_id'][0] + f'-top3-{args.cutoff}cutoff')
