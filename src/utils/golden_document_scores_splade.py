import pyterrier as pt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import pyterrier as pt
import pyt_splade
from pyterrier_pisa import PisaIndex
from collections import defaultdict
import sys
import csv

QUERIES_ROOT = "./queries"
SPLADE_INDEX_PATH = "./indexes/SPLADE-pisa-passage"
CORPUS_PATH = "./corpus.jsonl"
OUTPUT_CSV = "golden_doc_scores_analysis.csv"
IDX_TO_PID_PATH = './dataset/passages/splade/idx_to_pid_splade_overlap.json'
PASSAGE_TSV_PATH = './dataset/passages/splade/corpus_splade_tokenized_overlap.tsv'

import pandas as pd1
import numpy as np
import pyterrier as pt
import pyterrier_alpha as pta

max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

def tsv_corpus_generator(file_path: str, skip_duplicate_docnos: bool = False):
    seen_docnos = set()
    with open(file_path, 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        
        for row in reader:
            docno = row[0]
            if skip_duplicate_docnos and docno not in seen_docnos:
                seen_docnos.add(docno)
                yield { 'docno': docno, 'text': row[1] }

def load_models(splade_path):
    splade_index = PisaIndex(splade_path, stemmer='none')
    splade = pyt_splade.Splade(device='cuda:0', model='naver/splade-v3', max_length=512)
    search_pipeline = splade.scorer('text', 350, True)

    return splade_index, search_pipeline

def get_passage_text(df):
    passages = set(df['docno'].values)
    
    passage_id_to_text = {}
    for data in tsv_corpus_generator(PASSAGE_TSV_PATH, True):
        docno = int(data['docno'])
        if docno in passages:
            passage_id_to_text[docno] = data['text']
    print(len(passage_id_to_text))
    df['text'] = df['docno'].map(passage_id_to_text)
    
    return df

def get_golden_documents(root_dir, retriever, idx_to_pid):
    #Build a reverse map: docnos => passages
    docno_to_passages = defaultdict(list)
    for idx, docno in enumerate(idx_to_pid):
        docno_to_passages[docno].append(idx)

    all_queries = []
    
    topic_files = list(Path(root_dir).glob('*/*/*.ner.jsonl'))

    for query_file in tqdm(topic_files, desc="Processing topic files"):
        qrel_file = (
            query_file.parent / 'qrel.txt'
        )
            
        with open(query_file, 'r') as inp:
            qids, queries = [], []
            for line in inp:
                obj = json.loads(line)
                qids.append(obj['query_id'])
                queries.append(obj['query'])
                
        topics = pd.DataFrame({'qid': qids, 'query': queries})
        queries_df = topics.astype({'qid': 'object', 'query': 'object'})

        qrels_df = pd.read_csv(qrel_file, usecols=[0, 2, 3], names=['qid', 'docno', 'label'], header=None, sep=' ',
                            dtype={'qid': 'object', 'docno': 'object', 'label': 'int64'})
                    
        queries_with_golden = pd.merge(queries_df, qrels_df, on='qid')

        queries_with_golden['passages'] = queries_with_golden['docno'].map(docno_to_passages)
        queries_with_golden = queries_with_golden.explode('passages').rename(columns={'docno': 'docno_real', 'passages': 'docno'})

        print(queries_with_golden)

        #Transform
        all_queries.append(queries_with_golden)

    all_queries = pd.concat(all_queries)
    all_queries = get_passage_text(all_queries)

    return retriever.transform(all_queries)

def get_passage_exact_scores(root_dir, retriever, idx_to_pid):
    #Build a reverse map: docnos => passages
    docno_to_passages = defaultdict(list)
    for idx, docno in enumerate(idx_to_pid):
        docno_to_passages[docno].append(idx)

    all_queries = []
    
    topic_files = list(Path(root_dir).glob('*/*/*.ner.jsonl'))

    for query_file in tqdm(topic_files, desc="Processing topic files"):
        qrel_file = (
            query_file.parent / 'qrel.txt'
        )
            
        with open(query_file, 'r') as inp:
            qids, queries = [], []
            for line in inp:
                obj = json.loads(line)
                qids.append(obj['query_id'])
                queries.append(obj['query'])
                
        topics = pd.DataFrame({'qid': qids, 'query': queries})
        queries_df = topics.astype({'qid': 'object', 'query': 'object'})

        qrels_df = pd.read_csv(qrel_file, usecols=[0, 2, 3], names=['qid', 'docno', 'label'], header=None, sep=' ',
                            dtype={'qid': 'object', 'docno': 'object', 'label': 'int64'})
                    
        queries_with_golden = pd.merge(queries_df, qrels_df, on='qid')

        queries_with_golden['passages'] = queries_with_golden['docno'].map(docno_to_passages)
        queries_with_golden = queries_with_golden.explode('passages').rename(columns={'docno': 'docno_real', 'passages': 'docno'})

        print(queries_with_golden)

        #Transform
        all_queries.append(queries_with_golden)

    all_queries = pd.concat(all_queries)
    all_queries = get_passage_text(all_queries)

    return retriever.transform(all_queries)

def main():
    model, retriever = load_models(SPLADE_INDEX_PATH)
    with open(IDX_TO_PID_PATH, 'r') as inp:
        idx_to_pid = json.loads(inp.readline())
    
    results_df = get_golden_documents(QUERIES_ROOT, retriever, idx_to_pid)


    print(results_df)
    
    results_df.to_csv(OUTPUT_CSV, sep='\t', index=False)
    
if __name__ == "__main__":
    main()