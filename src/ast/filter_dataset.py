import json
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
from src.classical.tsvHandler import tsv_corpus_generator

def filter_corpus(doc_ids, corpus_path, output_path, overwrite=False):
    """Filters a .jsonl corpus and returns it in a pd dataframe"""
    
    out_path = Path(output_path)

    if out_path.is_file() and not overwrite:
        print("Warning: Used already pre-filtered corpus.")
        return pd.read_json(out_path, lines=True)
        
    print(f"Filtering corpus into {output_path}...")
    
    target_ids = set(doc_ids) 
    docs_found = 0
    
    try:
        total_size = os.path.getsize(corpus_path)
    except OSError:
        total_size = None

    with open(corpus_path, 'r', encoding='utf-8') as infile, \
         open(out_path, 'w', encoding='utf-8') as outfile, \
         tqdm(total=total_size, unit='B', unit_scale=True, desc="Filtering") as pbar:
        
        for line in infile:
            pbar.update(len(line.encode('utf-8')))
            
            doc = json.loads(line)
            
            if doc['id'] in target_ids:
                outfile.write(line)
                docs_found += 1
                
                if docs_found == len(target_ids):
                    print("\nFound all target documents. Stopping early!")
                    break

    print(f"\nExtraction complete. Saved {docs_found} documents.")
    
    return pd.read_json(out_path, lines=True)

#Filter passages (they are in tsv format)
def filter_passage(passage_ids, passage_paths, output_path, overwrite=False):
    out_path = Path(output_path)

    if out_path.is_file() and not overwrite:
        print("Warning: Used already pre-filtered passages.")
        return pd.read_json(out_path, lines=True)
        
    print(f"Filtering passages into {output_path}...")

    gen = tsv_corpus_generator(passage_paths)

    with open(out_path, 'w', encoding='utf-8') as outfile:
        for data in gen:
            if data['docno'] in passage_ids:
                outfile.write(f'{json.dumps(data)}\n')
    
    return pd.read_json(out_path, lines=True)

    
