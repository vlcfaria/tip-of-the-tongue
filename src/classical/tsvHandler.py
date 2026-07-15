import csv
import sys

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
            
            if skip_duplicate_docnos:
                if docno in seen_docnos:
                    continue
                seen_docnos.add(docno)
            
            yield { 'docno': docno, 'text': row[1] }

def tsv_corpus_generator_optimized(file_path: str, chunk_size: int = 10000):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        
        chunk = []
        for row in reader:
            if len(row) < 2: 
                continue
            chunk.append({'docno': row[0], 'text': row[1]})
            
            if len(chunk) >= chunk_size:
                yield from chunk
                chunk = []
                
        if chunk:  # Yield the remainder
            yield from chunk