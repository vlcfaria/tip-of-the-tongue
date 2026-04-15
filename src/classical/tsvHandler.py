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
            if skip_duplicate_docnos and docno not in seen_docnos:
                seen_docnos.add(docno)
                yield { 'docno': docno, 'text': row[1] }