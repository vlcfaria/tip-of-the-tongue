import csv
import sys

max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

def tsv_corpus_generator(file_path: str):
    with open(file_path, 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        
        for row in reader:
            yield { 'docno': row[0], 'text': row[1] }