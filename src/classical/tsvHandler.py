import csv

def read_corpus_generator(file_path: str):
    with open(file_path, 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        
        for row in reader:
            yield { 'docno': row[0], 'text': row[1] }