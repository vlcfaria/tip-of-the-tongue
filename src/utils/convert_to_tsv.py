#Script that converts the corpus to a simple tsv AND ALSO cleans the corpus (since it had weird utf-8 escaped encodings)
import json
import csv

DATASET_PATH = './dataset/corpus.jsonl'
OUTPUT_PATH = './dataset/corpus_cleaned.tsv'

if __name__ == '__main__':
    with open(DATASET_PATH, 'r', encoding='utf-8') as inp:
        with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as outp:
            writer = csv.writer(outp, delimiter='\t')
            for line in inp:
                raw = json.loads(line)
                doc_id = raw['doc_id']
                full_text = '\n\n'.join([raw['title'], raw['text']])
                writer.writerow([doc_id, full_text])
