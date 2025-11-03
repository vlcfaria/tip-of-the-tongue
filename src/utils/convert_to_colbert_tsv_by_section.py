#Colbert library requires that the ids are the same as the lines
import json
import csv

DATASET_PATH = './dataset/corpus.jsonl'
OUTPUT_PATH = './dataset/corpus_colbert.tsv'
TRANSLATION_PATH = './dataset/idx_to_pid.json'

if __name__ == '__main__':
    pids = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as inp:
        with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as outp:
            writer = csv.writer(outp, delimiter='\t')
            count = 0
            for line in inp:
                raw = json.loads(line)
                doc_id = raw['doc_id']

                #Split document into passages
                for passage in raw['sections']:
                    pids.append(doc_id) #Point this passage back to this docid
                    section_text = f"{raw['title']}. {raw['text'][passage['start']:passage['end']]}"
                    writer.writerow([count, section_text])
                    count += 1

    with open(TRANSLATION_PATH, 'w') as outp:
        json.dump(pids, outp)
