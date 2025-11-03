import json

def iter_jsonl(filename: str, transform):
    'Index an entity jsonl file, calling `transform` for every loaded json entity'

    with open(filename, 'rt', encoding='utf-8') as file:
        for l in file:
            raw = json.loads(l)

            yield transform(raw)

def transform_raw(raw: dict[str, str]) -> dict[str,str]:
    'Transform an entity dictionary into another entity with a single text field'

    text = ' \n '.join([raw['title'], raw['text']])

    return {'docno': raw['doc_id'], 'text': text}

def transform_fields(raw: dict[str, str]) -> dict[str,str]:
    'Transform an entity dictionary into an entity with multiple fields'

    return {'docno': raw['id'],
            'title': raw['title'],
            'text': raw['text'], 
            'keywords': ' \n '.join(raw['keywords'])}