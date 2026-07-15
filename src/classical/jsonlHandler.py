import json

def iter_jsonl(filename: str, transform, skip_duplicate_docnos: bool = False):
    'Index an entity jsonl file, calling `transform` for every loaded json entity'

    seen_docnos = set()
    with open(filename, 'rt', encoding='utf-8') as file:
        for l in file:
            raw = json.loads(l)
            transformed = transform(raw)
            
            if skip_duplicate_docnos:
                docno = transformed.get('docno')
                if docno in seen_docnos:
                    continue
                seen_docnos.add(docno)
                
            yield transformed

def transform_raw(raw: dict[str, str]) -> dict[str,str]:
    'Transform an entity dictionary into another entity with a single text field'

    text = ' \n '.join([raw['title'], raw['text']])

    return {'docno': raw['id'], 'text': text}

def transform_fields(raw: dict[str, str]) -> dict[str,str]:
    'Transform an entity dictionary into an entity with multiple fields'

    return {'docno': raw['id'],
            'title': raw['title'],
            'text': raw['text'], 
            'keywords': ' \n '.join(raw['keywords'])}