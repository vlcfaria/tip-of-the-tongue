import json

def read_file(p):
    ids = []
    objs = {}
    with open(p, 'r') as inp:
        for line in inp:
            obj = json.loads(line)

            objs[obj['query_id']] = obj
            ids.append(obj['query_id'])
    return set(ids), objs

ALL = 'queries/test-2025/test-2025-queries.jsonl'
REWRITTEN = 'queries/test-2025/rewritten-queries.jsonl'

(a, a_objs), (b, b_objs) = read_file(ALL), read_file(REWRITTEN)

missing = a - b

if len(missing) == 0: exit(0)



print("MISSING")

with open('missing.jsonl', 'w') as outp:
    for q in missing:
        print(len(a_objs[q]['query']))
        outp.write(f'{json.dumps(a_objs[q])}\n')

with open('hallucinated.jsonl', 'w') as outp:
    for q in a_objs.values():
        if len(q['query']) > 3000:
            print(len(q['query']), q['query_id'])
            outp.write(f'{json.dumps(q)}\n')

