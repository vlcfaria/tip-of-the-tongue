import pandas as pd
import json
import re
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm

API_KEY = "EMPTY" 
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "tip-of-tongue-rewriter"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class RewriteExtraction(BaseModel):
    cues: List[str] = Field(
        description="A clean list of extracted details, facts, or context clues from the user's prompt."
    )
    dense_rewrite: str = Field(
        description="A fluid, well-formed natural language sentence summarizing the description, optimized for semantic search."
    )

SYSTEM_PROMPT = """You are an expert retrieval assistant. Analyze the user's Tip-of-the-Tongue description.
1. Extract a clear list of atomic search cues (keywords, fragments, entities).
2. Synthesize those cues into a single, highly descriptive but concise while retaining nuance, grammatically fluent sentence that a dense retrieval model can embed effectively.

You must respond ONLY with a valid JSON object. Do not include markdown formatting, preambles, or explanations."""

query_files = [
    #'queries/dev1/dev1-2025-queries.jsonl',
    #'queries/dev2/dev2-2025-queries.jsonl',
    #'queries/dev3/dev3-2025-queries.jsonl',
    #'queries/test-2025/test-2025-queries.jsonl',
    #'queries/train-2025/train-2025-queries.jsonl'
    'queries/test-2025/missing.jsonl'
]

MAX_RETRIES = 10

for qf in query_files:
    queries = []
    with open(qf, 'r') as f:
        for line in f:
            queries.append(json.loads(line))

    output_file = Path(qf).parent / 'rewritten-queries.jsonl'
    
    for q in tqdm(queries, desc=f"Processing {qf}"):
        retries = 0
        success = False

        while retries < MAX_RETRIES and not success:
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Input: {q['query']}"},
                    ],
                    temperature=0.1,
                    top_p=0.95,
                    max_tokens=1024,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "rewrite_extraction",
                            "strict": True,
                            "schema": RewriteExtraction.model_json_schema()
                        }
                    }
                )

                response_content = completion.choices[0].message.content
        
                parsed_json = json.loads(response_content)
                validated_data = RewriteExtraction(**parsed_json)

                cleaned_cues = [
                    cue.strip().rstrip(".,;?!") 
                    for cue in validated_data.cues
                ]
            
                with open(output_file, 'a') as f:
                    data = {
                        "query_id": q['query_id'],
                        "cues": cleaned_cues,
                        "bm25_query": ", ".join(cleaned_cues),
                        "dense_query": validated_data.dense_rewrite
                    }
                    f.write(f"{json.dumps(data)}\n")
                
                success = True
                        
            except (json.JSONDecodeError, ValidationError) as e:
                retries += 1
                if retries == MAX_RETRIES:
                    print(f"Failed to process query {q['query_id']} after {MAX_RETRIES} attempts. Skipping.")
                    print(f"Last error: {e}")
                    print(f"Raw output: {response_content}")
