#Script that implements query-rewriting, following the method as https://trec.nist.gov/pubs/trec34/papers/SRCB.mllm.tot.pdf, save the queries as `rewritten-queries.jsonl`

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
MODEL_NAME = "casperhansen/deepseek-r1-distill-llama-70b-awq"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class CueExtraction(BaseModel):
    cues: list[str] = Field(
        description="A clean list of extracted details, facts, or context clues from the user's prompt. No conversational text."
    )

def get_prompt(question):
    return f"""You are an intelligent assistant helping users with Tip-of-the-Tongue (ToT) known-item
    retrieval—situations where someone is trying to recall a specific entity (e.g., a movie, book, person,
    song, place, event, etc.) they previously encountered but can’t remember a reliable identifier like
    the name or title. Instead, they provide vague or partial memories. Your task is to extract a list
    of cues from the user’s input. Cues are any meaningful pieces of information that may help in
    identifying the intended entity.
    Cues may include (but are not limited to):
    1. Descriptions of appearance, content, or notable features
    2. Plot fragments, scenes, events, or quotes
    3. Sensory details (visuals, sounds, smells, etc.)
    4. Emotions or atmosphere
    5. Context of experience (e.g., “saw it as a kid,” “on a trip to Italy,” “heard it on the radio”)
    6. Time period when experienced or when it was popular/released
    7. Language, region, or culture of origin
    8. Associated people, groups, or objects
    9. Comparisons (e.g., “felt like a mix between X and Y”)
    10. Any other detail, however imprecise, that the user remembers
    Return the cues in a clear, structured list. Be concise but retain nuance. Directly output the list, according to the given schema.

    Input: {question}
    """

query_files = [
    'queries/2023/test/queries.jsonl',
    'queries/2023/train/queries.jsonl',
    'queries/2024/train/queries.jsonl',
    'queries/2024/test-partial/queries.jsonl'
]

MAX_RETRIES = 3

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
                        {"role": "user", "content": get_prompt(q['query'])},
                    ],
                    temperature=0.1,
                    top_p=0.95,
                    max_tokens=1024,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "cue_extraction",
                            "schema": CueExtraction.model_json_schema()
                        }
                    }
                )

                response_content = completion.choices[0].message.content
        
                parsed_json = json.loads(response_content)
                validated_data = CueExtraction(**parsed_json)

                cleaned_cues = [
                    cue.strip().rstrip(".,;?!") 
                    for cue in validated_data.cues
                ]
            
                with open(output_file, 'a') as f:
                    data = {
                        "query_id": q['query_id'],
                        "cues": cleaned_cues,
                        "query": ", ".join(cleaned_cues)
                    }
                    f.write(f"{json.dumps(data)}\n")
                
                success = True
                        
            except (json.JSONDecodeError, ValidationError) as e:
                retries += 1
                if retries == MAX_RETRIES:
                    print(f"Failed to process query {q['query_id']} after {MAX_RETRIES} attempts. Skipping.")
                    print(f"Last error: {e}")
                    print(f"Raw output: {response_content}")
