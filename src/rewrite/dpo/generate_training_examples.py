import asyncio
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import sys
import textwrap
from tqdm import tqdm
import json
import aiofiles
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils import extract_query, extract_thinking
from consts import get_user_prompt, get_system_prompt, RETRIEVERS
import pandas as pd
from pathlib import Path

API_KEY = "EMPTY" 
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "tip-of-tongue-rewriter"
N_REWRITES = 5

client = AsyncOpenAI(
    api_key='',
    base_url=BASE_URL
)

MAX_CONCURRENT_REQUESTS = 50
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

query_files = ['queries/dpo-train/queries.jsonl']

async def rewrite_query(query_id, query, retriever, candidate_id, max_retries=20):
    """Rewrites a ToT query with an automatic retry mechanism for API and formatting errors."""

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": get_system_prompt()},
                        {"role": "user", "content": get_user_prompt(query[:3000], retriever)}
                    ],
                    temperature=0.7,
                    max_tokens=8000,
                )

                response_content = response.choices[0].message.content.strip()

                extracted_query = extract_query(response_content)
                extracted_thinking = extract_thinking(response_content)

                #Ensure model thinked and outputted the query
                if extracted_query is None or extracted_thinking is None:
                    print(f"[WARN] query_id={query_id} ({retriever}): wrong formatting, retrying...")
                    continue

                return query_id, retriever, candidate_id, extracted_query, extracted_thinking, response_content
            except Exception as e:
                print(f"Error on query {query_id}: {e}")
                #await asyncio.sleep(0.5)
        return query_id, retriever, candidate_id, None, None, None # Too long

async def main(experiment_name="sft"):
    for qf in query_files:
        print("READING", qf)
        df = pd.read_json(qf, lines=True)
        print("read")
        directory = Path(qf).parent

        temp_output_file = directory / f'dpo-train-queries-{experiment_name}-temp.jsonl'
        final_output_file = directory / f'dpo-train-queries-{experiment_name}.jsonl'

        if final_output_file.exists():
            print(f"Final output {final_output_file} already exists. Skipping {qf}...")
            continue

        processed_pairs = set()
        try:
            print(final_output_file)
            with open(temp_output_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    processed_pairs.add((data.get('query_id'), data.get('retriever'), data.get('candidate_id')))
            print(f"Found {len(processed_pairs)} already processed candidates. Resuming...")
        except FileNotFoundError:
            pass

        # Fixed: Checking against 'query_id' instead of 'id' to match your row iteration
        print(f"Starting batch generation for queries...")
            
        tasks = [
            rewrite_query(row['query_id'], row['query'], retr, c_id) 
            for _, row in df.iterrows()
            for retr in RETRIEVERS
            for c_id in range(N_REWRITES)
            if (row['query_id'], retr, c_id) not in processed_pairs
        ]

        print(f"Starting batch generation for {len(tasks)} tasks...")
        
        if tasks:
            print(f"Starting batch generation for {len(tasks)} tasks...")
            
            async with aiofiles.open(temp_output_file, mode='a') as out_file:
                for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Rewriting"):
                    query_id, retriever, candidate_id, query, thinking, raw_response = await f
                    
                    if query is not None:
                        result_dict = {
                            "query_id": query_id, 
                            "retriever": retriever, 
                            "candidate_id": candidate_id,
                            "query": query, 
                            "thinking": thinking,
                            "raw_response": raw_response
                        }
                        await out_file.write(json.dumps(result_dict) + "\n")
        
        print("Batch generation complete! Merging results...")
        
        results_df = pd.read_json(temp_output_file, lines=True)
        
        results_df['true_query_id'] = results_df['query_id'].astype(str)

        results_df['query_id'] = results_df['query_id'].astype(str) + "$" + results_df['candidate_id'].astype(str)

        #results_df.to_json(final_output_file, orient='records', lines=True)

        queries_pivot = results_df.pivot(index='query_id', columns='retriever', values='query')
        queries_pivot.columns = [f"{col}_query" for col in queries_pivot.columns]

        print(queries_pivot)
        
        # 2. Group the thinking into a nested dictionary per query_id 
        # Output structure: {'BM25': '...', 'ColBERT': '...'}
        thinking_nested = (
            results_df.groupby('query_id')
            .apply(lambda x: dict(zip(x['retriever'], x['thinking'])))
            .reset_index(name='thinking')
        )
        
        # 3. Combine the flattened queries and the nested thinking
        grouped_results = queries_pivot.reset_index().merge(thinking_nested, on='query_id')

        df['query_id'] = df['query_id'].astype(str)
        
        # 4. Merge back into the original dataframe
        final_df = grouped_results
        final_df.to_json(final_output_file, orient='records', lines=True)
        
        print(f"Successfully processed and merged {qf}!")

if __name__ == "__main__":
    asyncio.run(main())