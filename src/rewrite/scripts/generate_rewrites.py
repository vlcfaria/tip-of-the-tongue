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
import argparse

API_KEY = "EMPTY" 
BASE_URL = "http://localhost:8000/v1"

client = AsyncOpenAI(
    api_key='',
    base_url=BASE_URL
)

MAX_CONCURRENT_REQUESTS = 50
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

query_files = [
    #'queries/dev1/dev1-2025-queries.jsonl',
    #'queries/dev2/dev2-2025-queries.jsonl',
    #'queries/dev3/dev3-2025-queries.jsonl',
    'queries/test-2025/test-2025-queries.jsonl',
    #'queries/train-2025/train-2025-queries.jsonl',
    'queries/2026/dev/queries-dev-en.jsonl',
    #'queries/2026/train/queries-train-en.jsonl'
    #'queries/sft-train/queries.jsonl'
]

async def rewrite_query(model, query_id, query, retriever, max_retries=20):
    """Rewrites a ToT query with an automatic retry mechanism for API and formatting errors."""

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": get_system_prompt()},
                        {"role": "user", "content": get_user_prompt(query[:3000], retriever)}
                    ],
                    temperature=0.3,
                    max_tokens=8000,
                )

                response_content = response.choices[0].message.content.strip()

                extracted_query = extract_query(response_content)
                extracted_thinking = extract_thinking(response_content)

                #Ensure model thinked and outputted the query
                if extracted_query is None or extracted_thinking is None:
                    print(f"[WARN] query_id={query_id} ({retriever}): wrong formatting, retrying...")
                    continue

                return query_id, retriever, extracted_query, extracted_thinking
            except Exception as e:
                print(f"Error on query {query_id}: {e}")
                #await asyncio.sleep(0.5)
        return query_id, retriever, None, None # Too long

async def main(model, experiment_name="sft"):
    for qf in query_files:
        print("READING", qf)
        df = pd.read_json(qf, lines=True)
        print("read")
        directory = Path(qf).parent

        temp_output_file = directory / f'rewritten-queries-{experiment_name}-temp.jsonl'
        final_output_file = directory / f'rewritten-queries-{experiment_name}.jsonl'

        if final_output_file.exists():
            print(f"Final output {final_output_file} already exists. Skipping {qf}...")
            continue

        processed_pairs = set()
        try:
            with open(temp_output_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Store a tuple of (query_id, retriever)
                    processed_pairs.add((data.get('query_id'), data.get('retriever')))
            print(f"Found {len(processed_pairs)} already processed query-retriever pairs. Resuming...")
        except FileNotFoundError:
            pass

        # Fixed: Checking against 'query_id' instead of 'id' to match your row iteration
        print(f"Starting batch generation for queries...")
            
        tasks = [
            rewrite_query(model, row['query_id'], row['query'], retr) 
            for _, row in df.iterrows()
            for retr in RETRIEVERS
            if (row['query_id'], retr) not in processed_pairs
        ]

        print(f"Starting batch generation for {len(tasks)} tasks...")
        
        if tasks:
            print(f"Starting batch generation for {len(tasks)} tasks...")
            
            async with aiofiles.open(temp_output_file, mode='a') as out_file:
                for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Rewriting"):
                    query_id, retriever, query, thinking = await f
                    
                    if query is not None:
                        result_dict = {"query_id": query_id, "retriever": retriever, "query": query, "thinking": thinking}
                        await out_file.write(json.dumps(result_dict) + "\n")
        else:
            print("No tasks left to process for this file.")
        
        print("Batch generation complete! Merging results...")
        
        results_df = pd.read_json(temp_output_file, lines=True)
        
        # 1. Pivot the queries so they become flat columns like 'BM25_query', 'ColBERT_query'
        queries_pivot = results_df.pivot(index='query_id', columns='retriever', values='query')
        queries_pivot.columns = [f"{col}_query" for col in queries_pivot.columns]
        
        # 2. Group the thinking into a nested dictionary per query_id 
        # Output structure: {'BM25': '...', 'ColBERT': '...'}
        thinking_nested = (
            results_df.groupby('query_id')
            .apply(lambda x: dict(zip(x['retriever'], x['thinking'])))
            .reset_index(name='thinking')
        )
        
        # 3. Combine the flattened queries and the nested thinking
        grouped_results = queries_pivot.reset_index().merge(thinking_nested, on='query_id')
        
        # 4. Merge back into the original dataframe
        final_df = df.merge(grouped_results, on='query_id', how='left')
        final_df.to_json(final_output_file, orient='records', lines=True)
        
        print(f"Successfully processed and merged {qf}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate rewritten queries via vLLM.")
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["base", "simpo"], 
        default="base",
        help="Toggle between the base SFT model and the SimPO aligned model."
    )
    args = parser.parse_args()

    # 2. Map the choice to the actual vLLM served-model-name
    model_mapping = {
        "base": "tip-of-tongue-rewriter-base", 
        "simpo": "simpo-aligned-rewriter"
    }

    # 3. Replace your hardcoded MODEL_NAME
    MODEL_NAME = model_mapping[args.model_type]
    print("Rewriting with model", MODEL_NAME)
    asyncio.run(main(MODEL_NAME, "sft" if args.model_type == 'base' else 'simpo'))