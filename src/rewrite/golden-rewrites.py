import asyncio
import pandas as pd
from openai import AsyncOpenAI
import os
import json
import aiofiles
from tqdm import tqdm
from consts import get_retriever_prompt
from utils import extract_query, is_title_leaked, get_system_prompt, get_user_prompt

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

MAX_CONCURRENT_REQUESTS = 64
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

OUTPUT_FILE = "golden-rewrites-intermediate.jsonl"
FINAL_OUTPUT_FILE = "golden-rewrites-final.jsonl"
MODEL_NAME = "casperhansen/deepseek-r1-distill-llama-70b-awq"

async def generate_golden_rewrite(query_id, tot_query, retriever_type, title, max_retries=3):
    """Generates the Think-Then-Rewrite golden sequence for a specific retriever."""
    
    system_prompt = get_system_prompt()
    
    user_prompt = get_user_prompt(tot_query, retriever_type)
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.6,
                    max_tokens=2500
                )
                content = response.choices[0].message.content.strip()

                if extract_query(content) is None:
                    print(f"[WARN] query_id={query_id} ({retriever_type}): no <query> tag found, retrying...")
                    await asyncio.sleep(1)
                    continue

                if is_title_leaked(content, title):
                    print(f"[WARN] query_id={query_id} ({retriever_type}): title leaked in generation, retrying...")
                    await asyncio.sleep(1)
                    continue

                return {"query_id": query_id, "retriever": retriever_type, "generation": content}
            except Exception as e:
                print(f"Error on doc {query_id} ({retriever_type}): {e}")
                await asyncio.sleep(2 ** attempt)
        return {"query_id": query_id, "retriever": retriever_type, "generation": None}

async def process_query(row):
    """Fires off all three retriever tasks for a single query concurrently."""
    retrievers = ["BM25", "SPLADE", "DENSE"]
    tasks = [
        generate_golden_rewrite(row['query_id'], row['query'], r, row['title'])
        for r in retrievers
    ]
    return await asyncio.gather(*tasks)

async def main():
    input_filepath = "queries/sft-train/queries.jsonl"
    df = pd.read_json(input_filepath, lines=True)

    processed_ids = set()
    try:
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                processed_ids.add(json.loads(line)['query_id'])
        print(f"Found {len(processed_ids)} completely processed documents. Resuming...")
    except FileNotFoundError:
        pass 

    df_to_process = df[~df['query_id'].isin(processed_ids)]
    print(f"Starting generation for {len(df_to_process)} documents (x3 retrievers each)...")
        
    tasks = [process_query(row) for _, row in df_to_process.iterrows()]
    
    async with aiofiles.open(OUTPUT_FILE, mode='a') as out_file:
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating Rewrites"):
            results = await f
            
            query_id = results[0]['query_id']
            result_dict = {
                "query_id": query_id,
                "rewrites": {res['retriever']: res['generation'] for res in results}
            }
            await out_file.write(json.dumps(result_dict) + "\n")
            await out_file.flush()
    
    print("Batch generation complete! Merging results...")
    
    results_df = pd.read_json(OUTPUT_FILE, lines=True)
    final_df = df.merge(results_df, on='query_id', how='left')
    final_df.to_json(FINAL_OUTPUT_FILE, orient='records', lines=True)
    
    print("Golden dataset finalized!")

if __name__ == "__main__":
    asyncio.run(main())