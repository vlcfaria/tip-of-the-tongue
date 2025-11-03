import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #same order as nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "32"

import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import math

DATASET_PATH = './dataset/corpus.jsonl'
OUTPUT_PATH = './dataset/corpus_splade_tokenized_overlap.tsv'
TRANSLATION_PATH = './dataset/idx_to_pid_splade_overlap.json'

CHUNK_SIZE = 256 - 2 #Account for CLS and SEP
OVERLAP = 64
DOC_BATCH_SIZE = 1000
TOKENIZER_NAME = 'naver/splade-v3'
NUM_PROCS = 32

print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

def process_and_chunk_batch(batch):
    """
    This function processes a batch of documents (as a dict) 
    and returns a dict of the resulting passages.

    This version prepends the *title* to *every* chunk
    and adjusts the text chunk size to respect the token limit.
    """
    
    batch_passage_texts = []
    batch_passage_pids = []
    
    for i, doc_id in enumerate(batch['doc_id']):
        title_str = batch.get('title', [''])[i]
        text_str = batch.get('text', [''])[i]

        #Tokenize the title
        title_prefix = f"{title_str}. " if title_str else ""
        title_encoding = tokenizer(
            title_prefix, 
            add_special_tokens=False, 
            max_length=CHUNK_SIZE, 
            truncation=True
        )
        
        title_tokens = title_encoding.input_ids
        title_len = len(title_tokens)
        
        #Calculate remaining space for the main text
        text_chunk_size = CHUNK_SIZE - title_len

        #Handle edge case: Title is longer than or equals the chunk size
        if text_chunk_size <= 0:
            batch_passage_texts.append(title_prefix)
            batch_passage_pids.append(doc_id)
            print(f'Warning! {doc_id} title above limit on tokenization')
            continue

        text_stride = text_chunk_size - OVERLAP
        if text_stride < 1:
            text_stride = 1

        text_encoding = tokenizer(
            text_str,
            add_special_tokens=False,
            truncation=False, # We will do our own chunking
            padding=False,
            return_offsets_mapping=True
        )
        text_tokens = text_encoding.input_ids
        text_offsets = text_encoding.offset_mapping

        start = 0
        while True:
            end = start + text_chunk_size
            
            chunk_text_ids = text_tokens[start:end]
            chunk_text_offsets = text_offsets[start:end]

            if not chunk_text_ids:
                break
            
            # Find valid character offsets
            valid_offsets = [o for o in chunk_text_offsets if o is not None]
            
            if not valid_offsets:
                 # This chunk might be just padding or empty, skip
                if end >= len(text_tokens):
                    break
                start += text_stride
                continue

            # Get the character boundaries from the *original text string*
            start_char = valid_offsets[0][0]
            end_char = valid_offsets[-1][1]
            
            passage_text_body = text_str[start_char:end_char]
            
            # 7. Assemble the final passage
            final_passage_text = f"{title_prefix}{passage_text_body}"
            
            batch_passage_texts.append(final_passage_text.strip())
            batch_passage_pids.append(doc_id)
            
            # If this is the last chunk, stop
            if end >= len(text_tokens):
                break
            
            # Move the window
            start += text_stride
    
    return {
        "passage_text": batch_passage_texts,
        "original_doc_id": batch_passage_pids
    }

def main():
    print(f"Loading dataset from {DATASET_PATH}...")

    ds = load_dataset('json', data_files=DATASET_PATH, split='train')
    
    print(f"Original dataset size: {len(ds)} documents")

    print(f"Starting parallel processing with {NUM_PROCS} workers...")

    passage_ds = ds.map(
        process_and_chunk_batch,
        batched=True,
        batch_size=DOC_BATCH_SIZE,
        num_proc=NUM_PROCS,
        remove_columns=ds.column_names
    )

    print(f"\nFinished processing. Total passages created: {len(passage_ds)}")

    # 1. Save the passage-to-doc_id mapping
    print(f"Saving index-to-document_id mapping to {TRANSLATION_PATH}...")
    # We can just extract the column directly
    pids = passage_ds['original_doc_id']
    with open(TRANSLATION_PATH, 'w') as outp:
        json.dump(pids, outp)
    
    # 2. Save the final TSV
    print(f"Adding passage index and saving to {OUTPUT_PATH}...")
    
    # Add the new passage_idx column (0, 1, 2, ...)
    passage_ds = passage_ds.add_column("passage_idx", range(len(passage_ds)))
    
    # Select and reorder columns for the final TSV
    passage_ds = passage_ds.select_columns(["passage_idx", "passage_text"])
    
    # Save to TSV, also in parallel and batched
    passage_ds.to_csv(
        OUTPUT_PATH,
        sep='\t',
        header=False,
        index=False,
        batch_size=DOC_BATCH_SIZE * 10 # Use a larger batch for writing
    )
    
    print("Done.")

if __name__ == '__main__':
    main()