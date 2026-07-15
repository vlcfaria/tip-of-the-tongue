import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #same order as nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "32"

import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import math

DATASET_PATH = './dataset/corpus-cleaned-sections.jsonl'
OUTPUT_PATH = './dataset/corpus_colbert_tokenized_overlap.tsv'
TRANSLATION_PATH = './dataset/idx_to_pid_colbert_overlap.json'

CHUNK_SIZE = 180 - 2 #Account for CLS and SEP
OVERLAP = 16
DOC_BATCH_SIZE = 1000
TOKENIZER_NAME = 'colbert-ir/colbertv2.0'
NUM_PROCS = 24
MIN_TOKENS = 35
MAX_PREFIX_TOKENS = 64 # Prevent unusually long titles/headings from consuming the whole chunk

print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

def process_and_chunk_batch(batch):
    """
    Processes a batch of documents containing sections.
    Prepends 'Title - Heading. ' dynamically based on where the sliding 
    window starts, allowing text to freely flow across section boundaries.
    """
    batch_passage_texts = []
    batch_passage_pids = []
    
    for i, doc_id in enumerate(batch['id']):
        title_str = batch.get('title', [''])[i]
        
        # Extract sections. Default to empty list if missing.
        sections = batch.get('sections', [[]])[i] 
        
        if not sections:
            continue
            
        doc_text = ""
        section_boundaries = [] # Will store: (start_char_idx, end_char_idx, heading_string)
        
        # 1. Reconstruct the full document text while tracking where sections start/end
        for sec in sections:
            heading = sec.get('heading', '').strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
            body = sec.get('text', '').strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
            
            sec_str = ""
            if heading:
                suffix = "" if heading[-1] in ".!?:;" else "."
                # Simply inject the label. The previous body's trailing space handles the gap.
                sec_str += f"Section: {heading}{suffix} "
                
            if body:
                suffix = "" if body[-1] in ".!?" else "."
                sec_str += f"{body}{suffix} "
            
            start_c = len(doc_text)
            doc_text += sec_str
            end_c = len(doc_text)
            
            section_boundaries.append((start_c, end_c, heading))
            
        text_encoding = tokenizer(
            doc_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_offsets_mapping=True
        )
        text_tokens = text_encoding.input_ids
        text_offsets = text_encoding.offset_mapping
        
        start = 0
        while start < len(text_tokens):
            # 3. Find the valid character start offset for the current token window
            start_char = None
            for offset in text_offsets[start:]:
                if offset is not None:
                    start_char = offset[0]
                    break
                    
            if start_char is None:
                break # Reached the end with no valid characters
                
            # 4. Determine which section this chunk is currently starting in
            active_heading = ""
            for (sec_start, sec_end, h) in section_boundaries:
                if sec_start <= start_char < sec_end:
                    active_heading = h
                    break
                    
            # 5. Construct the dynamic prefix
            if title_str and active_heading:
                suffix = "" if active_heading[-1] in ".!?" else "."
                prefix_str = f"{title_str} - Section: {active_heading}{suffix} "
            elif title_str:
                suffix = "" if title_str[-1] in ".!?" else "."
                prefix_str = f"{title_str}{suffix} "
            elif active_heading:
                suffix = "" if active_heading[-1] in ".!?" else "."
                prefix_str = f"Section: {active_heading}{suffix} "
            else:
                prefix_str = ""
                
            # Tokenize prefix to find its exact length. Truncate if wildly long.
            prefix_encoding = tokenizer(
                prefix_str,
                add_special_tokens=False,
                max_length=MAX_PREFIX_TOKENS,
                truncation=True
            )
            prefix_tokens = prefix_encoding.input_ids
            
            # If the prefix was truncated, decode it back to string to ensure clean text
            if len(prefix_tokens) == MAX_PREFIX_TOKENS:
                prefix_str = tokenizer.decode(prefix_tokens).strip() + " "
                
            prefix_len = len(prefix_tokens)
            available_space = CHUNK_SIZE - prefix_len
            
            # 6. Extract the chunk body using the remaining available space
            end = start + available_space
            chunk_text_ids = text_tokens[start:end]
            
            # Discard if it's too short AND it's not the very first chunk
            if len(chunk_text_ids) < MIN_TOKENS and start > 0:
                break
                
            chunk_offsets = [o for o in text_offsets[start:end] if o is not None]
            if not chunk_offsets:
                start += available_space
                continue
                
            chunk_start_char = chunk_offsets[0][0]
            chunk_end_char = chunk_offsets[-1][1]
            
            passage_text_body = doc_text[chunk_start_char:chunk_end_char]
            
            # 7. Deduplication Check
            clean_body = passage_text_body.lstrip()
            
            # Look for the exact string we embedded in Step 1
            expected_heading_text = f"Section: {active_heading}"
            
            if active_heading and clean_body.startswith(expected_heading_text):
                # Slice out the heading and strip any residual colons, periods, or spaces
                clean_body = clean_body[len(expected_heading_text):].lstrip(" .:")
                
            # 8. Assemble the final passage
            final_passage_text = f"{prefix_str}{clean_body}".strip()
            
            batch_passage_texts.append(final_passage_text)
            batch_passage_pids.append(doc_id)
            
            if end >= len(text_tokens):
                break
                
            prev_start = start
            start = end - OVERLAP
            
            if start <= prev_start:
                start = prev_start + 1
                
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

    print(f"Saving index-to-document_id mapping to {TRANSLATION_PATH}...")
    pids = passage_ds['original_doc_id']
    with open(TRANSLATION_PATH, 'w') as outp:
        json.dump(list(pids), outp)
    
    print(f"Adding passage index and saving to {OUTPUT_PATH}...")
    passage_ds = passage_ds.add_column("passage_idx", range(len(passage_ds)))
    passage_ds = passage_ds.select_columns(["passage_idx", "passage_text"])
    
    passage_ds.to_csv(
        OUTPUT_PATH,
        sep='\t',
        header=False,
        index=False,
        batch_size=DOC_BATCH_SIZE * 10
    )
    
    print("Done.")

if __name__ == '__main__':
    main()