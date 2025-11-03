from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm # To see progress
import csv

import sys

max_int = sys.maxsize

while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

tokenizer = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')

passage_lengths = []
batch_size = 5000
sample = 1_000_000
count = 0
batch_texts = []

print(f"Reading and tokenizing passages in batches of {batch_size}...")

with open('./dataset/corpus_colbert.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    
    # We'll use tqdm on the reader itself
    for row in tqdm(reader, desc="Processing passages"):
        # Assuming TSV format: pid \t passage_text
        pid, text = row[0], row[1]
        batch_texts.append(text)

        count += 1
        
        # When the batch is full, process it
        if len(batch_texts) >= batch_size:
            # Tokenize the entire batch at once
            tokenized_batch = tokenizer(batch_texts, truncation=False)
            
            # Get the lengths from the tokenized batch
            passage_lengths.extend([len(ids) for ids in tokenized_batch.input_ids])
            
            # Reset the batch
            batch_texts = []
        
        if (count > sample): break

    # --- IMPORTANT ---
    # Process the final batch (which might be smaller than batch_size)
    if batch_texts:
        tokenized_batch = tokenizer(batch_texts, truncation=False)
        passage_lengths.extend([len(ids) for ids in tokenized_batch.input_ids])

print("\nTokenization complete. Calculating stats...")

lengths_arr = np.array(passage_lengths)

print(f"Total passages analyzed: {len(lengths_arr)}")
print(f"Min length:    {lengths_arr.min()}")
print(f"Mean length:   {lengths_arr.mean():.2f}")
print(f"Median length: {np.median(lengths_arr)}")
print(f"Max length:    {lengths_arr.max()}")

# --- This is the most important part ---
# Check percentiles against your doc_maxlen
# Let's assume your doc_maxlen is 180
doc_maxlen = 220

print("\n--- Percentiles ---")
p_80 = np.percentile(lengths_arr, 80)
p_90 = np.percentile(lengths_arr, 90)
p_95 = np.percentile(lengths_arr, 95)
p_99 = np.percentile(lengths_arr, 99)

print(f"80th percentile: {p_80:.0f}")
print(f"90th percentile: {p_90:.0f}")
print(f"95th percentile: {p_95:.0f}")
print(f"99th percentile: {p_99:.0f}")

# Calculate how many passages are being truncated
truncated_count = (lengths_arr > doc_maxlen).sum()
truncation_percent = (truncated_count / len(lengths_arr)) * 100

print(f"\n--- Truncation Report (for doc_maxlen={doc_maxlen}) ---")
print(f"{truncated_count} out of {len(lengths_arr)} passages ({truncation_percent:.2f}%) are truncated.")