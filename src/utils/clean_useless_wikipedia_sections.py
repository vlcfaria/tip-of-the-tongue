import os
import re
from datasets import load_dataset
import argparse
import unicodedata

# --- Configuration ---
INPUT_PATH = './dataset/trec-tot-2025-corpus.jsonl'
OUTPUT_PATH = './dataset/corpus-cleaned-sections.jsonl'
NUM_PROCS = 32

# --- Regex Patterns ---
FOOTER_PATTERN = re.compile(
    r'\n(?:\s*)(?:See also|References|Notes|Further reading|External links|Bibliography)(?:\s*)\n', 
    re.IGNORECASE
)
NEWLINE_PATTERN = re.compile(r'\n{2,}')

def normalize_unicode(text: str) -> str:
    """
    Normalizes Unicode characters and strips diacritics/accents 
    for exact-match lexical retrieval.
    """
    if not isinstance(text, str):
        return text
        
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def clean_flat_text_for_bm25(batch):
    """
    Original cleaning function: Returns a single flat string per document,
    with Unicode normalization applied to both text and titles.
    """
    cleaned_texts = []
    
    for text_str in batch.get('text', []):
        if not text_str:
            cleaned_texts.append("")
            continue
            
        split_text = FOOTER_PATTERN.split(text_str)
        main_text = split_text[0] 
        main_text = NEWLINE_PATTERN.sub('. ', main_text)
        main_text = main_text.replace('\n', ' ').strip()
        
        # Apply unicode normalization to the body text
        main_text = normalize_unicode(main_text)
        cleaned_texts.append(main_text)
        
    result = {"text": cleaned_texts}
    
    # Also normalize titles if they exist in the dataset
    if 'title' in batch:
        cleaned_titles = [normalize_unicode(t) if t else "" for t in batch['title']]
        result["title"] = cleaned_titles
        
    return result

def clean_structured_sections_for_colbert(batch):
    """
    Contextual cleaning function: Returns a list of section dictionaries.
    """
    cleaned_sections_batch = []
    
    for text_str in batch.get('text', []):
        if not text_str:
            cleaned_sections_batch.append([])
            continue
            
        split_text = FOOTER_PATTERN.split(text_str)
        main_text = split_text[0] 
        
        blocks = re.split(r'\n{2,}', main_text)
        sections = []
        current_heading = ""
        current_text = []
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
                
            # Heuristic for headings
            if len(block) < 80 and not block.endswith(('.', '?', '!', '"', "'")):
                if current_text:
                    sections.append({
                        "heading": current_heading, 
                        "text": " ".join(current_text)
                    })
                    current_text = []
                current_heading = block
            else:
                block = block.replace('\n', ' ')
                current_text.append(block)
                
        if current_text:
            sections.append({
                "heading": current_heading, 
                "text": " ".join(current_text)
            })
            
        cleaned_sections_batch.append(sections)
        
    return {"sections": cleaned_sections_batch}

def main():
    parser = argparse.ArgumentParser(description="Clean Wikipedia corpus.")
    parser.add_argument(
        '--parse-sections', 
        action='store_true', 
        help="Outputs structured sections for ColBERT. If omitted, defaults to flat text for BM25."
    )
    args = parser.parse_args()

    # The toggle is now controlled by the CLI argument
    PARSE_SECTIONS = args.parse_sections

    print(f"Loading dataset from {INPUT_PATH}...")
    ds = load_dataset('json', data_files=INPUT_PATH, split='train')
    print(f"Original dataset size: {len(ds)} documents")
    
    if PARSE_SECTIONS:
        print("Mode: STRUCTURED SECTIONS (Optimized for ColBERT Contextual Chunking)")
        map_function = clean_structured_sections_for_colbert
        cols_to_remove = ['text'] if 'text' in ds.column_names else [] 
    else:
        print("Mode: FLAT TEXT (Optimized for BM25)")
        map_function = clean_flat_text_for_bm25
        cols_to_remove = []

    print(f"Cleaning text using {NUM_PROCS} workers...")

    cleaned_ds = ds.map(
        map_function,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROCS,
        remove_columns=cols_to_remove
    )
    
    print(f"Saving cleaned dataset to {OUTPUT_PATH}...")
    cleaned_ds.to_json(
        OUTPUT_PATH,
        batch_size=10000,
        num_proc=NUM_PROCS
    )
    
    print("Done. You can now inspect the cleaned corpus.")

if __name__ == '__main__':
    main()