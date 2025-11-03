import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import spacy
import json
import argparse
import sys
from pathlib import Path

def process_files(root_dir: Path, nlp: spacy.Language) -> None:
    query_files = list(root_dir.rglob("queries.jsonl"))
    
    if not query_files:
        print(f"No 'queries.jsonl' files found in {root_dir}.")
        return

    print(f"Found {len(query_files)} file(s) to process:")
    for f in query_files:
        print(f"  - {f}")

    for file_path in query_files:
        print(f"\n--- Processing: {file_path} ---")
        
        output_path = file_path.with_name("queries.ner.jsonl")
        
        lines_data = []
        texts_to_process = []
        
        with file_path.open('r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    data = json.loads(line)
                    lines_data.append(data)
                    texts_to_process.append(data.get("query", ""))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in {file_path}: {line.strip()}", file=sys.stderr)

        if not texts_to_process:
            print("File is empty or contains no valid JSON. Skipping.")
            continue

        processed_docs = list(nlp.pipe(texts_to_process, batch_size=32))

        try:
            with output_path.open('w', encoding='utf-8') as outfile:
                for original_data, doc in zip(lines_data, processed_docs):
                    
                    entities = []
                    for ent in doc.ents:
                        entities.append({
                            "text": ent.text,
                            "label": ent.label_
                        })
                    
                    pos_counts = {}
                    pos_to_terms = {}
                    for token in doc:
                        pos_tag = token.pos_
                        term_text = token.text
                        pos_counts[pos_tag] = pos_counts.get(pos_tag, 0) + 1
                        pos_to_terms.setdefault(pos_tag, []).append(term_text)
                    
                    original_data["named_entities"] = entities
                    original_data["pos_counts"] = pos_counts
                    original_data["pos_to_terms"] = pos_to_terms
                    
                    outfile.write(json.dumps(original_data) + "\n")
                    
            print(f"Successfully created: {output_path}")
            
        except IOError as e:
            print(f"Error writing to {output_path}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Find 'queries.jsonl' files and add NER data using spaCy.")
    parser.add_argument("root_dir", type=str, help="The root directory to start searching from.")
    args = parser.parse_args()
    
    root_path = Path(args.root_dir)
    if not root_path.is_dir():
        print(f"Error: Path '{args.root_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)


    try:
        nlp = spacy.load("en_core_web_trf")
        print("Loaded 'en_core_web_trf' model.")
    except IOError:
        print(f"Error: Model 'en_core_web_trf' not found.", file=sys.stderr)
        print("Please download it by running:", file=sys.stderr)
        print("python -m spacy download en_core_web_trf", file=sys.stderr)
        sys.exit(1)

    process_files(root_path, nlp)

if __name__ == "__main__":
    main()