import json
import csv
from pathlib import Path

def convert_jsonl_to_tsv(input_filepath):
    """
    Converts a JSONL file to a TSV format and saves it in the same
    directory as the input file under the name 'rewritten_queries.jsonl'.
    """
    # Create a Path object to easily manipulate file paths
    input_path = Path(input_filepath)
    
    # Construct the output path: same parent directory, specific filename
    output_path = input_path.parent / 'rewritten-queries.tsv'
    
    # Open the input file for reading and output file for writing
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        
        # Set up the CSV writer with a tab delimiter
        tsv_writer = csv.writer(outfile, delimiter='\t')
        
        # Process the JSONL file line by line
        for line in infile:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Parse the JSON object
            data = json.loads(line)
            
            # Extract the required fields (defaulting to empty string if missing)
            qid = data.get('query_id', '')
            query = data.get('query', '')
            
            # Write the row directly to the file (no header)
            tsv_writer.writerow([qid, query])
            
    return output_path

# Example usage:
if __name__ == "__main__":
    # You can put a full or relative path here, e.g., 'data/my_folder/input.jsonl'
    files = [
        'queries/2023/test/rewritten-queries.jsonl',
        'queries/2023/train/rewritten-queries.jsonl',
        'queries/2024/test-partial/rewritten-queries.jsonl',
        'queries/2024/train/rewritten-queries.jsonl'
    ]

    for f in files:
        saved_path = convert_jsonl_to_tsv(f)
