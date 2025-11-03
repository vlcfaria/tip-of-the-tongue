import argparse
import csv
import json
from pathlib import Path
from typing import Generator

def find_jsonl_files(directory: Path) -> Generator[Path, None, None]:
    """Recursively finds all .jsonl files in a given directory."""
    print(f"🔍 Searching for .jsonl files in '{directory}'...")
    yield from directory.rglob('*.jsonl')

def convert_file(jsonl_path: Path) -> None:
    """
    Converts a single JSONL file to a TSV file.

    The new file will have the same name but with a .tsv extension.
    The header of the TSV is determined by the keys of the first JSON object in the file.
    """
    tsv_path = jsonl_path.with_suffix('.tsv')
    print(f"🔄 Converting '{jsonl_path.name}' -> '{tsv_path.name}'")

    with jsonl_path.open('r', encoding='utf-8') as infile, \
            tsv_path.open('w', newline='', encoding='utf-8') as outfile:

        writer = csv.writer(outfile, delimiter='\t')

        for line_num, line in enumerate(infile, 1):
            # Skip empty lines
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                # Ensure the parsed line is a dictionary (JSON object)
                if not isinstance(data, dict):
                    print(f"  ⚠️ Warning: Line {line_num} is not a JSON object. Skipping.")
                    continue

                # Write the data row, ensuring values align with the header
                writer.writerow([data['query_id'], data['query'].replace('\n', ' ')])

            except json.JSONDecodeError:
                print(f"  ⚠️ Warning: Could not decode JSON on line {line_num}. Skipping.")
    
    print(f"  ✅ Successfully converted.")


def main():
    """Main function to parse arguments and start the conversion process."""
    parser = argparse.ArgumentParser(
        description="Recursively find and convert JSONL files to TSV format.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "root_directory",
        type=str,
        help="The path to the root directory to search for .jsonl files."
    )
    args = parser.parse_args()
    
    root_path = Path(args.root_directory)

    if not root_path.is_dir():
        print(f"Error: The specified path '{root_path}' is not a valid directory.")
        return

    jsonl_files = list(find_jsonl_files(root_path))
    
    if not jsonl_files:
        print("No .jsonl files were found.")
        return
        
    print(f"Found {len(jsonl_files)} file(s) to convert.\n")

    for file_path in jsonl_files:
        convert_file(file_path)
    
    print("\n🎉 All tasks complete.")


if __name__ == "__main__":
    main()