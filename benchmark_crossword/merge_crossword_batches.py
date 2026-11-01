"""Merge five extracted Kaggle batch downloads without rerunning inference.

Example: python crosswords_kaggle/merge_crossword_batches.py downloads --output crossword_merged
The downloads directory can contain extracted batch folders or the individually downloaded
*_results.jsonl and matching *_manifest.json files. This script does not extract ZIPs.
"""
import argparse
from pathlib import Path
from crossword_batches import merge_batches

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.directory.rglob('crosswords10_batch*_results.jsonl'))
    rows = merge_batches(files, args.output)
    print(f'Merged {len(rows)} results: ten puzzles, two methods. Reports: {args.output}')
