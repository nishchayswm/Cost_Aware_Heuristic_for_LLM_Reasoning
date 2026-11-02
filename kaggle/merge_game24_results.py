"""Merge the six named result downloads, checking experiment and puzzle identity.

Usage: python kaggle/merge_game24_results.py downloaded/*_results.jsonl --output merged_game24_results.jsonl
"""
import argparse
import json
from pathlib import Path

METHODS = ('heuristic_bestfirst', 'without_quality', 'without_success',
           'without_depth', 'without_cost', 'without_diversity')
IDENTITY = ('task', 'model', 'seed', 'dataset_sha256', 'selection_sha256',
            'base_config_sha256')


def merge(paths, output, expected_puzzles=100):
    rows, seen, numbers, identity = [], set(), {}, None
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            current = tuple(row[key] for key in IDENTITY)
            if identity is None:
                identity = current
            if current != identity:
                raise ValueError('Cannot merge different datasets, selections, models, seeds or configs')
            if row['method'] not in METHODS:
                raise ValueError('Unknown method: ' + str(row['method']))
            key = (row['question_id'], row['seed'], row['method'])
            if key in seen:
                raise ValueError('Duplicate result: ' + str(key))
            seen.add(key)
            nums = row['numbers']
            if row['question_id'] in numbers and numbers[row['question_id']] != nums:
                raise ValueError('Puzzle numbers differ: ' + row['question_id'])
            numbers[row['question_id']] = nums
            rows.append(row)
    if len(numbers) != expected_puzzles:
        raise ValueError(f'Expected {expected_puzzles} puzzles, found {len(numbers)}')
    for qid in numbers:
        if {r['method'] for r in rows if r['question_id'] == qid} != set(METHODS):
            raise ValueError('Missing methods for puzzle: ' + qid)
    rows.sort(key=lambda r: (r['question_id'], METHODS.index(r['method'])))
    # Never overwrite a previous merged experiment.
    with Path(output).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('merged_game24_results.jsonl'))
    args = parser.parse_args()
    # Expand globs here as well, for shells that pass them through literally.
    paths = []
    for path in args.files:
        paths.extend(sorted(path.parent.glob(path.name)) if '*' in path.name else [path])
    count = merge(paths, args.output)
    print(f'Merged {count} results from 100 puzzles into {args.output}')
