"""Recover only printed metrics from Game24 logs; never invent missing results."""
import argparse
import ast
import base64
import hashlib
import json
import re
import zlib
from pathlib import Path


def recover(text, notebook):
    nb = json.loads(Path(notebook).read_text(encoding='utf-8'))
    cell = next(c for c in nb['cells'] if c['metadata'].get('file') == 'data/game24/game24_test.jsonl')
    tree = ast.parse(''.join(cell['source']))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode')
    raw = zlib.decompress(base64.b64decode(ast.literal_eval(call.args[0])))
    assert hashlib.sha256(raw).hexdigest() == cell['metadata']['sha256']
    puzzles = [json.loads(line) for line in raw.decode().splitlines()]
    rows, headers, seen = [], {}, set()
    pending = None
    for line in text.splitlines():
        header = re.fullmatch(r'\s*PUZZLE (\d+)/(\d+) ([\w-]+): (\[[\d, ]+\])\s*', line)
        result = re.fullmatch(r'\s*(heuristic_bestfirst|without_\w+): solved=(True|False) calls=(\d+) tokens=(\d+)\s*', line)
        if header:
            index, total, qid, nums = header.groups()
            index = int(index)
            if int(total) != len(puzzles) or not 1 <= index <= len(puzzles):
                raise ValueError('Unexpected puzzle count/index')
            expected = puzzles[index - 1]
            if qid != expected['id'] or json.loads(nums) != expected['numbers']:
                raise ValueError('Log does not match reference test dataset at puzzle ' + str(index))
            headers[index] = expected
            pending = index
        elif result:
            if pending is None:
                raise ValueError('Result line without puzzle header')
            method, solved, calls, tokens = result.groups()
            key = (pending, method)
            if key in seen:
                raise ValueError('Duplicate printed result: ' + str(key))
            seen.add(key)
            rows.append({'task': 'game24', 'question_id': puzzles[pending - 1]['id'],
                         'numbers': puzzles[pending - 1]['numbers'], 'puzzle_index': pending,
                         'method': method, 'correct': solved == 'True',
                         'calls': int(calls), 'total_tokens': int(tokens),
                         'prediction': None, 'seed': None, 'model': None,
                         'stop_reason': None, 'input_tokens': None, 'output_tokens': None,
                         'inference_seconds': None, 'unknown_cost': None,
                         'recovered_from_console': True,
                         'recovery_source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'reference_dataset_sha256': hashlib.sha256(raw).hexdigest(),
                         'prediction_availability': 'not_printed',
                         'configuration_verified': False})
            pending = None
    if not rows or len({r['method'] for r in rows}) != 1:
        raise ValueError('Expected printed results for exactly one method')
    present = {r['puzzle_index'] for r in rows}
    missing = [{'puzzle_index': i, 'question_id': puzzle['id'], 'numbers': puzzle['numbers'],
                'reason': 'result_line_missing' if i in headers else 'puzzle_absent_from_paste'}
               for i, puzzle in enumerate(puzzles, 1) if i not in present]
    summary = {'method': rows[0]['method'], 'planned_puzzles': len(puzzles),
               'recovered_results': len(rows), 'correct': sum(r['correct'] for r in rows),
               'accuracy_on_recovered_only': sum(r['correct'] for r in rows) / len(rows),
               'missing_results': len(missing),
               'total_calls_on_recovered': sum(r['calls'] for r in rows),
               'total_tokens_on_recovered': sum(r['total_tokens'] for r in rows),
               'mean_calls_on_recovered': sum(r['calls'] for r in rows) / len(rows),
               'mean_tokens_on_recovered': sum(r['total_tokens'] for r in rows) / len(rows),
               'note': 'Console recovery only. Null prediction means expression unavailable, not unsolved. '
                       'Seed/model/config are unverified; full-experiment accuracy is unknown. '
                       'Keep this provenance when combining with original results.'}
    return rows, missing, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--notebook', type=Path, default=Path('benchmark_game24/Game24_without_cost.ipynb'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.log.read_text(encoding='utf-8-sig')
    rows, missing, summary = recover(source, args.notebook)
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = 'game24_test_' + summary['method'] + '_recovered'
    for suffix, records in [('results', rows), ('missing', missing)]:
        (args.output / (prefix + '_' + suffix + '.jsonl')).write_text(
            ''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    (args.output / (prefix + '_summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (args.output / (prefix + '_console.txt')).write_text(source, encoding='utf-8')
    print(json.dumps(summary, indent=2))
