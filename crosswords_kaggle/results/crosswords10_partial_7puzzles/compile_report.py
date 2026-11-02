"""Compile the accompanying user-supplied console log without inventing absent metrics."""
import hashlib
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'console.txt').read_text(encoding='utf-8')
methods = ('tot_dfs', 'heuristic_bestfirst')
records = []
puzzle = None
for line in source.splitlines():
    header = re.fullmatch(r'PUZZLE (\d+)/(\d+) (crossword-\d+)', line)
    if header:
        index, planned, qid = header.groups()
        puzzle = dict(puzzle_index=int(index), question_id=qid)
        continue
    if not line.strip():
        continue
    result = re.fullmatch(r'\s*(tot_dfs|heuristic_bestfirst)\s+words=(\d+)% letters=(\d+)% '
                          r'solved=(True|False) calls=(\d+) tokens=(\d+) stop=(\w+) invalid_responses=(\d+)', line)
    if not result or puzzle is None:
        raise ValueError('Unrecognized log line: ' + line)
    method, words, letters, solved, calls, tokens, stop, invalid = result.groups()
    records.append({**puzzle, 'method': method, 'word_accuracy': int(words)/100,
                    'letter_accuracy': int(letters)/100, 'solved': solved == 'True',
                    'calls': int(calls), 'total_tokens': int(tokens), 'stop_reason': stop,
                    'invalid_responses': int(invalid), 'recovered_from_console': True,
                    'prediction': None, 'input_tokens': None, 'output_tokens': None,
                    'inference_seconds': None, 'expanded_states': None,
                    'seed': None, 'model': None})
assert len(records) == 14
keys = {(r['question_id'], r['method']) for r in records}
assert len(keys) == len(records), 'Duplicate results'
ids = list(dict.fromkeys(r['question_id'] for r in records))
assert len(ids) == 7 and all((qid, method) in keys for qid in ids for method in methods)
summary = []
for method in methods:
    group = [r for r in records if r['method'] == method]
    summary.append(dict(method=method, puzzles=len(group), solved=sum(r['solved'] for r in group),
                        mean_word_accuracy=mean(r['word_accuracy'] for r in group),
                        mean_letter_accuracy=mean(r['letter_accuracy'] for r in group),
                        mean_calls=mean(r['calls'] for r in group),
                        mean_total_tokens=mean(r['total_tokens'] for r in group),
                        total_calls=sum(r['calls'] for r in group),
                        total_tokens=sum(r['total_tokens'] for r in group),
                        invalid_responses=sum(r['invalid_responses'] for r in group)))
paired = []
for qid in ids:
    pair = {r['method']: r for r in records if r['question_id'] == qid}
    a, b = pair['tot_dfs'], pair['heuristic_bestfirst']
    paired.append(dict(question_id=qid,
                       dfs_word_accuracy=a['word_accuracy'], heuristic_word_accuracy=b['word_accuracy'],
                       dfs_letter_accuracy=a['letter_accuracy'], heuristic_letter_accuracy=b['letter_accuracy'],
                       word_accuracy_difference=b['word_accuracy']-a['word_accuracy'],
                       letter_accuracy_difference=b['letter_accuracy']-a['letter_accuracy'],
                       token_difference=b['total_tokens']-a['total_tokens']))
a, b = summary
comparison = dict(direction='heuristic_bestfirst minus tot_dfs',
                  mean_word_accuracy_difference=b['mean_word_accuracy']-a['mean_word_accuracy'],
                  mean_letter_accuracy_difference=b['mean_letter_accuracy']-a['mean_letter_accuracy'],
                  mean_call_difference=b['mean_calls']-a['mean_calls'],
                  mean_token_difference=b['mean_total_tokens']-a['mean_total_tokens'],
                  token_increase_fraction=b['total_tokens']/a['total_tokens']-1,
                  word_accuracy_wins=sum(p['word_accuracy_difference']>0 for p in paired),
                  word_accuracy_ties=sum(p['word_accuracy_difference']==0 for p in paired),
                  word_accuracy_losses=sum(p['word_accuracy_difference']<0 for p in paired))
compiled = dict(provenance=dict(source='User-pasted console output',
                               source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                               planned_puzzles=10, reported_puzzles=7, reported_method_runs=14,
                               unreported_puzzle_indices=[8,9,10],
                               unreported_status='Unknown; absent from supplied output, not scored as failures',
                               missing_fields='Null fields were not present in console output; original files were not used'),
                results=records, summary=summary, paired_comparisons=paired, comparison=comparison)
(ROOT/'crosswords10_partial_results.json').write_text(json.dumps(compiled, indent=2)+'\n', encoding='utf-8')
(ROOT/'crosswords10_partial_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
lines = ['# Crossword benchmark: partial report', '',
         'Source: user-supplied console output. Seven paired puzzles, 14 method runs, out of 10 planned puzzles.',
         'Puzzles 8–10 are unreported; their results and completion status are unknown.', '',
         '| Method | Mean word accuracy | Mean letter accuracy | Fully solved | Mean calls | Mean tokens | Total tokens | Invalid responses |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in summary:
    lines.append(f'| {row["method"]} | {row["mean_word_accuracy"]:.2%} | {row["mean_letter_accuracy"]:.2%} | '
                 f'{row["solved"]}/{row["puzzles"]} | {row["mean_calls"]:.2f} | {row["mean_total_tokens"]:,.2f} | '
                 f'{row["total_tokens"]:,} | {row["invalid_responses"]} |')
lines += ['', '## Per-puzzle scores', '',
          '| Puzzle | DFS words | Heuristic words | DFS letters | Heuristic letters | Heuristic − DFS tokens |',
          '|---|---:|---:|---:|---:|---:|']
for p in paired:
    lines.append(f'| {p["question_id"]} | {p["dfs_word_accuracy"]:.0%} | {p["heuristic_word_accuracy"]:.0%} | '
                 f'{p["dfs_letter_accuracy"]:.0%} | {p["heuristic_letter_accuracy"]:.0%} | {p["token_difference"]:+,} |')
lines += ['', '## Interpretation', '',
          f'The heuristic has higher mean word accuracy by {comparison["mean_word_accuracy_difference"]*100:.2f} percentage points '
          f'and higher mean letter accuracy by {comparison["mean_letter_accuracy_difference"]*100:.2f} percentage points.',
          f'On word accuracy it wins on {comparison["word_accuracy_wins"]} puzzles, ties on {comparison["word_accuracy_ties"]}, '
          f'and loses on {comparison["word_accuracy_losses"]}.',
          f'The heuristic uses {comparison["token_increase_fraction"]:.2%} more tokens in total '
          f'and {comparison["mean_call_difference"]:.2f} more calls per puzzle on average.',
          'Neither method solves a complete puzzle. Every reported run ends at the expansion budget.',
          'The search now progresses beyond initial format failures, but candidate quality, evaluation and '
          'search coverage still need investigation. The log alone cannot establish which caused the low scores.',
          'These seven puzzles from one partial run support a descriptive comparison only; they do not establish '
          'a general advantage or that more expansions would solve the puzzles.', '',
          '## Recovery limitations', '',
          'Predicted grids, raw model responses, exact expansion counts, runtime, input/output token split, '
          'model and seed were not supplied. They are null in JSON. A null prediction means unavailable, '
          'not necessarily an empty grid. Reported percentages are retained as printed.',
          'No outcomes are fabricated for absent puzzles. Source console text and this compilation script are included.']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(dict(summary=summary,comparison=comparison),indent=2))
