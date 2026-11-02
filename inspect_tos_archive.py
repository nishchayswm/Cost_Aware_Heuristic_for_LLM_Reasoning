"""Audit saved ToS traces without extracting archives or making model requests.

Replay uses ONLY each archived final model, not new generation or answer-guided
repair. Its outcomes are diagnostic and must not be reported as a fresh benchmark.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import zipfile

from benchmark_engine import Config
from tos_search import local_bfs


def inspect_archive(archive, output):
    with zipfile.ZipFile(archive) as zipped:
        paths = [name for name in zipped.namelist() if name.split('/')[-1] == 'full_results.jsonl']
        if len(paths) != 1:
            raise ValueError('Expected exactly one full_results.jsonl')
        if zipped.getinfo(paths[0]).file_size > 50_000_000:
            raise ValueError('Trace file exceeds audit size limit')
        rows = [json.loads(line) for line in zipped.read(paths[0]).decode('utf-8').splitlines() if line.strip()]
    audit = []
    for row in rows:
        if not str(row['method']).startswith('Thought of Search'):
            continue
        events = []
        try:
            value, _, status, expansions = local_bfs(row['search_model'], Config(**row['config']), events)
            error = None
        except (ValueError, SyntaxError, TypeError, KeyError, ArithmeticError, IndexError, RecursionError) as exc:
            value, status, expansions, error = None, 'model_error', None, str(exc)
        # Reference comparison happens after execution; it never enters the model or search.
        outcome = 'missing' if value is None else 'correct' if abs(value-row['expected']) < 1e-4 else 'wrong'
        audit.append(dict(source_id=row.get('source_id', row['problem_id']),
                          original_outcome=row['outcome'], original_status=row['status'],
                          original_error=row.get('last_error'), replay_outcome=outcome,
                          replay_answer=value, replay_status=status, replay_error=error,
                          local_expansions=expansions))
    errors = Counter(event['error'] for row in rows for event in row['trace'] if event.get('error'))
    summary = dict(archive=str(archive), original_outcomes=dict(Counter(r['original_outcome'] for r in audit)),
                   original_statuses=dict(Counter(r['original_status'] for r in audit)),
                   rejection_events=dict(errors),
                   final_model_replay_outcomes=dict(Counter(r['replay_outcome'] for r in audit)),
                   note='Offline replay of archived final code only; not a new Qwen benchmark.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output/'case_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    report = '# ToS archive audit\n\n' + summary['note'] + '\n\n'
    report += 'Original outcomes: ' + json.dumps(summary['original_outcomes']) + '.\n\n'
    report += 'Rejection messages count events, not distinct failed problems:\n\n'
    for error, count in errors.most_common():
        report += f'- {error}: {count}\n'
    report += ('\nThe original code runner rejected common Python patterns. The revised interpreter '
               'supports bounded assignments, conditionals, loops and dictionary operations. '
               'It also allows changing numeric state keys and requests repair after failed search. '
               'These execution changes do not establish that generated mathematics matches the question.\n\n')
    report += '| Case | Original | Saved final-code replay | Replay error |\n|---|---|---|---|\n'
    for case in audit:
        error = str(case['replay_error'] or '').replace('|', '\\|').replace('\n', ' ')
        report += f'| {case["source_id"]} | {case["original_outcome"]} | {case["replay_outcome"]} | {error} |\n'
    (output/'AUDIT.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Audit saved to', output)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--output', default='kaggle/results/tos_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    inspect_archive(args.archive, args.output)
