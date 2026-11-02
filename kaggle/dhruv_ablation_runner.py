"""Six paired GSM8K runs: V6 priority and five ranking-feature ablations.

All variants use the same generation, judge, budgets, starting candidate and
final selector. A named ablation removes that feature wherever it appears in
the *priority score*. Q/P remain available to the common final selector.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from costbench.__main__ import config as load_config, doctor, fingerprint
from costbench.backend import BackendError, Session, append_json, read_jsonl, request
from costbench.graph import number
from costbench.labels import load_corrections, score
from costbench.methods import Engine


FEATURES = ('quality', 'success', 'depth', 'cost', 'diversity')
VARIANTS = ('heuristic_bestfirst',) + tuple('without_' + name for name in FEATURES)
LABELS = {'heuristic_bestfirst': 'Main: five-factor priority',
          **{'without_' + name: 'Without ' + name for name in FEATURES}}


def variant_config(base, excluded):
    cfg = copy.deepcopy(base)
    if excluded is not None:
        if excluded not in FEATURES:
            raise ValueError('Unknown ablation: ' + str(excluded))
        weights = cfg['weights']
        remaining = 1 - weights[excluded]
        if remaining <= 0:
            raise ValueError('Cannot ablate the only positive weight')
        cfg['weights'] = {key: (0.0 if key == excluded else value / remaining)
                          for key, value in weights.items()}
    return cfg


class RankingAblationEngine(Engine):
    def __init__(self, session, question, excluded=None):
        super().__init__(session, question)
        if excluded is not None and excluded not in FEATURES:
            raise ValueError('Unknown ablation: ' + str(excluded))
        self.excluded = excluded

    def features(self, nodes):
        nodes = super().features(nodes)
        if self.excluded is None:
            return nodes
        weights = self.cfg['weights']
        for node in nodes:
            values = node.features
            # Remove Q/P from interactions as well as their direct weighted term.
            if self.excluded == 'quality':
                values['diversity'] = values['raw_diversity'] * math.sqrt(node.success)
            elif self.excluded == 'success':
                values['depth'] = min(1.0, node.depth / self.cfg['max_depth'])
                values['diversity'] = values['raw_diversity'] * math.sqrt(node.quality)
            values[self.excluded] = 0.0
            values['heuristic'] = sum(weights[key] * values[key] *
                                      (-1 if key in ('depth', 'cost') else 1)
                                      for key in FEATURES)
            values['excluded_from_priority'] = self.excluded
        return nodes


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def report(root):
    root = Path(root)
    rows = read_jsonl(root / 'results.jsonl')
    by_key = {(row['question_id'], row['seed'], row['method']): row for row in rows}
    cases = sorted({(row['question_id'], row['seed']) for row in rows})
    complete = [case for case in cases
                if all((*case, method) in by_key for method in VARIANTS)]
    summary = []
    for method in VARIANTS:
        group = [by_key[(*case, method)] for case in complete]
        n = len(group)
        mean = lambda field: round(statistics.mean(row[field] for row in group), 2) if n else None
        summary.append({'method': method, 'label': LABELS[method], 'questions': n,
                        'correct': sum(row['correct'] for row in group),
                        'accuracy': sum(row['correct'] for row in group) / n if n else None,
                        'mean_calls': mean('calls'), 'mean_input_tokens': mean('input_tokens'),
                        'mean_output_tokens': mean('output_tokens'),
                        'mean_total_tokens': mean('total_tokens'),
                        'mean_seconds': mean('inference_seconds'),
                        'unknown_cost_results': sum(bool(row['unknown_cost']) for row in group)})
    write_csv(root / 'summary.csv', summary, list(summary[0]))
    comparisons = []
    for method in VARIANTS[1:]:
        accuracy = [int(by_key[(*case, 'heuristic_bestfirst')]['correct']) -
                    int(by_key[(*case, method)]['correct']) for case in complete]
        tokens = [by_key[(*case, 'heuristic_bestfirst')]['total_tokens'] -
                  by_key[(*case, method)]['total_tokens'] for case in complete]
        if accuracy:
            rng = random.Random(73)
            samples = sorted(statistics.mean(rng.choices(accuracy, k=len(accuracy)))
                             for _ in range(2000))
            low, high = samples[49], samples[1949]
        else:
            low = high = None
        comparisons.append({'ablation': method, 'paired_questions': len(complete),
                            'main_minus_ablation_accuracy': statistics.mean(accuracy) if accuracy else None,
                            'bootstrap_low': low, 'bootstrap_high': high,
                            'main_minus_ablation_mean_tokens': statistics.mean(tokens) if tokens else None})
    write_csv(root / 'paired_comparisons.csv', comparisons, list(comparisons[0]))
    wide = []
    for qid, seed in cases:
        found = next(row for row in rows if row['question_id'] == qid and row['seed'] == seed)
        item = {'question_id': qid, 'seed': seed, 'question': found['question'],
                'gold': found['gold'], 'gold_reviewed': found['gold_reviewed']}
        for method in VARIANTS:
            row = by_key.get((qid, seed, method), {})
            for field in ('prediction', 'correct', 'correct_reviewed', 'calls',
                          'total_tokens', 'stop_reason', 'unknown_cost'):
                item[method + '__' + field] = row.get(field, '')
        wide.append(item)
    write_csv(root / 'per_question.csv', wide,
              list(wide[0]) if wide else ['question_id', 'seed', 'question', 'gold'])
    if rows:
        write_csv(root / 'results_long.csv', rows, list(dict.fromkeys(
            key for row in rows for key in row)))
    lines = ['# Five-factor ranking ablations', '',
             f'{len(complete)} complete six-variant question blocks; {len(rows)} saved method results.', '',
             '| Variant | Correct / N | Accuracy | Mean calls | Mean tokens | Unknown-cost rows |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summary:
        accuracy_text = f'{row["accuracy"]:.1%}' if row['accuracy'] is not None else 'pending'
        lines.append(f'| {row["label"]} | {row["correct"]}/{row["questions"]} | '
                     f'{accuracy_text} | {row["mean_calls"]} | {row["mean_total_tokens"]} | '
                     f'{row["unknown_cost_results"]} |')
    lines += ['', 'Only complete paired blocks enter the summary and comparisons.',
              'Each ablation removes one feature from priority ranking and renormalizes the other weights.',
              'Removing Q or P also removes its depth/diversity interaction in priority.',
              'The judge still computes Q/P and the common final answer selector still uses both.',
              'All variants use the same seed stream for identical prompts; later branches can diverge.',
              'Failed HTTP calls have unknown token cost; resumed rows flag it and totals are lower bounds.',
              'A small development run is diagnostic, not evidence of a general improvement.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def run(args):
    base = load_config(args.config)
    if args.model:
        base['model'] = args.model
    if args.seed is not None:
        base['seed'] = args.seed
    if args.limit < 1:
        raise ValueError('limit must be positive')
    dataset = Path(args.dataset)
    questions = read_jsonl(dataset)[:args.limit]
    if not questions or len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Empty dataset or duplicate IDs')
    if any(number(q['answer']) is None for q in questions):
        raise ValueError('Dataset has a nonnumeric answer')
    corrections = load_corrections(args.label_corrections)
    for question in questions:
        score(question, None, corrections)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = doctor(base)
    config_by_method = {method: variant_config(base, None if method == VARIANTS[0]
                                                else method.removeprefix('without_'))
                        for method in VARIANTS}
    manifest = {'source_code_sha256': fingerprint(),
                'ablation_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                'corrections_sha256': hashlib.sha256(Path(args.label_corrections).read_bytes()).hexdigest(),
                'question_ids': [q['id'] for q in questions], 'seed': base['seed'],
                'methods': list(VARIANTS), 'configs': config_by_method,
                'model_digest': env['model']['digest']}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))['experiment'] != manifest:
            raise ValueError('Code/config/data/model changed; use a new output directory')
    else:
        manifest_path.write_text(json.dumps({'experiment': manifest, 'environment': env,
                                             'created_utc': datetime.now(timezone.utc).isoformat()},
                                            indent=2), encoding='utf-8')
    lock = output / 'run.lock'
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Run lock exists; confirm no runner is active before resuming')
    os.close(handle)
    try:
        report(output)
        existing = {(row['question_id'], row['seed'], row['method'])
                    for row in read_jsonl(output / 'results.jsonl')}
        if all((q['id'], base['seed'], method) in existing
               for q in questions for method in VARIANTS):
            print('Already complete; report refreshed.')
            return
        warm = request(base['base_url'], '/api/chat',
                       {'model': base['model'], 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
                        'stream': False, 'keep_alive': '30m',
                        'options': {'num_ctx': base['num_ctx'], 'num_predict': 8}},
                       base['timeout_seconds'])
        append_json(output / 'warmups.jsonl', {'response': warm})
        for index, question in enumerate(questions):
            order = list(VARIANTS)
            shift = index % len(order)
            order = order[shift:] + order[:shift]
            print(f'QUESTION {index + 1}/{len(questions)}: {question["id"]}', flush=True)
            for method in order:
                if (output / 'pause.request').exists():
                    (output / 'pause.request').unlink()
                    report(output)
                    print('Paused between methods; rerun unchanged command to resume.')
                    return
                if (question['id'], base['seed'], method) in existing:
                    print(method, 'already saved', flush=True)
                    continue
                cfg = config_by_method[method]
                excluded = None if method == VARIANTS[0] else method.removeprefix('without_')
                log = output / 'calls' / f'{question["id"]}_{base["seed"]}_{method}.jsonl'
                # The policy seed identity is shared across variants; log files stay distinct.
                session = Session(cfg, question['id'], 'heuristic_bestfirst', base['seed'],
                                  log, retry_failed=args.retry_failed)
                started = time.perf_counter()
                try:
                    result = RankingAblationEngine(session, question['question'], excluded).run(
                        'heuristic_bestfirst')
                except BackendError:
                    report(output)
                    raise
                prediction = number(result['prediction']) if result['prediction'] is not None else None
                row = {'question_id': question['id'], 'question': question['question'],
                       'seed': base['seed'], 'method': method, 'label': LABELS[method],
                       'excluded_from_priority': excluded,
                       'status': 'ok', **result, 'prediction': prediction, **session.metrics(),
                       'execution_seconds_this_attempt': round(time.perf_counter() - started, 4),
                       **score(question, prediction, corrections)}
                trace = output / 'traces' / f'{question["id"]}_{base["seed"]}_{method}.json'
                trace.parent.mkdir(parents=True, exist_ok=True)
                trace.write_text(json.dumps(row, indent=2), encoding='utf-8')
                row.pop('candidates', None)
                row.pop('events', None)
                row['trace'] = str(trace.relative_to(output))
                append_json(output / 'results.jsonl', row)
                existing.add((question['id'], base['seed'], method))
                print(f'  {method:25s} answer={prediction} correct={row["correct"]} '
                      f'calls={row["calls"]} tokens={row["total_tokens"]}', flush=True)
                report(output)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--label-corrections', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--model')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print('Interrupted; saved methods and successful calls remain on disk.', file=sys.stderr)
        return 130
    except Exception as exc:
        print('ERROR:', exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
