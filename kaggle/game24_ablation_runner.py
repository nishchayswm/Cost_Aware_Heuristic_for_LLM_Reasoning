"""Run Game of 24 with the original heuristic and five leave-one-out variants."""
from __future__ import annotations
import argparse, copy, csv, hashlib, json
from pathlib import Path
from costbench.game24 import GameEngine, METHODS as GAME_METHODS
from costbench.backend import Session, read_jsonl
from costbench.__main__ import config as load_config, doctor

FEATURES = ('quality', 'success', 'depth', 'cost', 'diversity')
VARIANTS = ('heuristic_bestfirst',) + tuple('without_' + x for x in FEATURES)

def variant_config(base, excluded):
    cfg = copy.deepcopy(base)
    if excluded:
        remaining = 1.0 - cfg['weights'][excluded]
        cfg['weights'] = {k: (0.0 if k == excluded else v / remaining) for k, v in cfg['weights'].items()}
    return cfg

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True); p.add_argument('--limit', type=int, default=100)
    p.add_argument('--seed', type=int); p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--method', choices=VARIANTS, default=None)
    a = p.parse_args(); base = load_config(a.config)
    if a.seed is not None: base['seed'] = a.seed
    puzzles = read_jsonl(a.dataset)[:a.limit]
    if len(puzzles) != a.limit: raise ValueError('Dataset has fewer than LIMIT puzzles')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / 'calls').mkdir(exist_ok=True)
    methods_to_run = [a.method] if a.method else list(VARIANTS)
    # Include the run identity in every download, even after ZIP extraction.
    prefix = out.name + '_' if a.method else ''
    results_path = out / (prefix + 'results.jsonl')
    summary_path = out / (prefix + 'summary.csv')
    manifest_path = out / (prefix + 'manifest.json')
    identity = {'task': 'game24', 'seed': base['seed'], 'model': base['model'],
                'dataset_sha256': hashlib.sha256(Path(a.dataset).read_bytes()).hexdigest(),
                'selection_sha256': hashlib.sha256(json.dumps(puzzles, sort_keys=True).encode()).hexdigest(),
                'base_config_sha256': hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()}
    manifest = {**identity, 'limit': a.limit, 'methods': methods_to_run,
                'question_ids': [p['id'] for p in puzzles], 'config': base}
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Experiment settings changed; choose a new output directory')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    saved = {(r['question_id'], r['method']) for r in read_jsonl(results_path)} if results_path.exists() else set()
    for i, puzzle in enumerate(puzzles, 1):
        print(f'PUZZLE {i}/{len(puzzles)} {puzzle["id"]}: {puzzle["numbers"]}', flush=True)
        for method in methods_to_run:
            if (puzzle['id'], method) in saved: continue
            excluded = None if method == VARIANTS[0] else method.removeprefix('without_')
            cfg = variant_config(base, excluded)
            session = Session(cfg, puzzle['id'], method, cfg['seed'], out / 'calls' / f'{puzzle["id"]}_{method}.jsonl', retry_failed=a.retry_failed)
            result = GameEngine(session, puzzle['numbers']).run('heuristic_bestfirst')
            row = {**identity, 'question_id': puzzle['id'], 'numbers': puzzle['numbers'], 'method': method, 'excluded_from_priority': excluded, **result, **session.metrics()}
            row.pop('nodes', None); row.pop('events', None)
            with results_path.open('a', encoding='utf-8') as f: f.write(json.dumps(row) + '\n')
            saved.add((puzzle['id'], method))
            print(f'  {method}: solved={row["correct"]} calls={row["calls"]} tokens={row["total_tokens"]}', flush=True)
    rows = read_jsonl(results_path)
    summary = []
    for method in methods_to_run:
        group = [r for r in rows if r['method'] == method]
        summary.append({'method': method, 'n': len(group), 'correct': sum(bool(r['correct']) for r in group),
                        'accuracy': sum(bool(r['correct']) for r in group) / len(group) if group else None,
                        'mean_calls': sum(r.get('calls', 0) for r in group) / len(group) if group else None,
                        'mean_total_tokens': sum(r.get('total_tokens', 0) for r in group) / len(group) if group else None})
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print('Completed:', results_path, '| summary:', summary_path)

if __name__ == '__main__': main()
