"""Run seeded experiments and save every result immediately to a new directory."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark_engine import Config, METHODS, evaluate, numeric
from benchmark_runtime import ollama_backend
from problems import STANDARD_PROBLEMS, HARD_PROBLEMS
from benchmark_data import prepare_benchmarks


def run_suite(backend, output, seeds=(11, 23, 37), problems=None, methods=None,
              dataset_provenance=None, profile='custom'):
    methods = list(METHODS if methods is None else methods)
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError('Select distinct known benchmark methods')
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Select distinct seeds')
    problems = problems if problems is not None else STANDARD_PROBLEMS + HARD_PROBLEMS
    cases = []
    for index, problem in enumerate(problems):
        case = dict(problem) if isinstance(problem, dict) else {
            'question': problem[0], 'expected': problem[1], 'dataset': 'custom',
            'source_id': f'custom:{index}', 'source_split': 'custom', 'evaluation_split': profile}
        if not isinstance(case.get('question'), str) or not case['question'].strip() or not numeric(case['expected']):
            raise ValueError('Each problem needs a question and finite numeric expected answer')
        case.setdefault('dataset', 'custom')
        case.setdefault('source_id', f'custom:{index}')
        case.setdefault('evaluation_split', profile)
        cases.append(case)
    if not cases or len({(case['dataset'], case['source_id']) for case in cases}) != len(cases):
        raise ValueError('Need nonempty problems with distinct dataset/source IDs')
    if len({case['evaluation_split'] for case in cases}) != 1:
        raise ValueError('Do not mix development and evaluation problems in one experiment')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'engine_version': '3', 'seeds': seeds, 'methods': methods,
                'model': 'qwen2.5:7b', 'config': vars(Config()),
                'baseline_versions': {'thought_of_search': 'tos-arithmetic-2'},
                'problems': cases, 'profile': profile, 'dataset_provenance': dataset_provenance,
                'planned_solver_runs': len(cases)*len(methods)*len(seeds),
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'policy': 'No cache; shared step generator, terminal review and budgets. '
                          'Greedy uses one candidate and beam=1. ToT alone judges every step. '
                          'Standard CoT uses one complete 4096-token request without review. '
                          'Thought of Search synthesizes a per-problem restricted Python model, '
                          'then local BFS with no LLM calls or terminal review during search. '
                          'All synthesis/repair calls count; no domain-level amortization. '
                          'ToS v2 uses bounded Python statements and repairs unsuccessful searches.',
                'references': {'thought_of_search': 'https://arxiv.org/abs/2404.11833'}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with (output / 'selected_problems.jsonl').open('x', encoding='utf-8') as selected:
        for case in cases:
            selected.write(json.dumps(case, ensure_ascii=False) + '\n')
    rows = []
    with (output / 'full_results.jsonl').open('w', encoding='utf-8') as handle:
        for seed in seeds:
            for pid, case in enumerate(cases):
                for method in methods:
                    result = evaluate(backend, case['question'], case['expected'], method, Config(seed=seed), pid)
                    result['experiment_id'] = output.name
                    result['profile'] = profile
                    for key in ('dataset', 'source_id', 'source_split', 'evaluation_split', 'question_sha256'):
                        if key in case:
                            result[key] = case[key]
                    handle.write(json.dumps(result) + '\n')
                    handle.flush()
                    rows.append(result)
                    detail = ''
                    if result['outcome'] == 'missing':
                        detail = f' [{result["status"]}: {str(result.get("last_error") or "no goal reached")[:160]}]'
                    elif result['outcome'] == 'wrong':
                        detail = f' [answer={result["ans"]}, expected={result["expected"]}]'
                    print(f'Seed {seed}, {case["source_id"]}, {method}: '
                          f'{result["outcome"]} ({result["calls"]} calls){detail}', flush=True)
    from plot_results import generate_report_and_plots
    generate_report_and_plots(str(output), input_files=[str(output / 'full_results.jsonl')])
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='kaggle/results/run_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--profile', choices=['pilot', 'development', 'evaluation', 'regression'], default='pilot')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--limit', type=int, help='Optional cap for a diagnostic run')
    parser.add_argument('--data-cache', default='kaggle/data/official')
    parser.add_argument('--offline', action='store_true', help='Require existing dataset snapshots')
    parser.add_argument('--tos-only', action='store_true', help='Run just the added ToS adaptation')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.profile == 'regression':
        selected, provenance = STANDARD_PROBLEMS + HARD_PROBLEMS, None
    else:
        subsets, provenance = prepare_benchmarks(args.data_cache, download=not args.offline)
        selected = subsets[args.profile]
    seeds = args.seeds or ([11, 23, 37] if args.profile == 'evaluation' else [11])
    run_suite(ollama_backend, args.output, seeds, selected[:args.limit],
              methods=[METHODS[5]] if args.tos_only else None,
              dataset_provenance=provenance, profile=args.profile)
