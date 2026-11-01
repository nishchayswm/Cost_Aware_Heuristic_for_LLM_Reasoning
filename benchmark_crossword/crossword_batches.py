"""Run and export disjoint crossword batches, and merge their named exports."""
import csv
import hashlib
import json
from pathlib import Path

import crossword_bench as bench


def batch_puzzles(puzzles, plan, batch_id):
    if bench.digest(puzzles) != plan['dataset_sha256']:
        raise ValueError('The frozen ten-puzzle selection changed')
    if type(batch_id) is not int or not 1 <= batch_id <= len(plan['batches']):
        raise ValueError('Unknown batch ID')
    indices = plan['batches'][batch_id - 1]['indices']
    return [puzzles[i] for i in indices]


def identity(plan, cfg, runtime, seed):
    if runtime['model'] != plan['model']['id'] or runtime['revision'] != plan['model']['revision']:
        raise ValueError('Model differs from the shared frozen plan')
    return dict(plan_sha256=bench.digest(plan), full_dataset_sha256=plan['dataset_sha256'],
                config=cfg, seed=seed, model=runtime['model'], revision=runtime['revision'],
                backend=runtime['backend'], packages=runtime['packages'],
                code_sha256=hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest())


def run_batch(puzzles, client, cfg, runtime, project, plan, batch_id, seed=42):
    selected = batch_puzzles(puzzles, plan, batch_id)
    common = identity(plan, cfg, runtime, seed)
    name = (f'crosswords10_batch{batch_id:02d}of05_seed{seed}_'
            f'tokens{cfg["max_tokens"]}_calls{cfg["max_calls"]}_exp{cfg["max_expansions"]}')
    output = Path(project) / 'results' / name
    metadata = dict(batch_id=batch_id, experiment=common,
                    experiment_sha256=bench.digest(common), plan=plan,
                    question_ids=[p['id'] for p in selected], runtime=runtime)
    path = output / 'batch_manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != metadata:
        raise ValueError('Batch changed; use a fresh project/run directory')
    bench.write_json(path, metadata)
    try:
        bench.run(selected, client, cfg, runtime, output, seed=seed)
    finally:
        export_batch(output)
    return output


def export_batch(output):
    """Export only saved records, including partial batches after interruption."""
    output = Path(output)
    metadata = json.loads((output / 'batch_manifest.json').read_text(encoding='utf-8'))
    rows = []
    for path in sorted((output / 'records').glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        rows.append(dict(row, batch_id=metadata['batch_id'],
                         experiment_sha256=metadata['experiment_sha256']))
    result = output / (output.name + '_results.jsonl')
    result.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    bench.write_json(output / (output.name + '_manifest.json'), metadata)
    summary = bench.report(output)
    if summary:
        with (output / (output.name + '_summary.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
    return result


def merge_batches(files, output):
    """Require all five batches, identical settings and all 20 unique method results."""
    rows, seen, manifests = [], set(), {}
    common = None
    for file in files:
        file = Path(file)
        prefix = file.name.removesuffix('_results.jsonl')
        metadata = json.loads(file.with_name(prefix + '_manifest.json').read_text(encoding='utf-8'))
        if common is None:
            common = metadata
        if metadata['experiment_sha256'] != bench.digest(metadata['experiment']):
            raise ValueError('Invalid experiment fingerprint')
        if metadata['experiment']['plan_sha256'] != bench.digest(metadata['plan']):
            raise ValueError('Invalid plan fingerprint')
        if metadata['experiment'] != common['experiment'] or metadata['plan'] != common['plan']:
            raise ValueError('Dataset, model, packages, config, seed or code differs between batches')
        batch_id = metadata['batch_id']
        if batch_id in manifests:
            raise ValueError('Duplicate batch: ' + str(batch_id))
        manifests[batch_id] = metadata
        plan = metadata['plan']
        if type(batch_id) is not int or not 1 <= batch_id <= len(plan['batches']):
            raise ValueError('Unknown batch ID')
        expected_ids = [plan['question_ids'][i] for i in plan['batches'][batch_id-1]['indices']]
        if metadata['question_ids'] != expected_ids:
            raise ValueError('Incorrect batch puzzle assignment')
        for line in file.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            row = json.loads(line)
            key = (row['question_id'], row['seed'], row['method'])
            if key in seen:raise ValueError('Duplicate result: ' + str(key))
            if (row['question_id'] not in expected_ids or row['method'] not in bench.METHODS
                    or row['seed'] != metadata['experiment']['seed'] or row['batch_id'] != batch_id
                    or row['experiment_sha256'] != metadata['experiment_sha256']):
                raise ValueError('Result does not match its batch manifest')
            seen.add(key); rows.append(row)
    if common is None or set(manifests) != set(range(1, 6)):
        raise ValueError('Need all five batch exports')
    plan = common['plan']
    expected = {(qid, common['experiment']['seed'], method)
                for qid in plan['question_ids'] for method in bench.METHODS}
    if seen != expected:raise ValueError('Incomplete run: expected both methods for all ten puzzles')
    rows.sort(key=lambda r: (plan['question_ids'].index(r['question_id']), bench.METHODS.index(r['method'])))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'crosswords10_merged_results.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    bench.write_json(output / 'manifest.json', dict(experiment=common['experiment'], plan=plan,
                                                  batches=list(manifests.values())))
    for row in rows:
        bench.write_json(output / 'records' / f'{row["question_id"]}_{row["method"]}.json', row)
    bench.report(output)
    return rows
