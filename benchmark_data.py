"""Official GSM8K/SVAMP snapshots and deterministic, disjoint benchmark subsets.

Sampling depends only on IDs and a fixed selection seed, never model outcomes.
Cache hashes and selected records are exported so results can be reproduced.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request

SOURCES = {
    'gsm8k_train': {
        'url': 'https://raw.githubusercontent.com/openai/grade-school-math/master/'
               'grade_school_math/data/train.jsonl',
        'dataset': 'gsm8k', 'source_split': 'train', 'format': 'jsonl', 'count': 7473,
        'reference': 'https://arxiv.org/abs/2110.14168',
        'license': 'MIT (see upstream repository)',
    },
    'gsm8k_test': {
        'url': 'https://raw.githubusercontent.com/openai/grade-school-math/master/'
               'grade_school_math/data/test.jsonl',
        'dataset': 'gsm8k', 'source_split': 'test', 'format': 'jsonl', 'count': 1319,
        'reference': 'https://arxiv.org/abs/2110.14168',
        'license': 'MIT (see upstream repository)',
    },
    'svamp': {
        'url': 'https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json',
        'dataset': 'svamp', 'source_split': 'challenge', 'format': 'json', 'count': 1000,
        'reference': 'https://arxiv.org/abs/2103.07191',
        'license': 'MIT (see upstream repository)',
    },
}
SELECTION_SEED = 203
# Freeze the verified upstream bytes, including when Kaggle downloads afresh.
SOURCE_SHA256 = {
    'gsm8k_train': '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465',
    'gsm8k_test': '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14',
    'svamp': '5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e',
}


def parse_reference(value):
    """Reference-only parser, never applied to model output or sent to solvers."""
    if isinstance(value, bool):
        raise ValueError('boolean reference answer')
    text = str(value).strip()
    if '####' in text:
        text = text.rsplit('####', 1)[1].strip()
    text = text.replace(',', '').replace('$', '').strip()
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)', text):
        raise ValueError('unsupported reference answer: ' + text)
    answer = float(text)
    if not math.isfinite(answer):
        raise ValueError('nonfinite reference answer')
    return answer


def fingerprint(question):
    normalized = ' '.join(question.lower().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def normalize_records(source, records):
    spec = SOURCES[source]
    normalized = []
    identities = set()
    for index, row in enumerate(records):
        if spec['dataset'] == 'gsm8k':
            question = row['question'].strip()
            expected = parse_reference(row['answer'])
            original_id = str(index)
        else:
            question = (row['Body'].strip() + ' ' + row['Question'].strip()).strip()
            expected = parse_reference(row['Answer'])
            original_id = str(row['ID'])
        source_id = f'{spec["dataset"]}:{spec["source_split"]}:{original_id}'
        if not question or source_id in identities:
            raise ValueError('empty question or duplicate source ID')
        identities.add(source_id)
        normalized.append(dict(question=question, expected=expected,
                               dataset=spec['dataset'], source_split=spec['source_split'],
                               source_id=source_id, question_sha256=fingerprint(question)))
    return normalized


def read_snapshot(source, cache_dir, download=True):
    spec = SOURCES[source]
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (source + '.' + spec['format'])
    metadata_path = cache / (source + '.metadata.json')
    if not path.exists():
        if not download:
            raise FileNotFoundError('Missing dataset snapshot: ' + str(path))
        if metadata_path.exists():
            raise ValueError('Snapshot missing but provenance exists; restore it or use a new cache directory')
        request = urllib.request.Request(spec['url'], headers={'User-Agent': 'CostAwareBenchmark/3.1'})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(20_000_001)
        if len(raw) > 20_000_000:
            raise ValueError('unexpectedly large dataset download')
        if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256[source]:
            raise ValueError('Upstream dataset changed; review it before changing the frozen source hash')
        records = (json.loads(raw) if spec['format'] == 'json' else
                   [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()])
        # Validate before caching a potentially truncated or wrong download.
        normalize_records(source, records)
        if len(records) != spec['count']:
            raise ValueError('Unexpected upstream dataset size for ' + source)
        metadata = dict(spec, sha256=hashlib.sha256(raw).hexdigest(), rows=len(records))
        with path.open('xb') as handle:
            handle.write(raw)
        with metadata_path.open('x', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)
    raw = path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    actual_hash = hashlib.sha256(raw).hexdigest()
    if (metadata['sha256'] != actual_hash or actual_hash != SOURCE_SHA256[source]
            or metadata['url'] != spec['url']):
        raise ValueError('Dataset snapshot provenance/hash mismatch: ' + str(path))
    records = (json.loads(raw) if spec['format'] == 'json' else
               [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()])
    if len(records) != spec['count']:
        raise ValueError('Unexpected cached dataset size')
    return normalize_records(source, records), metadata


def fixed_sample(records, count, seed, excluded=()):
    if count < 0:
        raise ValueError('sample count must be nonnegative')
    used = set(excluded)
    ordered = sorted(records, key=lambda row: hashlib.sha256(
        f'{seed}:{row["source_id"]}'.encode('utf-8')).hexdigest())
    result = []
    for row in ordered:
        if row['question_sha256'] not in used and len(result) < count:
            used.add(row['question_sha256'])
            result.append(dict(row))
    if len(result) != count:
        raise ValueError(f'Requested {count} distinct problems, only {len(result)} available')
    return result


def build_subsets(catalog, selection_seed=SELECTION_SEED, dev_count=30, gsm_count=100, svamp_count=50):
    # Exclude all GSM8K test questions from development, not just the sampled test subset.
    test_fingerprints = {row['question_sha256'] for row in catalog['gsm8k_test']}
    development = fixed_sample(catalog['gsm8k_train'], dev_count, selection_seed, test_fingerprints)
    used = {row['question_sha256'] for row in development}
    gsm = fixed_sample(catalog['gsm8k_test'], gsm_count, selection_seed, used)
    used.update(row['question_sha256'] for row in gsm)
    svamp = fixed_sample(catalog['svamp'], svamp_count, selection_seed, used)
    for row in development:
        row['evaluation_split'] = 'development'
    for row in gsm + svamp:
        row['evaluation_split'] = 'evaluation'
    return {'development': development, 'evaluation': gsm + svamp,
            'pilot': development[:min(20, len(development))]}


def prepare_benchmarks(cache_dir='kaggle/data/official', selection_seed=SELECTION_SEED, download=True):
    catalog, provenance = {}, {}
    for source in SOURCES:
        catalog[source], provenance[source] = read_snapshot(source, cache_dir, download)
    subsets = build_subsets(catalog, selection_seed)
    return subsets, {'selection_seed': selection_seed, 'sources': provenance,
                     'sampling': 'SHA256 rank of selection seed and source ID; exact question deduplication',
                     'development': '30 GSM8K train; excludes all exact GSM8K test matches',
                     'evaluation': '100 GSM8K test + 50 SVAMP challenge; no development overlap',
                     'pilot': 'first 20 selected development questions; not final evaluation'}


def export_subsets(output, subsets, provenance):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, records in subsets.items():
        with (output / (name + '.jsonl')).open('x', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    (output / 'dataset_manifest.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='kaggle/data/official')
    parser.add_argument('--output', default='kaggle/data/subsets_seed203')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    subsets, provenance = prepare_benchmarks(args.cache, download=not args.offline)
    export_subsets(args.output, subsets, provenance)
    print('Prepared:', {name: len(rows) for name, rows in subsets.items()})
