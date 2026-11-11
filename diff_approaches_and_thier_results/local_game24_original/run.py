"""Local/Ollama equivalent of Game24_Original_ToT_Kaggle.ipynb (no inference on import)."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import re
import sys
import urllib.request

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from costbench import game24

DEFAULT_DATA = PROJECT/'data/original_tot'
REPO = 'princeton-nlp/tree-of-thought-llm'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'costbench-game24-local'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()

def save_checked(path, content):
    """Never silently replace an existing experiment artifact."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Existing artifact differs: {path}. Use a fresh directory.')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(content)
    tmp.replace(path)

def json_bytes(value):
    return json.dumps(value, indent=2).encode('utf-8')

def select_split(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if len(rows)<1000 or 'Puzzles' not in rows[0]:
        raise ValueError('Original CSV must contain Puzzles and at least 1000 rows')
    puzzles=[]
    for index in range(900,1000):
        numbers=list(map(int, rows[index]['Puzzles'].split()))
        if len(numbers)!=4 or any(not 1<=n<=13 for n in numbers):
            raise ValueError(f'Invalid original puzzle at row {index}')
        puzzles.append(dict(id=f'tot24-{index:04d}',numbers=numbers))
    if len({tuple(sorted(p['numbers'])) for p in puzzles})!=100:
        raise ValueError('Duplicate puzzle multisets in the official split')
    return puzzles

def prepare(directory=DEFAULT_DATA, commit=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    manifest_path=directory/'source_manifest.json'
    if manifest_path.exists():
        meta=json.loads(manifest_path.read_text(encoding='utf-8'))
        if commit and meta['commit']!=commit:
            raise ValueError('Source revision changed; use a fresh dataset directory')
        raw=(directory/'24.csv').read_bytes()
        if sha(raw)!=meta['csv_sha256']:
            raise ValueError('Cached CSV hash mismatch')
        if meta.get('selected_zero_based_indices') != [900,1000]:
            raise ValueError('Cached manifest is not the official 900:1000 split')
    else:
        commit=commit or json.loads(fetch(f'https://api.github.com/repos/{REPO}/commits/master'))['sha']
        if not re.fullmatch('[0-9a-fA-F]{40}',commit):
            raise ValueError('--commit must be an exact 40-character Git revision')
        url=f'https://raw.githubusercontent.com/{REPO}/{commit}/src/tot/data/24/24.csv'
        raw=fetch(url)
        select_split(raw)  # Validate before persisting.
        license_bytes=fetch(f'https://raw.githubusercontent.com/{REPO}/{commit}/LICENSE')
        meta=dict(repository=REPO,commit=commit,url=url,csv_sha256=sha(raw),
                  selected_zero_based_indices=[900,1000],split='official ToT evaluation range')
        save_checked(directory/'24.csv',raw)
        save_checked(directory/'SOURCE_LICENSE.txt',license_bytes)
        save_checked(manifest_path,json_bytes(meta))
    puzzles=select_split(raw)
    dataset=directory/'tot_game24_900_999.jsonl'
    # Existing Kaggle exports may use Windows newlines. Compare records, preserve bytes.
    if dataset.exists():
        if [json.loads(s) for s in dataset.read_text(encoding='utf-8').splitlines() if s.strip()]!=puzzles:
            raise ValueError('Cached JSONL differs from the pinned source CSV')
    else:
        save_checked(dataset,(''.join(json.dumps(p)+'\n' for p in puzzles)).encode('utf-8'))
    print(f'Original ToT split: 100 puzzles, zero-based rows 900–999; commit {meta["commit"]}',flush=True)
    return dataset,puzzles,meta

def overlap_csv(puzzles):
    references={}
    for split in ('train','dev','test'):
        path=PROJECT/f'data/game24/game24_{split}.jsonl'
        if not path.exists():raise ValueError(f'Missing overlap reference: {path}')
        references[split]={tuple(sorted(json.loads(s)['numbers'])) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()}
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,['id','numbers','overlap_generated_train','overlap_generated_dev','overlap_generated_test'])
    writer.writeheader()
    for puzzle in puzzles:
        key=tuple(sorted(puzzle['numbers']))
        writer.writerow(dict(id=puzzle['id'],numbers=json.dumps(puzzle['numbers']),
                             **{f'overlap_generated_{s}':key in refs for s,refs in references.items()}))
    return stream.getvalue().encode('utf-8')

def report(output):
    root=Path(output)
    if not (root/'results.jsonl').exists():
        print('No results saved yet.');return
    game24.report(root)
    audit=root/'overlap_audit.csv'
    if audit.exists():
        with audit.open(encoding='utf-8',newline='') as f:
            clean={r['id'] for r in csv.DictReader(f) if all(r[f'overlap_generated_{s}']=='False' for s in ('train','dev','test'))}
        records=[json.loads(s) for s in (root/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        groups={}
        for r in records:groups.setdefault(r['question_id'],set()).add(r['method'])
        clean &= {qid for qid,methods in groups.items() if methods==set(game24.METHODS)}
        with (root/'disjoint_subset_summary.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,['method','n','accuracy','mean_tokens','mean_calls']);writer.writeheader()
            for method in game24.METHODS:
                rows=[r for r in records if r['question_id'] in clean and r['method']==method]
                writer.writerow(dict(method=method,n=len(rows),accuracy=sum(r['correct'] for r in rows)/len(rows) if rows else '',
                    mean_tokens=sum(r['total_tokens'] for r in rows)/len(rows) if rows else '',
                    mean_calls=sum(r['calls'] for r in rows)/len(rows) if rows else ''))
    print(f'Report: {root/"report.html"}')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare',help='Download original source only; no model calls')
    prep.add_argument('--directory',type=Path,default=DEFAULT_DATA)
    prep.add_argument('--commit',help='Optional exact source revision, e.g. from a Kaggle source_manifest.json')
    run=commands.add_parser('run',help='Evaluate all eight methods via local Ollama')
    run.add_argument('--directory',type=Path,default=DEFAULT_DATA)
    run.add_argument('--config',type=Path,default=PROJECT/'configs/game24_exp64.json')
    run.add_argument('--model',help='Override configured Ollama model name')
    run.add_argument('--base-url',help='Override configured Ollama URL')
    run.add_argument('--limit',type=int,default=100)
    run.add_argument('--seed',type=int,default=42)
    run.add_argument('--output',type=Path)
    run.add_argument('--retry-failed',action='store_true')
    rep=commands.add_parser('report',help='Regenerate reports without model inference')
    rep.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':prepare(args.directory,args.commit);return 0
        if args.command=='report':report(args.output);return 0
        if not 1<=args.limit<=100:raise ValueError('--limit must be between 1 and 100')
        dataset,puzzles,meta=prepare(args.directory)
        output=args.output or PROJECT/f'results/game24_original_n{args.limit}_seed{args.seed}_ollama'
        cfg=json.loads(args.config.read_text(encoding='utf-8'))
        if args.model:cfg['model']=args.model
        if args.base_url:cfg['base_url']=args.base_url
        cfg['seed']=args.seed
        cfg['local_runner_sha256']=sha(Path(__file__).read_bytes())
        cfg['backend_label']='ollama-local-original-tot'
        snapshot=output/'run_config.json'
        save_checked(snapshot,json_bytes(cfg))
        save_checked(output/'dataset_provenance.json',json_bytes(meta))
        save_checked(output/'overlap_audit.csv',overlap_csv(puzzles[:args.limit]))
        print(f'Model: {cfg["model"]}; methods: {len(game24.METHODS)}; puzzles: {args.limit}; seed: {args.seed}',flush=True)
        print(f'Saving to: {output}',flush=True)
        try:
            game24.run(argparse.Namespace(config=str(snapshot),dataset=str(dataset),limit=args.limit,
                       seed=args.seed,output=str(output),retry_failed=args.retry_failed))
        finally:
            if (output/'results.jsonl').exists():report(output)
        return 0
    except KeyboardInterrupt:
        print('Stopped. Completed calls/results are saved; rerun the same command to resume.');return 130
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
