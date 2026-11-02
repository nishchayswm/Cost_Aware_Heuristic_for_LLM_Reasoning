"""Freeze one dataset/model revision and package five independent Kaggle batches."""
import base64
import copy
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import crossword_bench as bench

ROOT = Path(__file__).resolve().parent


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Crossword-batch-builder'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def build():
    commit = json.loads(fetch('https://api.github.com/repos/princeton-nlp/tree-of-thought-llm/commits/master'))['sha']
    source_url = f'https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/{commit}/src/tot/data/crosswords/mini0505.json'
    raw = fetch(source_url)
    license_bytes = fetch(f'https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/{commit}/LICENSE')
    records = json.loads(raw)
    unique = {}
    for index, (clues, letters) in enumerate(records):
        board = ''.join(letters).upper()
        assert len(clues) == 10 and len(board) == 25 and board.isascii() and board.isalpha()
        unique.setdefault(board, dict(id=f'crossword-{index:04d}', source_index=index,
                                     clues=clues, solution=board))
    pool = random.Random(2026).sample(list(unique.values()), 50)
    puzzles = pool[:10]
    expected = [30, 81, 128, 131, 26, 57, 142, 107, 146, 140]
    if [p['source_index'] for p in puzzles] != expected:
        raise ValueError('Upstream data no longer yields the established ten-puzzle selection')
    model = 'Qwen/Qwen2.5-7B-Instruct'
    revision = json.loads(fetch('https://huggingface.co/api/models/' + model))['sha']
    plan = dict(version='crosswords10-parallel-v1', dataset_sha256=bench.digest(puzzles),
                question_ids=[p['id'] for p in puzzles], selection_seed=2026,
                model=dict(id=model, revision=revision),
                batches=[dict(batch_id=i+1, indices=[2*i, 2*i+1]) for i in range(5)])
    provenance = dict(repository='princeton-nlp/tree-of-thought-llm', commit=commit,
                      url=source_url, source_sha256=hashlib.sha256(raw).hexdigest(),
                      selection_seed=2026, source_count=len(records),
                      pool_ids=[p['id'] for p in pool], selected_ids=plan['question_ids'])
    files = {name: (ROOT/name).read_bytes() for name in
             ('crossword_bench.py', 'test_crossword_bench.py', 'test_crossword_batches.py',
              'crossword_batches.py', 'merge_crossword_batches.py', 'README.md')}
    files.update({'data/puzzles10.json': json.dumps(puzzles, indent=2).encode(),
                  'data/batch_plan.json': json.dumps(plan, indent=2).encode(),
                  'data/source_manifest.json': json.dumps(provenance, indent=2).encode(),
                  'data/SOURCE_LICENSE.txt': license_bytes})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            # Deterministic archive metadata for identical payloads across the five notebooks.
            info=zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode()
    payload_hash = hashlib.sha256(payload).hexdigest()
    template = json.loads((ROOT/'Crosswords10_Heuristic_vs_ToT_Kaggle.ipynb').read_text(encoding='utf-8'))
    target = ROOT/'parallel'
    target.mkdir(exist_ok=True)
    for batch_id in range(1, 6):
        notebook = copy.deepcopy(template)
        for cell in notebook['cells']:
            src = cell['source']
            if isinstance(src, list):src=''.join(src)
            if cell['cell_type'] == 'markdown':
                if src.startswith('# 10 Mini Crosswords:'):
                    src=(f'# Crossword batch {batch_id}/5 — puzzles {2*batch_id-1}–{2*batch_id}\n\n'
                         'Upload one DIFFERENT batch notebook to each of five Kaggle GPU sessions. '
                         'Run cells in order; restart after installation if needed. Both methods run '
                         'on each assigned puzzle. The ten-puzzle dataset, model revision and source '
                         'are frozen identically in all five notebooks.\n\n'
                         '**Limits per method/puzzle: 30,000 input+output tokens, 60 calls, '
                         '30 expansions.** The full experiment has 20 method runs; this notebook '
                         'has four. Download its uniquely named ZIP, or its named result and manifest '
                         'files, before ending the session. Partial results can also be exported.\n')
                elif src.startswith('## 1.'):
                    src='## 1. Offline tests and frozen batch assignment\nNo dataset download or resampling occurs here.\n'
                elif src.startswith('## 3.'):
                    src='## 3. Shared settings\nBoth methods use 30k tokens, 60 calls, 30 expansions, depth 20, branching 3 and frontier 60. All retries and judging calls count.\n'
                elif src.startswith('## 4.'):
                    src=f'## 4. Run batch {batch_id}/5\nRuns both methods on the assigned two puzzles. Completed records resume with identical settings.\n'
                elif src.startswith('## 6.'):
                    src='## 6. Download results or a partial checkpoint\nThis cell works after a kernel restart: it discovers saved runs for this batch without needing run-cell variables. The ZIP includes source, frozen inputs, manifests, raw calls and saved records.\n'
            elif 'subprocess.check_call' in src and "'pip','install'" in src:
                src="""import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers==4.57.6', 'tokenizers==0.22.2', 'accelerate==1.14.0',
    'bitsandbytes==0.50.2', 'huggingface_hub==0.36.2',
    'lm-format-enforcer==0.11.3', 'pandas', 'matplotlib'])
# Restart the kernel after installation, then continue with the next cell.
"""
            elif 'payload=base64.b64decode' in src:
                src=f"""from pathlib import Path
import base64, hashlib, io, json, os, sys, zipfile
BATCH_ID={batch_id}
PROJECT=Path('/kaggle/working/crosswords10_parallel_v1_batch{batch_id:02d}of05')
PROJECT.mkdir(parents=True, exist_ok=True)
payload=base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest()=={payload_hash!r}
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        target=(PROJECT/name).resolve()
        if PROJECT.resolve() not in target.parents:raise ValueError('Unsafe archive path')
        content=archive.read(name)
        if target.exists() and target.read_bytes()!=content:
            raise RuntimeError('Existing source differs; use a fresh project directory.')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():target.write_bytes(content)
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:sys.path.insert(0, str(PROJECT))
print('Project:', PROJECT)
"""
            elif 'PUZZLE_LIMIT=10' in src:
                src="""subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', '.', '-v'], check=True)
for module in ('crossword_batches', 'crossword_bench'):sys.modules.pop(module, None)
import crossword_bench as bench
import crossword_batches as batches
plan=json.loads((PROJECT/'data/batch_plan.json').read_text())
all_puzzles=json.loads((PROJECT/'data/puzzles10.json').read_text())
puzzles=batches.batch_puzzles(all_puzzles, plan, BATCH_ID)
print('Assigned batch:', BATCH_ID, '/ 5')
print('Puzzle IDs:', [p['id'] for p in puzzles])
"""
            elif "MODEL_ID='Qwen/Qwen2.5-7B-Instruct'" in src:
                begin=src.index("MODEL_ID='Qwen/Qwen2.5-7B-Instruct'")
                end=src.index('tokenizer=AutoTokenizer.from_pretrained')
                src=(src[:begin]+"MODEL_ID=plan['model']['id']\npin=dict(model=MODEL_ID, revision=plan['model']['revision'])\nbench.write_json(PROJECT/'model_pin.json', pin)\n"+src[end:])
            elif 'config=copy.deepcopy(bench.DEFAULT)' in src:
                src=src.replace('config=copy.deepcopy(bench.DEFAULT)',
                                "config=copy.deepcopy(bench.DEFAULT)\nconfig.update(max_tokens=30_000, max_calls=60, max_expansions=30)")
            elif 'RUN_BENCHMARK=False' in src:
                src="""RUN_BENCHMARK=True
SEED=42
if RUN_BENCHMARK:
    assert globals().get('PREFLIGHT_PASSED', False), 'Pass the preflight cell first.'
    output=batches.run_batch(all_puzzles, client, config, runtime, PROJECT, plan, BATCH_ID, seed=SEED)
    print('Saved results:', output)
else:
    print('Set RUN_BENCHMARK=True when ready.')
"""
            elif "export=Path('/kaggle/working')" in src:
                src=f"""from pathlib import Path
from IPython.display import FileLink, display
import json, os, shutil, sys, zipfile
PROJECT=Path('/kaggle/working/crosswords10_parallel_v1_batch{batch_id:02d}of05')
if str(PROJECT) not in sys.path:sys.path.insert(0, str(PROJECT))
import crossword_batches as batches
saved=sorted((PROJECT/'results').glob('*/batch_manifest.json'))
if not saved:print('No saved benchmark runs yet. Run setup, preflight and the batch cell first.')
previous_dir=Path.cwd()
try:
    os.chdir('/kaggle/working')
    for manifest in saved:
        output=manifest.parent
        batches.export_batch(output)
        export=Path('/kaggle/working')/(output.name+'.zip')
        with zipfile.ZipFile(export, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in PROJECT.rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.name!='run.lock':
                    archive.write(path, PROJECT.name/path.relative_to(PROJECT))
        display(FileLink(export.name))
        for suffix in ('results.jsonl', 'manifest.json', 'summary.csv'):
            source=output/(output.name+'_'+suffix)
            if source.exists():
                destination=Path('/kaggle/working')/source.name
                shutil.copy2(source, destination)
                display(FileLink(destination.name))
finally:
    os.chdir(previous_dir)
"""
            cell['source']=src
        for i, cell in enumerate(notebook['cells']):
            cell['id']=f'crosswords-batch{batch_id:02d}-{i:02d}'
            if cell['cell_type']=='code':
                cell['outputs']=[]; cell['execution_count']=None
                compile(cell['source'], cell['id'], 'exec')
        notebook['metadata'].update(batch_id=batch_id, batch_count=5,
                                    payload_sha256=payload_hash, plan=plan)
        path=target/f'Crosswords10_Batch{batch_id:02d}_of_05.ipynb'
        path.write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8', newline='\n')
        print('Built', path)
    print('Frozen dataset:', plan['dataset_sha256'], '| model revision:', revision)


if __name__ == '__main__':build()
