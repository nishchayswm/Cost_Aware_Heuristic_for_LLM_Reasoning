from pathlib import Path
import base64,hashlib,io,json,zipfile,textwrap
root=Path(__file__).resolve().parent.parent
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for pattern in ('costbench/*.py','configs/*.json','tests/test*.py','data/game24/*.json*'):
        for p in root.glob(pattern):z.write(p,p.relative_to(root).as_posix())
    for rel in ('kaggle/transformers_backend.py','data/label_corrections.json','GAME24.md'):
        p=root/rel
        if p.exists():z.write(p,rel)
encoded=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(s).strip()+'\n'))
def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=textwrap.dedent(s).strip()+'\n'))
md('''# Game of 24 — original ToT split on Kaggle
Self-contained project snapshot. Choose a GPU accelerator and enable Internet in Kaggle notebook settings. No paid API or Ollama service is needed. Import this notebook, run cells in order, then enable the benchmark in the final run cell.

**Dataset:** the official ToT repository's `24.csv`, zero-based rows **900–999**, following its [experiment script](https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/scripts/game24/bfs.sh). This preserves the official split rather than drawing a new random sample.

**Eight methods:** proposed heuristic, quality-only best-first, bounded breadth search, stochastic beam, CoT, fixed-budget verified sampling (legacy name `self_consistency`), early-stopping verified sampling, and a separate symbolic solver. The search controls share legal moves and exact two-number endgames. These are local adaptations, not a reproduction of the paper's GPT-4 methods.

**Backend difference:** Qwen2.5-7B-Instruct in bitsandbytes NF4; JSON prompting with Python validation, **no constrained decoding grammar**. Windows used Ollama/GGUF. Compare methods within this Kaggle run; do not attribute cross-backend differences solely to the heuristic. [Quantization documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

Keep weights/settings fixed after development. Original-split overlap is audited below; overlapping puzzles are not independent validation. This notebook has not been executed on a Kaggle GPU by the authoring agent.''')
code('''import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers>=4.51,<5', 'accelerate>=1.6,<2', 'bitsandbytes>=0.45,<1',
    'huggingface_hub<1', 'pandas', 'matplotlib'])
# If Kaggle requests a kernel restart after installation, restart and continue below.
''')
code('''from pathlib import Path
import base64, hashlib, io, json, os, zipfile
PROJECT = Path('/kaggle/working/cost_game24')
PROJECT.mkdir(parents=True, exist_ok=True)
ARCHIVE_B64 = "'''+encoded+'''"
raw = base64.b64decode(ARCHIVE_B64)
assert hashlib.sha256(raw).hexdigest() == "'''+digest+'''"
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for member in archive.infolist():
        target = (PROJECT / member.filename).resolve()
        if PROJECT.resolve() not in target.parents:
            raise ValueError('Unsafe archive path')
        # Never overwrite modified experiment source/config on a rerun.
        content = archive.read(member.filename)
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError(f'Existing project differs: {target}. Use a fresh PROJECT directory.')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): target.write_bytes(content)
os.chdir(PROJECT)
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
print('Project restored:', PROJECT)
''')
md('''## Offline checks
These mock the model and do not perform inference. The original dataset download below does not contain solutions or human difficulty labels in model inputs.''')
code('''subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q'], check=True)
''')
code('''import csv, urllib.request
DATA = PROJECT / 'data' / 'original_tot'
DATA.mkdir(parents=True, exist_ok=True)
REPOSITORY = 'princeton-nlp/tree-of-thought-llm'
# Optionally set a known exact commit before the FIRST run. Once saved, it is reused.
REQUESTED_TOT_COMMIT = None
source_manifest = DATA / 'source_manifest.json'
def get_bytes(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'cost-game24-research'})
    with urllib.request.urlopen(req, timeout=60) as response: return response.read()
if source_manifest.exists():
    provenance = json.loads(source_manifest.read_text())
    commit = provenance['commit']
    if REQUESTED_TOT_COMMIT and REQUESTED_TOT_COMMIT != commit:
        raise RuntimeError('Dataset revision changed; use a new dataset/output directory.')
    csv_bytes = (DATA / '24.csv').read_bytes()
    assert hashlib.sha256(csv_bytes).hexdigest() == provenance['csv_sha256']
else:
    commit = REQUESTED_TOT_COMMIT or json.loads(get_bytes(
        f'https://api.github.com/repos/{REPOSITORY}/commits/master'))['sha']
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit.lower()):
        raise ValueError('Provide an exact 40-character Git commit.')
    url = f'https://raw.githubusercontent.com/{REPOSITORY}/{commit}/src/tot/data/24/24.csv'
    csv_bytes = get_bytes(url)
    license_bytes = get_bytes(f'https://raw.githubusercontent.com/{REPOSITORY}/{commit}/LICENSE')
    (DATA / '24.csv').write_bytes(csv_bytes)
    (DATA / 'SOURCE_LICENSE.txt').write_bytes(license_bytes)
    provenance = dict(repository=REPOSITORY, commit=commit, url=url,
                      csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
                      selected_zero_based_indices=[900,1000], split='official ToT evaluation range')
    source_manifest.write_text(json.dumps(provenance, indent=2))
source_rows = list(csv.DictReader(io.StringIO(csv_bytes.decode('utf-8-sig'))))
assert len(source_rows) >= 1000 and 'Puzzles' in source_rows[0]
puzzles = []
for index in range(900,1000):
    nums = list(map(int, source_rows[index]['Puzzles'].split()))
    assert len(nums)==4 and all(1<=n<=13 for n in nums)
    puzzles.append(dict(id=f'tot24-{index:04d}', numbers=nums))
assert len({tuple(sorted(p['numbers'])) for p in puzzles}) == 100
ORIGINAL_DATASET = DATA / 'tot_game24_900_999.jsonl'
contents = ''.join(json.dumps(p)+'\\n' for p in puzzles)
if ORIGINAL_DATASET.exists(): assert ORIGINAL_DATASET.read_text()==contents
else: ORIGINAL_DATASET.write_text(contents)
print('Loaded original ToT split:', len(puzzles), 'puzzles; commit:', commit)
''')
code('''import pandas as pd
# Audit all prepared sets, not just the previously inspected development subset.
audit=[]
for puzzle in puzzles:
    key=tuple(sorted(puzzle['numbers']))
    record=dict(id=puzzle['id'], numbers=puzzle['numbers'])
    for split in ('train','dev','test'):
        path=PROJECT / f'data/game24/game24_{split}.jsonl'
        if not path.exists(): raise RuntimeError(f'Missing overlap reference: {path}')
        old=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        record['overlap_generated_'+split]=key in {tuple(sorted(r['numbers'])) for r in old}
    audit.append(record)
overlap=pd.DataFrame(audit)
overlap.to_csv(DATA/'overlap_audit.csv',index=False)
display(overlap.filter(like='overlap_').sum().to_frame('puzzles'))
# Do not silently remove overlaps: preserve the official 100-puzzle split.
# Later report a supplementary disjoint subset as well as the full official split.
''')
md('''## Load one local model
A single GPU is used. NF4 reduces model memory; actual availability depends on the Kaggle accelerator. CUDA is required by this notebook. Do not change model/backend between methods or during a run. No model weights or credentials are included in the final results archive.''')
code('''import importlib.metadata, platform, torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'
print(torch.cuda.get_device_name(0))
MODEL_ID='Qwen/Qwen2.5-7B-Instruct'
MODEL_REVISION=None  # On a fresh run, resolve once and record its exact commit.
model_pin=PROJECT/'model_revision.json'
if model_pin.exists():
    pin=json.loads(model_pin.read_text())
    assert pin['model']==MODEL_ID
    if MODEL_REVISION: assert MODEL_REVISION==pin['revision']
    MODEL_REVISION=pin['revision']
else:
    MODEL_REVISION=MODEL_REVISION or HfApi().model_info(MODEL_ID).sha
    model_pin.write_text(json.dumps(dict(model=MODEL_ID,revision=MODEL_REVISION),indent=2))
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,trust_remote_code=False)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,
    quantization_config=quant,device_map={'':0},torch_dtype=torch.float16,
    attn_implementation='sdpa',trust_remote_code=False).eval()
import importlib.util
adapter_spec=importlib.util.spec_from_file_location('costbench_kaggle_adapter', PROJECT/'kaggle/transformers_backend.py')
adapter_module=importlib.util.module_from_spec(adapter_spec)
adapter_spec.loader.exec_module(adapter_module)
TransformersBackend=adapter_module.TransformersBackend
runtime=dict(backend='transformers-nf4-json-prompting',python=platform.python_version(),
    gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
    packages={p:importlib.metadata.version(p) for p in ('torch','transformers','accelerate','bitsandbytes','huggingface_hub')},
    adapter_sha256=hashlib.sha256((PROJECT/'kaggle/transformers_backend.py').read_bytes()).hexdigest())
backend=TransformersBackend(model,tokenizer,torch,MODEL_ID,MODEL_REVISION,runtime)
import costbench.backend as transport
import costbench.__main__ as core
import costbench.game24 as game24
transport.request=backend
core.request=backend
# Existing replay, budget accounting, exact checker and policies remain active.
(PROJECT/'runtime.json').write_text(json.dumps(runtime,indent=2))
(PROJECT/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
cfg=json.loads((PROJECT/'configs/game24_exp64.json').read_text())
cfg.update(model=MODEL_ID,base_url='inprocess://transformers',runtime_fingerprint=backend.digest)
KAGGLE_CONFIG=PROJECT/'configs/kaggle_game24.json'
if KAGGLE_CONFIG.exists() and json.loads(KAGGLE_CONFIG.read_text()) != cfg:
    raise RuntimeError('Runtime/config changed. Use a new project/output for this experiment.')
KAGGLE_CONFIG.write_text(json.dumps(cfg,indent=2))
print('Model revision:',MODEL_REVISION,'; config saved:',KAGGLE_CONFIG)
''')
md('''## Run all eight methods
Set `RUN_BENCHMARK=True` when ready. The default runs the original 100-puzzle split once with seed 42. Add seeds 43 and 44 for repetitions. Set `INCLUDE_GENERATED=True` to rerun your generated test set under the same Kaggle backend. This is optional and roughly doubles runtime.

A full comparison can take hours. Results and successful calls are saved after each method/call. Repeating identical settings resumes. To recover across Kaggle sessions, download/save the output archive and restore the project to the same path before loading the notebook cells. A saved model revision alone does not preserve results. Do not remove an active `run.lock`; the runner removes it in a `finally` block on normal interruption. After a hard session death, remove a stale lock only after confirming no runner exists. Changing `LIMIT` or packages requires a fresh output directory.''')
code('''import argparse
RUN_BENCHMARK=False
SEEDS=[42]               # Later use [42,43,44] with the same frozen environment.
LIMIT=100
INCLUDE_GENERATED=False
RETRY_FAILED=False       # Explicitly opt in only after diagnosing a backend failure.
datasets={'original_tot':ORIGINAL_DATASET}
if INCLUDE_GENERATED: datasets['generated']=PROJECT/'data/game24/game24_test.jsonl'
if RUN_BENCHMARK:
    for dataset_name,dataset in datasets.items():
        for seed in SEEDS:
            output=PROJECT/'results'/f'{dataset_name}_n{LIMIT}_seed{seed}_nf4'
            args=argparse.Namespace(config=str(KAGGLE_CONFIG),dataset=str(dataset),limit=LIMIT,
                                    seed=seed,output=str(output),retry_failed=RETRY_FAILED)
            game24.run(args)
else:
    print('Ready. Set RUN_BENCHMARK=True to start local model inference.')
''')
code('''# View all completed run summaries and produce a supplementary overlap-free score.
for directory in sorted((PROJECT/'results').glob('*')) if (PROJECT/'results').exists() else []:
    if not (directory/'results.jsonl').exists(): continue
    game24.report(directory)
    print(directory.name)
    display(pd.read_csv(directory/'summary.csv'))
    if directory.name.startswith('original_tot'):
        records=[json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines()]
        frame=pd.DataFrame(records)
        complete=frame.groupby('question_id')['method'].nunique()
        complete=set(complete[complete==len(game24.METHODS)].index)
        clean_ids=set(overlap.loc[~overlap.filter(like='overlap_').any(axis=1),'id'])
        clean=frame[frame.question_id.isin(clean_ids & complete)]
        if len(clean):
            supplemental=clean.groupby('method').agg(n=('correct','size'),accuracy=('correct','mean'),mean_tokens=('total_tokens','mean'),mean_calls=('calls','mean'))
            supplemental.to_csv(directory/'disjoint_subset_summary.csv')
            print('Supplementary subset with no overlap in ANY generated split; not the official full split:')
            display(supplemental)
''')
md('''## Save/download checkpoints and reports
Run this cell after stopping or finishing the benchmark. Save Kaggle notebook output as well. The archive contains source, config, dataset provenance, exact model revision, package versions, and raw calls/results. It excludes downloaded model weights. Do not merge NF4 results with older GGUF results as though they were one experiment.

Interpretation: Game of 24 is exactly solvable without an LLM. Keep the symbolic reference visible. Search methods receive deterministic legal moves/endgames that full-solution sampling does not; gains over sampling are not attributable solely to the heuristic. Use quality-only best-first as the main same-engine control.''')
code('''from IPython.display import FileLink
export=Path('/kaggle/working/game24_kaggle_results.zip')
with zipfile.ZipFile(export,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ('costbench','configs','data','results','kaggle'):
        for path in (PROJECT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.name!='run.lock':
                archive.write(path,Path('cost_game24')/path.relative_to(PROJECT))
    for name in ('runtime.json','model_revision.json','pip_freeze.txt','GAME24.md'):
        path=PROJECT/name
        if path.exists():archive.write(path,Path('cost_game24')/name)
display(FileLink(str(export)))
''')
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},cells=cells)
for i,c in enumerate(nb['cells']):c['id']=f'cell-{i:02d}'
out=root/'kaggle/Game24_Original_ToT_Kaggle.ipynb';out.write_text(json.dumps(nb,indent=1),encoding='utf-8')
print(out, out.stat().st_size)
for i,c in enumerate(cells):
    if c['cell_type']=='code':compile(c['source'],f'cell_{i}','exec')
print('All code cells compile; outputs empty; no notebook cells executed.')
