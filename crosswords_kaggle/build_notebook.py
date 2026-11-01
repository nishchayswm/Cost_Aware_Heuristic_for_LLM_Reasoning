"""Build a portable notebook without executing any notebook cells."""
from pathlib import Path
import base64
import hashlib
import io
import json
import textwrap
import zipfile

ROOT=Path(__file__).resolve().parent
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in ('crossword_bench.py','test_crossword_bench.py','README.md'):
        archive.write(ROOT/name,name)
encoded=base64.b64encode(buffer.getvalue()).decode()
sha=hashlib.sha256(buffer.getvalue()).hexdigest()
cells=[]
def md(source):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(source).strip()+'\n'))
def code(source):cells.append(dict(cell_type='code',metadata={},source=textwrap.dedent(source).strip()+'\n',outputs=[],execution_count=None))

md('''# 50 Mini Crosswords: Your Heuristic vs ToT-style DFS
Self-contained Kaggle notebook. Enable **GPU** and **Internet**, then run the cells in order. The benchmark is disabled until you set `RUN_BENCHMARK=True`.

**Exactly 50 puzzles**, sampled once from the [original ToT mini-crossword dataset](https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/data/crosswords/mini0505.json). This is a custom seeded random split, not the paper's published evaluation subset. Selected IDs and source revision are saved.

Two methods: **ToT-style DFS** and **five-factor heuristic best-first**. The [official ToT crossword implementation](https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/scripts/crosswords/search_crosswords-dfs.ipynb) uses DFS. Our baseline adapts it to a local model and a shared engine; it is not an exact reproduction or a claim of current SOTA.

Both methods receive the same model, prompts, revision rules and limits. They run one puzzle at a time, printing both results before proceeding. Reference solutions never enter model prompts, pruning, stopping or final-state selection.

**Not run on a Kaggle GPU during creation.** Offline tests use mocks. Other benchmarks and previous project files are unchanged.''')
code('''import sys, subprocess
subprocess.check_call([sys.executable,'-m','pip','install','-q',
    'transformers>=4.51,<5','accelerate>=1.6,<2','bitsandbytes>=0.45,<1',
    'huggingface_hub<1','lm-format-enforcer==0.11.3','pandas','matplotlib'])
# Restart the kernel after installation, then continue with the next cell.
''')
code('''from pathlib import Path
import base64, hashlib, io, json, os, zipfile
PROJECT=Path('/kaggle/working/crosswords50_schema_v2')
PROJECT.mkdir(parents=True,exist_ok=True)
# Preserve the original selection and model pin when upgrading an existing session.
import shutil
previous=Path('/kaggle/working/crosswords50')
if (previous/'data').exists() and not (PROJECT/'data').exists():
    shutil.copytree(previous/'data',PROJECT/'data')
if (previous/'model_pin.json').exists() and not (PROJECT/'model_pin.json').exists():
    shutil.copy2(previous/'model_pin.json',PROJECT/'model_pin.json')
payload=base64.b64decode("'''+encoded+'''")
assert hashlib.sha256(payload).hexdigest()=="'''+sha+'''"
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        target=(PROJECT/name).resolve()
        if PROJECT.resolve() not in target.parents:raise ValueError('Unsafe archive path')
        content=archive.read(name)
        if target.exists() and target.read_bytes()!=content:
            raise RuntimeError('Existing source differs. Use a fresh PROJECT path.')
        if not target.exists():target.write_bytes(content)
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
print('Project:',PROJECT)
''')
md('''## 1. Check code and prepare the 50-puzzle dataset
The tests below make no network/model calls. Dataset preparation downloads the original source file and license; it does not run inference. Selection seed is separate from the LLM trial seed. Keep selected puzzles and heuristic settings fixed throughout evaluation.''')
code('''subprocess.run([sys.executable,'-m','unittest','test_crossword_bench','-q'],check=True)
sys.modules.pop('crossword_bench',None)
import crossword_bench as bench
SELECTION_SEED=2026
puzzles=bench.prepare_dataset(PROJECT/'data',selection_seed=SELECTION_SEED)
assert len(puzzles)==50
print('Selected:',len(puzzles),'puzzles')
print('First five IDs:',[p['id'] for p in puzzles[:5]])
print('Source metadata:',PROJECT/'data/source_manifest.json')
# Do not inspect solutions to tune the policy on these evaluation puzzles.
''')
md('''## 2. Load Qwen on one GPU
Uses `Qwen/Qwen2.5-7B-Instruct` in bitsandbytes 4-bit NF4. A model revision is pinned on first load. JSON Schema constrains both generation and evaluation. Move words must contain exactly five English letters. Scores and rating IDs are also validated locally; at most two retries are allowed and all calls/tokens count. Formatting constraints do not guarantee correct clue answers.

The same backend is used for both policies. Results from NF4 should not be pooled with older Ollama/GGUF experiments as if the backend were identical. Model loading can take several minutes.''')
code('''import torch, platform, importlib.metadata
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
assert torch.cuda.is_available(),'Enable GPU in Kaggle notebook settings.'
MODEL_ID='Qwen/Qwen2.5-7B-Instruct'
pin_path=PROJECT/'model_pin.json'
if pin_path.exists():
    pin=json.loads(pin_path.read_text())
    assert pin['model']==MODEL_ID
else:
    pin=dict(model=MODEL_ID,revision=HfApi().model_info(MODEL_ID).sha)
    bench.write_json(pin_path,pin)
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=pin['revision'],trust_remote_code=False)
if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=pin['revision'],
    quantization_config=quant,device_map={'':0},torch_dtype=torch.float16,
    attn_implementation='sdpa',trust_remote_code=False).eval()
client=bench.HFClient(model,tokenizer,torch)
runtime=dict(**pin,backend='transformers-nf4-json-schema-v2',gpu=torch.cuda.get_device_name(0),
    cuda=torch.version.cuda,python=platform.python_version(),
    packages={name:importlib.metadata.version(name) for name in ('torch','transformers','accelerate','bitsandbytes','huggingface_hub','lm-format-enforcer')})
bench.write_json(PROJECT/'runtime.json',runtime)
(PROJECT/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
print('Loaded:',MODEL_ID, runtime['gpu'],pin['revision'])
''')
md('''## 3. Frozen comparison settings
`H = 0.35 Q + 0.30 P - 0.05 D - 0.15 C + 0.15 V`

- **Q**: judged plausibility of complete clue answers, including implied crossing words.
- **P**: judged feasibility of the full puzzle; uncalibrated, not measured success probability.
- **D**: action depth / maximum depth, multiplied by `(1-P)`.
- **C**: future-work proxy based on unassigned-slot fraction and observed mean call tokens, normalized by budget.
- **V**: minimum word-assignment distance to another retained state, gated by `sqrt(Q*P)`.

Both policies propose at most 3 moves and batch-evaluate their children. Revising a word removes conflicting crossing assignments. DFS orders siblings by proposed confidence; best-first ranks the global frontier by H. Both prune at the same P threshold and choose their final output by the same `coverage * Q` ranking. A filled grid is never assumed correct.

Default per-method limits: **30,000 tokens, 40 calls, 20 expansions**. All generation and judging costs count. The full 50-puzzle comparison can take hours. Do not tune on these 50 test answers. See the embedded README for feature definitions and limitations.''')
code('''import copy
config=copy.deepcopy(bench.DEFAULT)
bench.write_json(PROJECT/'config.json',config)
print(json.dumps(config,indent=2))
''')
md('''## Preflight: validate generation and evaluation before the full run
This small synthetic check makes real model calls and saves their costs separately under `preflight/`. It checks response format and legal placements, without scoring against any evaluation answers. It must pass before the 50-puzzle run.''')
code('''import hashlib
preflight_config={**config,'max_expansions':1}
preflight_clues=['Fruit often baked in pies']*10
preflight_key=bench.digest(dict(config=preflight_config,runtime=runtime,
    code=hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest()))[:12]
preflight_dir=PROJECT/'preflight'/preflight_key
preflight_session=bench.Session(client,preflight_config,preflight_dir/'calls','format-preflight',42)
preflight_result,preflight_trace=bench.solve(preflight_clues,'heuristic_bestfirst',preflight_session,preflight_config)
bench.write_json(preflight_dir/'result.json',preflight_result)
bench.write_json(preflight_dir/'trace.json',preflight_trace)
PREFLIGHT_PASSED=(preflight_result['calls']>=2 and preflight_result['prediction']!='_'*25
    and preflight_result['stop_reason']!='invalid_model_response')
print(json.dumps(preflight_result,indent=2))
assert PREFLIGHT_PASSED,'Preflight failed: inspect the saved raw calls before running 50 puzzles.'
print('Preflight passed. Structured generation and evaluation produced a legal placement.')
''')
md('''## 4. Run exactly 50 puzzles
Set `RUN_BENCHMARK=True` and execute this cell. Start with trial seed 42. Additional seeds repeat the same 50 puzzles; they do not select more puzzles. Both methods run for each puzzle, with alternating order.

Completed methods and successful calls can be resumed with identical settings. A failed/interrupted inference whose token cost is unknown blocks that trial; start a fresh output trial rather than hide that cost. After a hard session reset, a stale `run.lock` must only be removed when no run is active. Save the export ZIP before ending your session.''')
code('''RUN_BENCHMARK=False
TRIAL_SEEDS=[42]  # Optional later repetition: [42,43,44]
RUN_LABEL='schema_v2'  # Use a new label for changed settings or a fresh trial.
if RUN_BENCHMARK:
    assert globals().get('PREFLIGHT_PASSED',False),'Run the preflight cell successfully first.'
    for seed in TRIAL_SEEDS:
        output=PROJECT/'results'/f'{RUN_LABEL}_seed{seed}'
        summary=bench.run(puzzles,client,config,runtime,output,seed=seed)
        print(json.dumps(summary,indent=2))
else:
    print('Ready. Set RUN_BENCHMARK=True to start the benchmark.')
''')
md('''## 5. Compare scores and costs
Word accuracy is the primary score. Also report full-grid success, letter accuracy, token cost, calls and runtime. Summaries use only paired completed puzzles. The paired bootstrap resamples puzzles (not individual letters); it does not correct for related clues across puzzles or replace a larger evaluation. A better score alone does not establish SOTA.''')
code('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
for directory in sorted((PROJECT/'results').glob('*')):
    if not (directory/'records').exists():continue
    summary=bench.report(directory)
    print(directory.name)
    table=pd.DataFrame(summary);display(table)
    paired_path=directory/'per_question.csv'
    if not paired_path.exists():continue
    paired=pd.read_csv(paired_path)
    delta=(paired['heuristic_bestfirst_word_accuracy']-paired['tot_dfs_word_accuracy']).to_numpy()
    rng=np.random.default_rng(123)
    means=rng.choice(delta,size=(5000,len(delta)),replace=True).mean(axis=1)
    comparison=dict(paired_puzzles=len(delta),mean_word_accuracy_difference=float(delta.mean()),
        bootstrap_95_percent_interval=np.quantile(means,[.025,.975]).tolist())
    bench.write_json(directory/'paired_comparison.json',comparison)
    print('Heuristic minus DFS:',comparison)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(table.method,table.mean_word_accuracy);axes[0].set_ylim(0,1)
    axes[0].set_title('Word accuracy');axes[0].tick_params(axis='x',rotation=15)
    axes[1].bar(table.method,table.mean_total_tokens)
    axes[1].set_title('Mean input + output tokens');axes[1].tick_params(axis='x',rotation=15)
    fig.tight_layout();fig.savefig(directory/'comparison.png',dpi=160);plt.show()
''')
md('''## 6. Download results or a checkpoint
Run after completing or stopping the benchmark. The ZIP includes code, exact configuration, dataset provenance, selected questions and answer keys, per-problem results, summaries, raw calls, traces and plots. It excludes model weights. Also save Kaggle notebook outputs.

For another session, attach your own ZIP as input and extract it under `/kaggle/working` before running setup; it restores the `crosswords50_schema_v2` directory. Resume requires the same experiment/runtime. A GPU/session interruption before a call is saved leaves unknown cost; use a fresh trial for clean cost measurement.''')
code('''from IPython.display import FileLink
export=Path('/kaggle/working/crosswords50_schema_v2_results.zip')
with zipfile.ZipFile(export,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in PROJECT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='run.lock':
            archive.write(path,Path('crosswords50_schema_v2')/path.relative_to(PROJECT))
previous_dir=Path.cwd()
try:
    os.chdir(export.parent)
    display(FileLink(export.name))
finally:
    os.chdir(previous_dir)
''')
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},cells=cells)
for index,cell in enumerate(cells):
    cell['id']=f'crosswords-{index:02d}'
    if cell['cell_type']=='code':compile(cell['source'],f'cell-{index}','exec')
path=ROOT/'Crosswords50_Heuristic_vs_ToT_Kaggle.ipynb'
path.write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8',newline='\n')
print('Created:',path,'; all code cells compile. No notebook cells executed.')
