"""Build a standalone Kaggle notebook for Dhruv's five-factor leave-one-out study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import build_dhruv_branch_notebook as original


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'kaggle' / 'Dhruv_five_factor_ablations.ipynb'
RUNNER = ROOT / 'kaggle' / 'dhruv_ablation_runner.py'

INTRO = """# Dhruv's five-factor priority ablations

This notebook runs **only six variants** on GSM8K: the V6 five-factor
`heuristic_bestfirst` and one leave-one-out ablation for each ranking feature:
Q (quality), P (success confidence), depth, predicted next-call token cost, and
TF-IDF diversity. It does not run CoT, self-consistency, ToT beam or other methods.

Each ablation sets one priority weight to zero and renormalizes the other four.
For Q/P removal, its contribution to the depth/diversity interactions is also
removed. Generation, judging, arithmetic checks, the initial complete-solution
candidate, search limits, and final-answer selection stay the same. Q and P still
affect the **common final-answer selector**; this is a ranking ablation, not a
whole-system deletion. All variants use the same random seed stream for identical
prompts. Actual calls and tokens are measured separately for every variant.

**Enable Internet and GPU in Kaggle before Run All.** Default is one development
question as a smoke test. Set `LIMIT` to 10 or 100 for more development questions;
freeze choices before `SPLIT='test'`. Six variants × 100 questions means 600
method runs, each potentially using many model calls.

The teammate's committed Python modules and fixed GSM8K datasets are embedded
below. Results flush after each variant. An identical rerun skips completed
variants and replays successful calls from an interrupted one. Failed Ollama calls
can be retried, but their unreported token cost remains unknown in the report.
"""

SETTINGS = """# Edit before Run All.
SPLIT = 'dev'                  # 'dev' first; 'test' only after settings are frozen
LIMIT = 1                      # 1 smoke question, 10 pilot, 100 full split
MODEL = 'qwen2.5:7b'
SEED = 42
RETRY_FAILED = True

if SPLIT not in ('dev', 'test'):
    raise ValueError('SPLIT must be dev or test')
if type(LIMIT) is not int or not 1 <= LIMIT <= 100:
    raise ValueError('LIMIT must be an integer from 1 to 100')
"""

DIRECTORIES = """from pathlib import Path
import os, sys

PROJECT_ROOT = Path('/kaggle/working/dhruv_ablation_proj')
for relative in ('costbench', 'tests', 'configs', 'data/game24', 'results'):
    (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print('Restoring Dhruv branch snapshot under', PROJECT_ROOT)
"""

OFFLINE_TESTS = """import subprocess, sys

# Teammate's tests use mocked model responses and make no Ollama requests.
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
               cwd=PROJECT_ROOT, check=True)
from costbench.__main__ import config as load_config
from ablation_runner import FEATURES, VARIANTS, variant_config
base = load_config(PROJECT_ROOT / 'configs/default.json')
assert len(VARIANTS) == 6 and len(FEATURES) == 5
for feature in FEATURES:
    variant = variant_config(base, feature)
    assert variant['weights'][feature] == 0
    assert abs(sum(variant['weights'].values()) - 1) < 1e-9
print('All five leave-one-out configs are valid.')
"""

RUN = """import json, re, subprocess, sys

config = json.loads((PROJECT_ROOT / 'configs/default.json').read_text(encoding='utf-8'))
config['model'] = MODEL
config['base_url'] = 'http://127.0.0.1:11434'
configured = PROJECT_ROOT / 'configs/kaggle_ablations.json'
configured.write_text(json.dumps(config, indent=2) + '\\n', encoding='utf-8')
dataset = PROJECT_ROOT / ('data/gsm8k_' + SPLIT + '_100.jsonl')
slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', MODEL)
run_name = f'gsm8k_{SPLIT}_n{LIMIT}_seed{SEED}_{slug}_five_factor_ablations'
output = PROJECT_ROOT / 'results' / run_name
command = [sys.executable, '-u', str(PROJECT_ROOT / 'ablation_runner.py'),
           '--config', str(configured), '--dataset', str(dataset),
           '--label-corrections', str(PROJECT_ROOT / 'data/label_corrections.json'),
           '--limit', str(LIMIT), '--seed', str(SEED), '--output', str(output)]
if RETRY_FAILED:
    command.append('--retry-failed')
print('Running six paired variants:', run_name, flush=True)
try:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
except subprocess.CalledProcessError:
    import os, shutil
    from pathlib import Path
    from IPython.display import FileLink, display
    if output.exists():
        archive = shutil.make_archive(
            str(Path('/kaggle/working') / (run_name + '_partial')), 'zip',
            root_dir=output)
        previous_dir = Path.cwd()
        try:
            os.chdir('/kaggle/working')
            print('Partial results saved; download then rerun this cell:')
            display(FileLink(Path(archive).name))
        finally:
            os.chdir(previous_dir)
    raise
print('Results:', output)
"""


def build():
    revision = original.git('rev-parse', original.REF).decode().strip()
    cells = [original.cell(INTRO + f'\nSource commit: `{revision}`.\n', 'markdown'),
             original.cell('## 1. Settings\n', 'markdown'),
             original.cell(SETTINGS, role='settings'),
             original.cell('## 2. Restore committed source and data\n', 'markdown'),
             original.cell(DIRECTORIES, role='directories')]
    for name in original.bundled_paths():
        relative = name.removeprefix('final_proj/')
        raw = original.git('show', f'{original.REF}:{name}')
        if relative.endswith('.py') or relative.startswith('configs/'):
            cells.append(original.cell(f'%%writefile {relative}\n' + raw.decode('utf-8'),
                                       role='source_file', file=relative,
                                       sha256=hashlib.sha256(raw).hexdigest()))
        else:
            cells.append(original.data_cell(relative, raw))
    runner_source = RUNNER.read_text(encoding='utf-8')
    cells += [original.cell('## 3. Ranking ablation runner\n', 'markdown'),
              original.cell('%%writefile ablation_runner.py\n' + runner_source,
                            role='ablation_runner', file='ablation_runner.py',
                            sha256=hashlib.sha256(runner_source.encode()).hexdigest()),
              original.cell('## 4. Offline tests\n', 'markdown'),
              original.cell(OFFLINE_TESTS, role='offline_tests'),
              original.cell('## 5. Start Ollama and pull the model\n', 'markdown'),
              original.cell(original.OLLAMA_SETUP, role='ollama_setup'),
              original.cell('## 6. Run the six variants\n', 'markdown'),
              original.cell(RUN, role='run'),
              original.cell('## 7. Download results\n', 'markdown'),
              original.cell(original.DOWNLOAD, role='download'),
              original.cell('## Reading the results\n\n'
                            '`summary.csv` compares only question blocks completed by all six variants. '
                            '`paired_comparisons.csv` reports **main minus ablation**: a positive '
                            'accuracy difference favors the main heuristic, while a positive token '
                            'difference means the main heuristic spent more tokens. Inspect '
                            '`per_question.csv`, `results.jsonl`, `traces/`, and `calls/` for '
                            'failure causes. `unknown_cost_results` flags resumed HTTP failures '
                            'whose token usage cannot be recovered.\n', 'markdown')]
    for index, item in enumerate(cells):
        item['id'] = f'dhruv-ablation-{index:03d}'
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True,
                   'language': 'python', 'sourceType': 'notebook'},
        'source_branch': original.REF, 'source_commit': revision,
    }, 'cells': cells}
    OUTPUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n',
                      encoding='utf-8')
    print('Built', OUTPUT, '| cells', len(cells))
    return OUTPUT


if __name__ == '__main__':
    build()
