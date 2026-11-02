"""Build a self-contained Kaggle notebook for 100 Game of 24 ablation runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import build_dhruv_branch_notebook as original

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'kaggle' / 'Game24_five_factor_ablations.ipynb'
RUNNER = ROOT / 'kaggle' / 'game24_ablation_runner.py'

INTRO = """# Game of 24: five-factor heuristic ablations

This notebook runs 100 fixed, solvable Game of 24 test puzzles with six paired
variants: the original `heuristic_bestfirst` and five leave-one-feature-out
ablations for quality, success confidence, depth, predicted token cost, and
diversity. Each ablation sets one priority weight to zero and renormalizes the
remaining weights. The exact arithmetic checker remains shared by all methods.

The dataset is locally generated from unordered four-number multisets and is
embedded from the committed branch snapshot. It is not the original published
Thought of Search benchmark. Enable Internet and GPU in Kaggle before Run All.
The model is queried through Ollama; results are flushed after every puzzle and
variant so an interrupted run can resume.
"""

SETTINGS = """# Full requested experiment. Use LIMIT=1 first if you want a smoke test.
LIMIT = 100
MODEL = 'qwen2.5:7b-instruct'
SEED = 42
RETRY_FAILED = True
SPLIT = 'test'
if type(LIMIT) is not int or not 1 <= LIMIT <= 100:
    raise ValueError('LIMIT must be an integer from 1 to 100')
"""

DIRECTORIES = """from pathlib import Path
import os, sys
PROJECT_ROOT = Path('/kaggle/working/game24_ablation_proj')
for relative in ('costbench', 'tests', 'configs', 'data/game24', 'results'):
    (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print('Restoring project under', PROJECT_ROOT)
"""

RUN = """import json, re, subprocess, sys
config = json.loads((PROJECT_ROOT / 'configs/game24.json').read_text(encoding='utf-8'))
config['model'] = MODEL
config['base_url'] = 'http://127.0.0.1:11434'
configured = PROJECT_ROOT / 'configs/kaggle_game24.json'
configured.write_text(json.dumps(config, indent=2) + '\\n', encoding='utf-8')
dataset = PROJECT_ROOT / ('data/game24/game24_' + SPLIT + '.jsonl')
slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', MODEL)
run_name = f'game24_{SPLIT}_n{LIMIT}_seed{SEED}_{slug}_five_factor_ablations'
output = PROJECT_ROOT / 'results' / run_name
command = [sys.executable, '-u', str(PROJECT_ROOT / 'game24_ablation_runner.py'),
           '--config', str(configured), '--dataset', str(dataset),
           '--limit', str(LIMIT), '--seed', str(SEED), '--output', str(output)]
if RETRY_FAILED:
    command.append('--retry-failed')
print('Running six paired variants:', run_name, flush=True)
subprocess.run(command, cwd=PROJECT_ROOT, check=True)
print('Results:', output)
"""

DOWNLOAD = """import shutil
from pathlib import Path
from IPython.display import FileLink, display
archive = shutil.make_archive(str(Path('/kaggle/working') / run_name), 'zip', root_dir=output)
display(FileLink(Path(archive).name))
"""


def build():
    revision = original.git('rev-parse', original.REF).decode().strip()
    cells = [original.cell(INTRO + f'\nFrozen source commit: `{revision}`.\n', 'markdown'),
             original.cell('## 1. Settings\n', 'markdown'), original.cell(SETTINGS, role='settings'),
             original.cell('## 2. Restore committed Game of 24 source and data\n', 'markdown'),
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
    cells += [original.cell('## 3. Six-variant ablation runner\n', 'markdown'),
              original.cell('%%writefile game24_ablation_runner.py\n' + runner_source,
                            role='ablation_runner', file='game24_ablation_runner.py',
                            sha256=hashlib.sha256(runner_source.encode()).hexdigest()),
              original.cell('## 4. Offline checks\n', 'markdown'),
              original.cell("import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=PROJECT_ROOT, check=True)\nprint('Offline tests passed.')", role='offline_tests'),
              original.cell('## 5. Start Ollama and pull the model\n', 'markdown'),
              original.cell(original.OLLAMA_SETUP, role='ollama_setup'),
              original.cell('## 6. Run 100 Game of 24 puzzles across six variants\n', 'markdown'),
              original.cell(RUN, role='run'),
              original.cell('## 7. Download results\n', 'markdown'), original.cell(DOWNLOAD, role='download')]
    for index, item in enumerate(cells): item['id'] = f'game24-ablation-{index:03d}'
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': '3.11'},
                             'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True, 'language': 'python', 'sourceType': 'notebook'},
                             'source_commit': revision}, 'cells': cells}
    OUTPUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Built', OUTPUT, '| cells', len(cells))


if __name__ == '__main__': build()
