"""Build six uniform-dataset, one-variant Game of 24 Kaggle notebooks."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import build_dhruv_branch_notebook as original

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'kaggle' / 'game24_ablation_runner.py'
VARIANTS = ('heuristic_bestfirst', 'without_quality', 'without_success', 'without_depth', 'without_cost', 'without_diversity')
LABELS = {'heuristic_bestfirst':'original heuristic_bestfirst', **{x: x.replace('without_', 'without ') for x in VARIANTS[1:]}}

def build(method):
    output = ROOT / 'kaggle' / f'Game24_{method}.ipynb'
    revision = original.git('rev-parse', original.REF).decode().strip()
    intro = f'''# Game of 24 — {LABELS[method]}\n\nThis notebook runs exactly one method, `{method}`, on 100 fixed Game of 24 test puzzles.\n\nAll six notebooks embed the identical `game24_test.jsonl` dataset from the same frozen source snapshot, with the same puzzle IDs, ordering, model, and seed. This makes the six outputs directly comparable. The benchmark uses the exact arithmetic checker shared by the committed Game of 24 implementation.\n\nEnable Internet and GPU in Kaggle before Run All. Results are written after each puzzle and can be resumed.\n\nFrozen source commit: `{revision}`.\n'''
    settings = f'''LIMIT = 100\nMODEL = 'qwen2.5:7b-instruct'\nSEED = 42\nRETRY_FAILED = True\nMETHOD = {method!r}\nif not 1 <= LIMIT <= 100:\n    raise ValueError('LIMIT must be between 1 and 100')\n'''
    dirs = """from pathlib import Path
import os, sys
PROJECT_ROOT = Path('/kaggle/working/game24_' + METHOD)
for relative in ('costbench', 'tests', 'configs', 'data/game24', 'results'):
    (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
print('Restoring project under', PROJECT_ROOT)
"""
    run = """import json, re, subprocess, sys
config = json.loads((PROJECT_ROOT / 'configs/game24.json').read_text(encoding='utf-8'))
config['model'] = MODEL; config['base_url'] = 'http://127.0.0.1:11434'
configured = PROJECT_ROOT / 'configs/kaggle_game24.json'
configured.write_text(json.dumps(config, indent=2) + '\\n', encoding='utf-8')
dataset = PROJECT_ROOT / 'data/game24/game24_test.jsonl'
slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', MODEL)
run_name = f'game24_test_n{LIMIT}_seed{SEED}_{METHOD}_{slug}'
output = PROJECT_ROOT / 'results' / run_name
command = [sys.executable, '-u', str(PROJECT_ROOT / 'game24_ablation_runner.py'), '--config', str(configured), '--dataset', str(dataset), '--limit', str(LIMIT), '--seed', str(SEED), '--method', METHOD, '--output', str(output)]
if RETRY_FAILED: command.append('--retry-failed')
subprocess.run(command, cwd=PROJECT_ROOT, check=True)
print('Results:', output / (run_name + '_results.jsonl'))
"""
    download = """import os, shutil
from pathlib import Path
from IPython.display import FileLink, display
archive = shutil.make_archive(str(Path('/kaggle/working') / run_name), 'zip', root_dir=output)
previous_dir = Path.cwd()
try:
    os.chdir('/kaggle/working')
    display(FileLink(Path(archive).name))
    # Also expose individually named files without needing to extract a ZIP.
    for suffix in ('results.jsonl', 'summary.csv', 'manifest.json'):
        source = output / (run_name + '_' + suffix)
        if source.exists():
            target = Path('/kaggle/working') / source.name
            shutil.copy2(source, target)
            display(FileLink(target.name))
finally:
    os.chdir(previous_dir)
"""
    cells = [original.cell(intro, 'markdown'), original.cell('## 1. Settings\n', 'markdown'), original.cell(settings, role='settings'), original.cell('## 2. Restore identical source and dataset\n', 'markdown'), original.cell(dirs, role='directories')]
    for name in original.bundled_paths():
        relative = name.removeprefix('final_proj/')
        raw = original.git('show', f'{original.REF}:{name}')
        if relative.endswith('.py') or relative.startswith('configs/'):
            cells.append(original.cell(f'%%writefile {relative}\n' + raw.decode('utf-8'), role='source_file', file=relative, sha256=hashlib.sha256(raw).hexdigest()))
        else:
            cells.append(original.data_cell(relative, raw))
    runner_source = RUNNER.read_text(encoding='utf-8')
    cells += [original.cell('## 3. Runner\n', 'markdown'), original.cell('%%writefile game24_ablation_runner.py\n' + runner_source, role='runner', file='game24_ablation_runner.py', sha256=hashlib.sha256(runner_source.encode()).hexdigest()), original.cell('## 4. Offline checks\n', 'markdown'), original.cell("import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=PROJECT_ROOT, check=True)", role='offline_tests'), original.cell('## 5. Start Ollama and pull the model\n', 'markdown'), original.cell(original.OLLAMA_SETUP, role='ollama_setup'), original.cell(f'## 6. Run {method}\n', 'markdown'), original.cell(run, role='run'), original.cell('## 7. Download results\n', 'markdown'), original.cell(download, role='download')]
    for i, cell in enumerate(cells): cell['id'] = f'game24-{method}-{i:03d}'
    notebook = {'nbformat':4, 'nbformat_minor':5, 'metadata': {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}, 'language_info': {'name':'python','version':'3.11'}, 'kaggle': {'accelerator':'gpu','isInternetEnabled':True,'language':'python','sourceType':'notebook'}, 'source_commit': revision, 'dataset': 'game24_test.jsonl'}, 'cells': cells}
    output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Built', output)

if __name__ == '__main__':
    for variant in VARIANTS: build(variant)
