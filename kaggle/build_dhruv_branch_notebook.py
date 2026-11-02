"""Package Dhruv's committed final_proj snapshot as a self-contained Kaggle notebook."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import textwrap
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF = 'origin/dhruv8361343-patch-1'
OUTPUT = ROOT / 'kaggle' / 'Dhruv_branch_benchmark.ipynb'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def cell(source, kind='code', **metadata):
    item = {'cell_type': kind, 'metadata': metadata,
            'source': source.splitlines(keepends=True)}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    return item


def bundled_paths():
    names = git('ls-tree', '-r', '--name-only', REF, 'final_proj').decode().splitlines()
    return [name for name in names if (
        name.startswith('final_proj/costbench/') and name.endswith('.py')
        or name.startswith('final_proj/tests/') and name.endswith('.py')
        or name in ('final_proj/configs/default.json', 'final_proj/configs/game24.json')
        or name.startswith('final_proj/data/') and
        (name.endswith('.json') or name.endswith('.jsonl'))
    )]


def data_cell(relative_path, raw):
    compressed = base64.b64encode(zlib.compress(raw, level=9)).decode('ascii')
    chunks = textwrap.wrap(compressed, width=96)
    quoted = '\n'.join('    ' + repr(chunk) for chunk in chunks)
    source = (
        'import base64, zlib\n'
        'from pathlib import Path\n'
        f'path = Path({relative_path!r})\n'
        'path.parent.mkdir(parents=True, exist_ok=True)\n'
        'path.write_bytes(zlib.decompress(base64.b64decode(\n'
        + quoted + '\n)))\n'
    )
    return cell(source, role='bundled_data', file=relative_path,
                sha256=hashlib.sha256(raw).hexdigest())


SETTINGS = """# Edit these settings before Run All.
TASK = 'gsm8k'                 # 'gsm8k' or 'game24'
SPLIT = 'dev'                  # 'dev' first; use 'test' after settings are frozen
LIMIT = 1                      # Smoke run; use 10, then 100 for a larger run
MODEL = 'qwen2.5:7b'           # The model available to this team
SEED = 42
RETRY_FAILED = True           # On rerun, retry a failed Ollama call and replay saved calls

if TASK not in ('gsm8k', 'game24') or SPLIT not in ('dev', 'test'):
    raise ValueError('Choose TASK gsm8k/game24 and SPLIT dev/test')
if type(LIMIT) is not int or not 1 <= LIMIT <= 100:
    raise ValueError('LIMIT must be an integer from 1 to 100')
"""


DIRECTORIES = """from pathlib import Path
import os

PROJECT_ROOT = Path('/kaggle/working/dhruv_final_proj')
for relative in ('costbench', 'tests', 'configs', 'data/game24', 'results'):
    (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_ROOT)
print('Restoring committed branch snapshot under', PROJECT_ROOT)
"""


OFFLINE_TESTS = """import subprocess, sys

# These tests use mocked model responses; they make no Ollama requests.
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
               cwd=PROJECT_ROOT, check=True)
"""


OLLAMA_SETUP = """import shutil, subprocess, time, urllib.request

if shutil.which('ollama') is None:
    subprocess.run(['apt-get', 'update', '-y'], check=True)
    subprocess.run(['apt-get', 'install', '-y', 'zstd', 'pciutils'], check=True)
    subprocess.run('curl -fsSL https://ollama.com/install.sh | sh',
                   shell=True, check=True)

try:
    urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2).close()
except Exception:
    ollama_log = open(PROJECT_ROOT / 'ollama_server.log', 'a', encoding='utf-8')
    ollama_process = subprocess.Popen(['ollama', 'serve'], stdout=ollama_log,
                                      stderr=ollama_log)
    for attempt in range(60):
        try:
            urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2).close()
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('Ollama did not start. Check ollama_server.log')

subprocess.run(['ollama', 'pull', MODEL], check=True)
print('Ollama ready with', MODEL)
"""


RUN = """import json, re, subprocess, sys

base_config = 'configs/default.json' if TASK == 'gsm8k' else 'configs/game24.json'
dataset = ('data/gsm8k_' + SPLIT + '_100.jsonl' if TASK == 'gsm8k'
           else 'data/game24/game24_' + SPLIT + '.jsonl')
config = json.loads((PROJECT_ROOT / base_config).read_text(encoding='utf-8'))
config['model'] = MODEL
config['base_url'] = 'http://127.0.0.1:11434'
configured = PROJECT_ROOT / ('configs/kaggle_' + TASK + '.json')
configured.write_text(json.dumps(config, indent=2) + '\\n', encoding='utf-8')

slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', MODEL)
run_name = f'{TASK}_{SPLIT}_n{LIMIT}_seed{SEED}_{slug}'
output = PROJECT_ROOT / 'results' / run_name
module = 'costbench' if TASK == 'gsm8k' else 'costbench.game24'
command = [sys.executable, '-u', '-m', module, 'run', '--config', str(configured),
           '--dataset', str(PROJECT_ROOT / dataset), '--limit', str(LIMIT),
           '--seed', str(SEED), '--output', str(output)]
if RETRY_FAILED:
    command.append('--retry-failed')
print('Running:', ' '.join(command), flush=True)
try:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
except subprocess.CalledProcessError:
    # Successful methods and calls are already on disk. Offer a backup before retrying.
    import os, shutil
    from pathlib import Path
    from IPython.display import FileLink, display
    if output.exists():
        partial_archive = shutil.make_archive(
            str(Path('/kaggle/working') / (run_name + '_partial')), 'zip',
                                              root_dir=output)
        print('Partial results saved. Download this backup, then rerun this cell:')
        previous_dir = Path.cwd()
        try:
            os.chdir('/kaggle/working')
            display(FileLink(Path(partial_archive).name))
        finally:
            os.chdir(previous_dir)
    raise
print('Results:', output)
"""


DOWNLOAD = """import os, shutil
from pathlib import Path
from IPython.display import FileLink, display

archive = shutil.make_archive(str(Path('/kaggle/working') / run_name),
                              'zip', root_dir=output)
print('Download the result archive:')
previous_dir = Path.cwd()
try:
    os.chdir('/kaggle/working')
    display(FileLink(Path(archive).name))
finally:
    os.chdir(previous_dir)
print('Report is inside the ZIP as report.html')
"""


def build():
    revision = git('rev-parse', REF).decode().strip()
    paths = bundled_paths()
    if not paths:
        raise RuntimeError('No final_proj files found on ' + REF)
    cells = [
        cell('# Dhruv branch: Kaggle benchmark\n\n'
             f'Frozen source: `{REF}` at `{revision}`. This notebook restores the '
             'committed `final_proj` project, runs its offline tests, then starts one '
             'development question by default. It includes GSM8K and Game of 24.\n\n'
             '**Enable Internet and a GPU in Kaggle settings before Run All.** '
             'The first run installs Ollama and downloads Qwen `qwen2.5:7b`. '
             'Only standard-library Python packages are needed for the benchmark.\n\n'
             'For a larger run, change `LIMIT` to 10 or 100 in the settings cell. '
             'Set `SPLIT="test"` only after choosing settings on development data. '
             'Results are saved after each method and an identical run can resume. '
             'A failed Ollama call stops the run; rerun the benchmark cell to retry it. '
             'The failed call may have unknown token cost, which reports flag. '
             'Changing settings creates a separate output folder.\n', 'markdown'),
        cell('## 1. Settings\n', 'markdown'), cell(SETTINGS, role='settings'),
        cell('## 2. Restore the committed project\n', 'markdown'),
        cell(DIRECTORIES, role='directories'),
    ]
    for name in paths:
        relative = name.removeprefix('final_proj/')
        raw = git('show', f'{REF}:{name}')
        if relative.endswith('.py') or relative.startswith('configs/'):
            source = raw.decode('utf-8')
            cells.append(cell(f'%%writefile {relative}\n' + source,
                              role='source_file', file=relative,
                              sha256=hashlib.sha256(raw).hexdigest()))
        else:
            cells.append(data_cell(relative, raw))
    cells += [
        cell('## 3. Offline checks\n', 'markdown'), cell(OFFLINE_TESTS, role='offline_tests'),
        cell('## 4. Start Ollama and pull the model\n', 'markdown'),
        cell(OLLAMA_SETUP, role='ollama_setup'),
        cell('## 5. Run the selected benchmark\n', 'markdown'), cell(RUN, role='run'),
        cell('## 6. Download the results\n', 'markdown'),
        cell(DOWNLOAD, role='download'),
    ]
    for index, item in enumerate(cells):
        item['id'] = f'dhruv-{index:03d}'
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True,
                   'language': 'python', 'sourceType': 'notebook'},
        'source_branch': REF, 'source_commit': revision,
    }, 'cells': cells}
    OUTPUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n',
                      encoding='utf-8')
    print('Built', OUTPUT, '| commit', revision, '| bundled files', len(paths),
          '| cells', len(cells))
    return OUTPUT


if __name__ == '__main__':
    build()
