"""Mechanically embed the shared sources in the portable Kaggle notebook.

Run after changing the engine, dataset, runner, or plotting code. Notebook
outputs are deliberately cleared so old outputs cannot masquerade as v4 results.
"""
import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def readable_source(name):
    """Inline source, removing only repository imports and CLI entry points.

    Notebook cells share a namespace, so these imports would be redundant.
    Preserve the original text and line breaks instead of escaping whole files.
    """
    source = (ROOT / (name + '.py')).read_text(encoding='utf-8')
    if name == 'problems':
        # Wrap long question literals using adjacent strings; their values stay identical.
        problem_lines = source.splitlines(keepends=True)
        strings = [node for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)
                   and len(node.value) > 88 and node.lineno == node.end_lineno]
        for node in sorted(strings, key=lambda item: (item.lineno, item.col_offset), reverse=True):
            original = problem_lines[node.lineno - 1]
            chunks = textwrap.wrap(node.value, width=80, drop_whitespace=False,
                                   replace_whitespace=False, break_long_words=False,
                                   break_on_hyphens=False)
            wrapped = '(\n        ' + '\n        '.join(repr(chunk) for chunk in chunks) + '\n    )'
            problem_lines[node.lineno - 1] = (original[:node.col_offset] + wrapped
                                             + original[node.end_col_offset:])
        source = ''.join(problem_lines)
    lines = source.splitlines(keepends=True)
    internal = {'benchmark_engine', 'benchmark_runtime', 'problems', 'plot_results', 'llm', 'tos_search',
                'benchmark_data'}
    removed = set()
    for node in ast.walk(ast.parse(source)):
        internal_import = isinstance(node, ast.ImportFrom) and node.module in internal
        cli_guard = (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                     and isinstance(node.test.left, ast.Name)
                     and node.test.left.id == '__name__')
        if internal_import or cli_guard:
            removed.update(range(node.lineno - 1, node.end_lineno))
    return ''.join(line for index, line in enumerate(lines) if index not in removed)


def build():
    path = ROOT / 'kaggle/cost_aware_reasoning_benchmark.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    def cell(source, kind='code'):
        value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        return value
    install = '''# Enable Internet and select a GPU in Kaggle settings before running.
!nvidia-smi
!apt-get update -y && apt-get install -y zstd pciutils
!curl -fsSL https://ollama.com/install.sh | sh
!pip install ollama scikit-learn pandas matplotlib -q
import subprocess, urllib.request, time
try:
    urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2).close()
except Exception:
    ollama_log = open('ollama_server.log', 'a')
    ollama_process = subprocess.Popen(['ollama', 'serve'], stdout=ollama_log, stderr=ollama_log)
    for attempt in range(30):
        try:
            urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2).close()
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('Ollama failed to start; inspect ollama_server.log')
!ollama pull qwen2.5:7b
'''
    sections = [
        ('benchmark_engine', 'Shared solver and validation'),
        ('tos_search', 'Thought of Search: model synthesis and local BFS'),
        ('benchmark_runtime', 'Ollama client'),
        ('problems', 'Benchmark problems'),
        ('benchmark_data', 'Official GSM8K and SVAMP: fixed development and evaluation subsets'),
        ('plot_results', 'Metrics and charts'),
        ('run_benchmark', 'Benchmark runner'),
    ]
    source_cells = []
    for name, title in sections:
        source_cells.append(cell('## ' + title + '\n', 'markdown'))
        source_cell = cell(readable_source(name))
        source_cell['metadata']['source_module'] = name
        source_cells.append(source_cell)
    run = '''from datetime import datetime
PROFILE = 'pilot'  # 'pilot': 20 development questions; 'evaluation': 100 GSM8K + 50 SVAMP.
SEEDS = [11] if PROFILE != 'evaluation' else [11, 23, 37]
# None runs all six methods. Use [METHODS[5]] to run only the new ToS baseline.
METHODS_TO_RUN = None
OUTPUT = 'results_v4_' + datetime.now().strftime('%Y%m%d_%H%M%S')
subsets, data_provenance = prepare_benchmarks(cache_dir='benchmark_data_cache')
selected_problems = subsets[PROFILE]
print('Planned solver runs:', len(selected_problems)*len(SEEDS)*len(METHODS_TO_RUN or METHODS))
all_results = run_suite(ollama_backend, OUTPUT, seeds=SEEDS, methods=METHODS_TO_RUN,
                        problems=selected_problems, dataset_provenance=data_provenance, profile=PROFILE)
'''
    package = '''import shutil
archive = shutil.make_archive(OUTPUT, 'zip', OUTPUT)
print('Download:', archive)
'''
    notebook['cells'] = [cell('# Cost-Aware Reasoning Benchmark — v3\n\n'
        'Shared engine with structured output, bounded retries, arithmetic checks, terminal review, '
        'full traces and three seeded runs. See manifest.json for comparison policy. '
        'Official dataset profiles: pilot defaults to 20 GSM8K training questions and one seed. '
        'Switch PROFILE to evaluation for 100 GSM8K test + 50 SVAMP questions and three seeds. '
        'Dataset selection uses a separate fixed seed; IDs and source hashes are exported. '
        'Includes a Thought of Search arithmetic adaptation: generate search components '
        'up front, then execute local BFS without LLM calls during search. '
        'This per-problem restricted-code adaptation omits the original human validation '
        'and domain-level reuse. [Paper](https://arxiv.org/abs/2404.11833). '
        'Results from earlier implementations must be reported separately.\n', 'markdown'),
        cell(install), cell("Model = 'qwen2.5:7b'\n"), *source_cells,
        cell('## Run the benchmark\n', 'markdown'), cell(run),
        cell('## Download results\n', 'markdown'), cell(package)]
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    build()
