"""Build the standalone notebook; leave the original benchmark unchanged."""
import ast
import hashlib
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sync_notebook import readable_source


INTRO = """# Word-bank approach: cost-aware reasoning comparison

**Implementation revision: word-bank-search-3.** This notebook compares tree search
methods with Standard CoT. Keep its results separate from earlier 16-method pilots.

**Model: Qwen 2.5:7B through Ollama. Upload this notebook alone to Kaggle.**
Enable **Internet** and a **GPU accelerator**, then Run All. No repository upload is needed.
The default pilot runs 5 development questions × 9 methods × 1 seed = **45 solver runs**.
Search methods can make many calls; this is not a 45-call experiment.

This is a separate experiment. It does not replace the original notebook or its results.
All code is ordinary readable Python, with no escaped source blobs or generated-code `exec`.
First-time setup downloads Ollama, Qwen, and the public datasets. The default pilot is a
smoke test, not sufficient evidence for a research conclusion.

## What does the word bank mean here?

We select 200 GSM8K **training** examples, excluding development questions, all exact
GSM8K test matches, and all exact SVAMP matches. Their official worked solutions supply
positive reference steps. A frozen TF-IDF vectorizer represents words and two-word phrases.
For a new question, retrieve its three closest training questions locally; a candidate step's
bank score is its maximum cosine similarity to the reference steps for those questions.
The bank is never inserted wholesale into the LLM prompt. No API call is needed to score it.

This is a **reference-derived phrase bank**, not a classifier trained on human preferences.
There are no negative step labels. Similar wording is not proof of useful progress or correctness.
The exported global phrase table helps inspection; ranking uses the context-specific subset.
Numbers are removed from lexical features. Full training solutions are used locally to
derive phrase scores; they are not placed in LLM prompts. Evaluation answers never enter solver prompts or ranking.
Exact-match separation cannot rule out paraphrases or the model's pretraining exposure.

## Implemented comparisons

| Method | What changes | Main cost or limitation |
|---|---|---|
| Standard CoT | One full solution, up to 4096 output tokens | Strong baseline; no terminal reviewer |
| Tree-of-Thoughts (LLM-Judge) | Beam search, LLM scores candidates | Judge calls count in total tokens |
| Cost-Aware Priority Search | TF-IDF + predicted next-call token cost | Predicted cost is approximate |
| Quality-only ablation | Same search with cost weight zero | Isolates the token cost term |
| Thought of Search adaptation | Generate restricted search code, run local BFS, audit semantics | Extra audit calls count toward cost |
| Local Search: No Lexical Features | Arithmetic gate, novelty proxy, estimated next-call cost | Control for new search implementation |
| Local Search: Question TF-IDF | Add candidate–question similarity | Tests ordinary lexical relevance |
| Local Search: TF-IDF + Word Bank | Add similarity to relevant training reference steps | Main word-bank treatment |
| Local Search: Word Bank + Compact State | Send labels/equations instead of full step objects | May discard useful explanations |

All four new local-search variants share candidate generation, beam/depth limits and a
counted terminal LLM review. The last variant changes history serialization only.
Rejected terminal candidates receive one counted independent appeal and bounded recovery
feedback under the same budgets. Identical rejected terminal paths use a cached decision.
Branches with no usable candidates also receive bounded repair feedback. A verified partial
calculation is never automatically promoted to a final answer; completion and review are required.

## Search heuristic and fair interpretation

Invalid arithmetic is rejected. For surviving candidates:

`utility = 0.2 + 0.5*novel_equation - 0.1*repetition + 0.15*question_tfidf + 0.15*bank_score`

`priority = utility / estimated_next_call_tokens`

The appropriate lexical terms are disabled for ablations. Novel arithmetic is a weak
progress proxy: it does not establish that a required subgoal was solved. Token cost uses
estimated next-prompt size plus observed mean generation length, not estimated total cost
to a correct answer. The fixed weights are initial experimental settings, not learned optima.
Tune only on development data and freeze choices before evaluation.

Ranking generated candidates saves only potential **future** expansions. It does not refund
their generation tokens. All generation, repairs, judging, reviews and failed attempts are logged.
The shared per-problem budget uses approximate input-token preflight and actual returned usage.
It is **not a strict billing cap**; overshoots and requests with missing usage are reported.
If any request has unavailable/invalid usage, the affected method's aggregate cost is left
undefined rather than presenting partial usage as token savings. Observed usage is retained separately.
Existing baselines use this notebook's shared limits, so do not pool with older unrestricted runs.

## Primary references

- [Thought of Search](https://arxiv.org/abs/2404.11833): generated search components; our adaptation
  has no human semantic validation or domain-level cost amortization.
- [ChemReasoner](https://proceedings.mlr.press/v235/sprueill24a.html): professor-provided motivation
  for search with external feedback; not reproduced as an arithmetic baseline here.
- [LLMLingua](https://arxiv.org/abs/2310.05736): compression motivation; compact state here is deterministic,
  not the LLMLingua algorithm.
- [GSM8K](https://github.com/openai/grade-school-math) and
  [SVAMP](https://github.com/arkilpatel/SVAMP): checksum-verified official source snapshots.
"""

SETUP = """# Kaggle: Internet ON; GPU selected. Run this once per new session.
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
if not WORK_DIR.is_dir():
    raise RuntimeError('This setup cell is intended for Kaggle /kaggle/working.')
os.chdir(WORK_DIR)
subprocess.run(['nvidia-smi'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'ollama', 'scikit-learn', 'pandas', 'matplotlib'], check=True)
if shutil.which('ollama') is None:
    subprocess.run(['apt-get', 'update', '-y'], check=True)
    subprocess.run(['apt-get', 'install', '-y', 'zstd', 'pciutils'], check=True)
    installer = WORK_DIR / 'install_ollama_word_bank.sh'
    urllib.request.urlretrieve('https://ollama.com/install.sh', installer)
    subprocess.run(['sh', str(installer)], check=True)

def server_ready():
    try:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2):
            return True
    except Exception:
        return False

if not server_ready():
    server_log = open(WORK_DIR / 'word_bank_ollama.log', 'a')
    server_process = subprocess.Popen(['ollama', 'serve'], stdout=server_log, stderr=server_log)
    for attempt in range(60):
        if server_ready():
            break
        time.sleep(1)
    else:
        raise RuntimeError('Ollama failed to start; inspect word_bank_ollama.log.')
subprocess.run(['ollama', 'pull', 'qwen2.5:7b'], check=True)
print('Setup ready. Model download is not benchmark inference cost.')
"""

TRANSPORT = """# Actual Ollama usage; no application response cache and no silent API retries.
Model = 'qwen2.5:7b'

def ollama_backend(prompt, schema, temperature, seed, max_tokens):
    import ollama
    client = ollama.Client(host='http://127.0.0.1:11434', timeout=180)
    kwargs = dict(
        model=Model, messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': temperature, 'seed': seed,
                 'num_predict': max_tokens, 'num_ctx': 8192},
    )
    if schema is not None:
        kwargs['format'] = schema
    response = client.chat(**kwargs)
    return {'text': response['message']['content'],
            'prompt_tokens': response.get('prompt_eval_count'),
            'completion_tokens': response.get('eval_count'),
            'done_reason': response.get('done_reason', 'unknown')}
"""

CONFIGURATION = """# Edit this cell before a run. Defaults keep the first diagnostic manageable.
PROFILE = 'pilot'             # 'pilot', 'development', or 'evaluation'
LIMIT_PER_DATASET = 5          # None uses the full selected profile
SEEDS = [11]                  # For a final run, consider [11, 23, 37]
METHODS_TO_RUN = list(ALL_METHODS)
BANK_TRAINING_SIZE = 200
TOKEN_BUDGET_PER_PROBLEM = 16000
SEARCH_CONFIG = Config(max_calls=30, max_expansions=16, retries=1,
                       max_depth=10, beam=3, candidates=2,
                       audit_tos=True, review_appeal=True)

# To focus on the four local-search ablations:
# METHODS_TO_RUN = [METHODS[0], *NEW_METHODS]

# Optional hypothetical API prices in USD per million tokens. Local Ollama has no API invoice.
INPUT_USD_PER_MILLION = None
OUTPUT_USD_PER_MILLION = None

# Full evaluation: PROFILE='evaluation', LIMIT_PER_DATASET=None, SEEDS=[11, 23, 37].
# With all 9 methods: 150 questions x 3 seeds x 9 methods = 4050 solver runs.
# Freeze methods, weights and budgets BEFORE inspecting evaluation outcomes.
"""

DATA = """CACHE_DIR = WORK_DIR / 'word_bank_data_cache'
subsets, data_provenance = prepare_benchmarks(cache_dir=CACHE_DIR)
if PROFILE not in subsets:
    raise ValueError('Choose pilot, development or evaluation')
if LIMIT_PER_DATASET is not None and (type(LIMIT_PER_DATASET) is not int or LIMIT_PER_DATASET < 1):
    raise ValueError('LIMIT_PER_DATASET must be a positive integer or None')
if type(BANK_TRAINING_SIZE) is not int or BANK_TRAINING_SIZE < 2:
    raise ValueError('BANK_TRAINING_SIZE must be an integer of at least two')
selected_problems = []
counts = Counter()
for problem in subsets[PROFILE]:
    dataset = problem['dataset']
    if LIMIT_PER_DATASET is None or counts[dataset] < LIMIT_PER_DATASET:
        selected_problems.append(problem)
        counts[dataset] += 1

bank_started = time.perf_counter()
word_bank = prepare_word_bank(CACHE_DIR, subsets, BANK_TRAINING_SIZE)
bank_build_seconds = time.perf_counter() - bank_started
print('Selected:', dict(counts), '| training-bank examples:', len(word_bank.records))
print('Planned solver runs:', len(selected_problems) * len(SEEDS) * len(METHODS_TO_RUN))
print('Bank construction seconds:', round(bank_build_seconds, 3), '| LLM calls: 0')
print('Methods:')
for index, method in enumerate(METHODS_TO_RUN, 1):
    print(index, method)
"""

RUN = """from datetime import datetime, timezone
from uuid import uuid4
import importlib.metadata
from IPython.display import display

OUTPUT = WORK_DIR / ('word_bank_results_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                     + '_' + uuid4().hex[:6])
runtime_provenance = dict(data_provenance)
runtime_provenance.update(
    source_hashes=EMBEDDED_SOURCE_HASHES, bank_build_seconds=bank_build_seconds,
    package_versions={name: importlib.metadata.version(name) for name in
                      ('ollama', 'scikit-learn', 'numpy', 'pandas', 'matplotlib')},
)
# Capture installed model identity so mutable model tags do not obscure reproducibility.
import ollama
runtime_provenance['ollama_models'] = ollama.list().model_dump(mode='json')
print('Results will be written after each solver run to:', OUTPUT)
all_results, summary = run_word_bank_suite(
    ollama_backend, selected_problems, word_bank, OUTPUT, SEEDS, METHODS_TO_RUN,
    TOKEN_BUDGET_PER_PROBLEM, runtime_provenance, PROFILE, SEARCH_CONFIG,
    INPUT_USD_PER_MILLION, OUTPUT_USD_PER_MILLION,
)
display(summary)
"""

ANALYSIS = """# Isolate one change at a time. Positive accuracy delta is better;
# negative token delta is cheaper. No significance claim from a small pilot.
from IPython.display import Image, display
pairs = [
    (NEW_METHODS[0], NEW_METHODS[1], 'Add question TF-IDF'),
    (NEW_METHODS[1], NEW_METHODS[2], 'Add reference word bank'),
    (NEW_METHODS[2], NEW_METHODS[3], 'Compact the reasoning state'),
    (METHODS[4], METHODS[3], 'Add predicted next-call token penalty'),
    (METHODS[2], METHODS[3], 'Replace LLM step judge with local ranking'),
]
differences = []
for dataset in summary['dataset'].unique():
    table = summary[summary['dataset'] == dataset].set_index('method')
    for control, treatment, change in pairs:
        if control in table.index and treatment in table.index:
            a, b = table.loc[control], table.loc[treatment]
            differences.append({
                'dataset': dataset, 'change': change,
                'control': control, 'treatment': treatment,
                'accuracy_delta_percentage_points': b['accuracy_pct'] - a['accuracy_pct'],
                'mean_token_delta': b['mean_tokens'] - a['mean_tokens'],
                'mean_call_delta': b['mean_calls'] - a['mean_calls'],
            })
difference_table = pd.DataFrame(differences, columns=[
    'dataset', 'change', 'control', 'treatment', 'accuracy_delta_percentage_points',
    'mean_token_delta', 'mean_call_delta',
])
difference_table.to_csv(OUTPUT / 'ablation_differences.csv', index=False)
display(difference_table)
for chart in sorted(OUTPUT.glob('*_accuracy_vs_tokens.png')):
    display(Image(filename=str(chart)))
"""

PACKAGE = """# Also works after an interrupted benchmark, once OUTPUT was created.
import shutil
from IPython.display import FileLink, display

if not OUTPUT.exists():
    raise RuntimeError('No results directory exists yet.')
archive = shutil.make_archive(str(OUTPUT), 'zip', str(OUTPUT))
print('Download this archive from the link or Kaggle Output panel:')
display(FileLink(Path(archive).name))
"""


def cell(source, kind='code', **metadata):
    value = dict(cell_type=kind, metadata=metadata, source=source.splitlines(keepends=True))
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def experiment_source():
    source = (ROOT / 'kaggle/word_bank_experiment.py').read_text(encoding='utf-8')
    removed = set()
    unused_functions = {'concise_answer', 'solution_checks', 'execute_equations',
                        'program_answer', 'consistency_answer'}
    unused_assignments = {'EQUATIONS_SCHEMA'}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module in ('benchmark_engine', 'benchmark_data'):
            removed.update(range(node.lineno - 1, node.end_lineno))
        if isinstance(node, ast.FunctionDef) and node.name in unused_functions:
            removed.update(range(node.lineno - 1, node.end_lineno))
        if isinstance(node, ast.FunctionDef) and node.name == 'example':
            removed.update(range(node.lineno - 1, node.end_lineno))
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id in unused_assignments for target in node.targets):
            removed.update(range(node.lineno - 1, node.end_lineno))
    return ''.join(line for i, line in enumerate(source.splitlines(keepends=True)) if i not in removed)


def split_source(source, module):
    """Split top-level definitions into readable cells without changing executable text."""
    lines = source.splitlines(keepends=True)
    nodes = ast.parse(source).body
    starts = [0]
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            if start > starts[-1] and start - starts[-1] > 15:
                starts.append(start)
    starts.append(len(lines))
    return [cell(''.join(lines[a:b]), source_module=module)
            for a, b in zip(starts, starts[1:]) if ''.join(lines[a:b]).strip()]


def build():
    sources = {name: readable_source(name) for name in
               ('benchmark_engine', 'tos_search', 'benchmark_data')}
    sources['word_bank_experiment'] = experiment_source()
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    cells = [cell(INTRO, 'markdown'), cell('## 1. Kaggle setup\n', 'markdown'),
             cell(SETUP, role='setup'), cell(TRANSPORT, role='transport')]
    for name, source in sources.items():
        cells.append(cell('## Embedded source: ' + name + '\n', 'markdown'))
        cells.extend(split_source(source, name))
    cells += [cell('EMBEDDED_SOURCE_HASHES = ' + json.dumps(hashes, indent=4) + '\n'),
              cell('## 2. Experiment settings\n', 'markdown'), cell(CONFIGURATION, role='configuration'),
              cell('## 3. Prepare disjoint datasets and training phrase bank\n', 'markdown'),
              cell(DATA, role='data'), cell('## 4. Run all selected approaches\n', 'markdown'),
              cell(RUN, role='run'), cell('## 5. Compare differences\n', 'markdown'),
              cell(ANALYSIS, role='analysis'),
              cell(textwrap.dedent('''\
                  ## Reading the results

                  Check accuracy, missing answers and wrong answers together. Compare total **input + output**
                  tokens; fewer calls do not necessarily mean lower cost. `tokens_per_correct` divides all
                  spent tokens, including failures, by the number of correct answers. Zero correct answers
                  produce an undefined value. Each dataset is reported separately.

                  Seeds repeat questions and are not independent new test questions. Use a larger held-out
                  run before drawing conclusions; uncertainty analysis should resample question IDs, keeping
                  methods paired and seeds grouped. This notebook does not claim statistical significance.

                  A word-bank improvement means the same search configuration solves as many or more
                  questions for fewer tokens, or improves accuracy at a comparable budget. More positive
                  similarity scores alone are not an improvement. Inspect full traces for counterexamples.

                  Artifacts: `comparison_summary.csv`, `ablation_differences.csv`, `per_problem_metrics.csv`,
                  `per_seed_metrics.csv`, `termination_counts.csv`, charts, `full_results.jsonl`, manifest,
                  `missing_diagnostics.csv` (specific failure reasons and rejection counts),
                  selected questions, training examples, fitted vocabulary/IDF, and `word_bank_phrases.csv`.
                  Runtime includes local scoring; bank build time is recorded separately. Training references
                  have no LLM construction cost here, but curating real preference labels would have a cost.
                  Monetary values are hypothetical only when both prices were supplied.
                  '''), 'markdown'),
              cell('## 6. Download results\n', 'markdown'), cell(PACKAGE, role='package')]
    for index, entry in enumerate(cells):
        entry['id'] = f'word-bank-{index:03d}'
        if entry['cell_type'] == 'code':
            ast.parse(''.join(entry['source']))
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True, 'language': 'python',
                   'sourceType': 'notebook'},
    })
    path = ROOT / 'kaggle/Word_bank_approch.ipynb'
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Built:', path, '| cells:', len(cells))
    return path


if __name__ == '__main__':
    build()
