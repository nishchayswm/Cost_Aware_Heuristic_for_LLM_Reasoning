# Final project: cost-aware heuristic search for LLM reasoning

**Self-contained project. The v6 changes have offline tests and a separate live development evaluation.**
Your earlier pilot results remain untouched in `results/gsm8k100`.
Offline tests use synthetic fixtures and mocked model responses only. No real
benchmark scores are supplied or implied.

The main method uses incremental Tree-of-Thoughts-style node generation and
**bounded best-first search** guided by five factors: reasoning quality, success
confidence, depth, estimated additional token cost, and search diversity.
This is the corrected state-selection project, not the earlier Repair-or-Explore
action-selection experiment. It does not import code or results from outside this
folder when running.

## V6: live development evaluation

See [CHANGES_V6.md](CHANGES_V6.md) for the current formula and shared search fixes.
The proposed ranking now discounts depth using success confidence, gates diversity
by sqrt(Q*P), and includes estimated judging cost. All search methods share rubric
consistency adjustment, continuation of sound-but-incomplete answers, and a final
selector using sqrt(Q*P). Quality-only best-first isolates the state-ranking change.
All search methods also receive a budgeted full-solution starting candidate and
alternative sibling prompts. All methods generate numeric-only expressions;
step IDs remain internal bookkeeping. The weights and global budgets remain unchanged.

The current development run uses 10 fixed training-split questions with all six
methods, stored in `results/dev_v6_10`. Original results remain intact. Never mix
results produced by different code/config versions. This small development run
cannot establish SOTA or a reliable accuracy gain on unseen tasks.

## Game of 24

See [GAME24.md](GAME24.md) for the new task, eight-method comparison, exact
verification, local dataset preparation and run commands. Previous Game of 24 results are preserved; this corrected revision has only offline tests. Use `python -m costbench.game24`, not the GSM8K runner.

## 1. Requirements

- Windows PowerShell, Python **3.11 or newer**.
- Ollama running locally at `http://localhost:11434`.
- Default model: **`qwen2.5:7b-instruct`**, already found installed on your PC.
- RTX 4060 8 GB VRAM / 16 GB RAM: one quantized model is shared sequentially by all
  methods. The configured context is 8192 tokens.
- **No pip packages, paid API, remote judge or embedding model required.**

All commands below must be run from **final_proj**, not its parent project:

```powershell
cd "C:\Users\dhruv\OneDrive\Documents\COST\final_proj"
python --version
python -m costbench doctor
```

`doctor` checks the installed model and server without generating an answer. If
Ollama is not running, start its desktop application. If the model is missing:

```powershell
ollama pull qwen2.5:7b-instruct
```

## 2. Run the experiment yourself

Start with a **development** question to check your runtime and formatting:

```powershell
python -u -m costbench run --dataset data/gsm8k_dev_100.jsonl --limit 1 --output results/dev_check_v3
```

Run all **100 fixed GSM8K test questions**:

```powershell
python -u -m costbench run --output results/gsm8k100_v6
```

Optional PowerShell wrapper for the same full run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_benchmark.ps1
```

These commands **start real model inference**. They have not been executed here.
Keep the laptop plugged in and awake. Depending on generation speed and search
depth, the complete experiment can take hours. Estimate duration using development
questions rather than repeatedly inspecting test-set performance.

For a different model or seed, use a fresh output directory:

```powershell
python -u -m costbench run --seed 43 --output results/gsm8k100_v6_seed43
python -u -m costbench run --model qwen2.5:3b --output results/gsm8k100_v6_3b
```

## 3. Which approaches run?

There are **six methods**: the proposed algorithm, four previously discussed
baselines, and the additional quality-only control needed to isolate the heuristic.

| Terminal / CSV name | Method |
|---|---|
| `heuristic_bestfirst` | Your five-factor best-first search |
| `cot` | Single-path Chain-of-Thought |
| `self_consistency` | Six independent complete solutions and plurality voting |
| `tot_bfs` | ToT-style bounded breadth/beam search, greedy quality selection |
| `self_eval_beam` | Stochastic beam guided by prompted self-evaluation scores |
| `quality_bestfirst` | Same best-first engine, ranking by quality alone |

All methods use the same structured reasoning format and restricted arithmetic
checker. They are **calculator-assisted local adaptations**. In particular,
`self_eval_beam` is not an exact reproduction of the published probability-based
Self-Evaluation Guided Beam Search objective. No method is advertised as a verified
2026 SOTA reproduction. See [METHODOLOGY.md](METHODOLOGY.md).

## 4. Terminal output: all methods on each question

Execution is question-major:

```text
QUESTION 1/100: <question ID>
<question text>
Ground truth (evaluator only): <gold>

  heuristic_bestfirst  CORRECT/WRONG answer=... gold=... calls=... tokens=... seconds=...
  cot                 CORRECT/WRONG answer=... gold=... calls=... tokens=... seconds=...
  ...remaining methods...

Method                  Prediction      Original  Reviewed  Calls   Tokens    Seconds
heuristic_bestfirst      ...             ...       ...     ...       ...
cot                     ...             ...       ...     ...       ...
self_consistency        ...             ...       ...     ...       ...
tot_bfs                 ...             ...       ...     ...       ...
self_eval_beam          ...             ...       ...     ...       ...
quality_bestfirst       ...             ...       ...     ...       ...

QUESTION 2/100: ...
```

This is an **output template**, not experimental results. All six finish on one
question before the next starts. Method order rotates by question to reduce
systematic timing/cache-order bias. The summary table uses a fixed order. Methods
run sequentially, not as six simultaneous GPU workloads. Answers are not shared
between methods. The gold value is printed by the evaluator, never sent to the LLM.

## 5. Stored results and comparison

The full run creates files in `final_proj/results/gsm8k100_v6/`:

| File | Purpose |
|---|---|
| `per_question.csv` | **One row per question/seed; every method's prediction, correctness, calls, tokens, time and search statistics side by side** |
| `summary.csv` | Overall accuracy, Wilson interval, mean calls/tokens/time, expansion and failure statistics |
| `paired_comparisons.csv` | Proposed method minus each comparator, paired accuracy differences and question-bootstrap intervals |
| `report.html` | Browser dashboard: summary, accuracy/token scatter plot and individual-question table |
| `report.md` | Human-readable summary |
| `results_long.csv` | One row per method/question |
| `results.jsonl` | Append-only machine-readable results; **600 rows** when one seed finishes all 100 questions |
| `traces/*.json` | Nodes, parent IDs, evaluations, five-factor features, expansion order, duplicate/pruning events |
| `calls/*.jsonl` | Prompts, model replies, seeds, measured tokens, timings and backend errors |
| `manifest.json` | Code/data hashes, model digest, frozen config, selected question IDs |
| `warmups.jsonl` | Model warmup recorded separately from method costs |

Reports refresh after every saved method. Aggregate comparisons use only question
blocks containing **all six methods**, so partial runs do not compare unequal
question sets. The per-question table shows pending entries. Refresh the HTML page
manually to see updates. Files are local and contain the questions/model responses.

To regenerate reports without calling the LLM:

```powershell
python -m costbench report --output results/gsm8k100_v6
```

## 6. Pause, resume and backend errors

From a second PowerShell window opened in `final_proj`:

```powershell
python -m costbench status --output results/gsm8k100_v6
python -m costbench pause --output results/gsm8k100_v6
```

`pause` waits until the current method finishes and saves its result. Then rerun
the **same run command** to resume. Completed methods are skipped, and successful
calls from an interrupted method are replayed without another model request. Their
original cost remains counted. Do not change code/config/model/data within a run;
the manifest rejects mixed experiments. Use a new output folder after any change.

Ctrl+C also stops the runner, but an in-flight request might have consumed tokens
that were not returned/logged. Prefer the clean pause command for cost studies.
After a forced process termination, a stale `run.lock` may remain. Remove it only
after confirming that the runner process has stopped. Never run two processes
into the same output folder.

Backend errors stop the run rather than silently counting infrastructure problems
as incorrect answers. If Ollama reports a token-repeat error, timeout, or HTTP 500,
inspect the call log, check that the server is healthy, and explicitly retry:

```powershell
python -u -m costbench run --output results/gsm8k100_v6 --retry-failed
```

Successful saved calls are reused; the failed request is retried. Because its token
cost may be unknown, the affected result retains `unknown_cost=true` and a failed
attempt count. Its reported costs are lower bounds; do not make precise efficiency
claims from such runs. The dashboard warns when this occurs. Automatic unbounded
retries are intentionally absent.

## 7. Configuration and ablations

`configs/default.json` defines the model, search limits and five heuristic weights:

```text
quality 0.35, success 0.30, depth 0.05, cost 0.15, diversity 0.15
```

These are **provisional starting weights**, not optimized settings. Tune only on
the separate development data. All methods share ceilings of 32 calls, 6144 output
tokens and 30000 total tokens per question. Search also has depth 8, branch factor
2, beam width 2, frontier size 12 and at most 12 expanded states. Algorithms can
finish before those limits. Equal ceilings do not mean equal realized compute.

For an ablation, copy the config, set one weight to zero, renormalize the remaining
weights to sum to one, and run with `--config` and a fresh output directory:

```powershell
python -u -m costbench run --config configs/my_ablation.json --dataset data/gsm8k_dev_100.jsonl --output results/dev_ablation
```

Do not create weight settings based on test-set winners. The quality-only control
is already included in every default run.

## 8. Offline checks (no model, no GSM8K inference)

```powershell
python -m unittest discover -s tests -v
```

The tests mock the backend and use tiny synthetic questions. Scratch files are
created under `final_proj/tests/_tmp/` and cleaned automatically. They verify
scheduling, gold isolation, safe arithmetic, dynamic diversity, scoring, budgets,
replay, explicit failure recovery and reports. Passing them does not establish
model accuracy or runtime compatibility; the development run is your next check.

## 9. Dataset

Two fixed files are bundled from the earlier data preparation:

- `data/gsm8k_test_100.jsonl`: 100 test-split questions, sampling seed 42.
- `data/gsm8k_dev_100.jsonl`: 100 questions from the separate training split.
- `data/manifest.json`: official source URLs, source hashes and sampled indices.

No earlier predictions/results are copied. Data preparation can be reproduced
with `python scripts/prepare_data.py` (network needed). Existing manifest hashes
protect a run from silently changing data. Public GSM8K may overlap pretraining;
100 questions support a pilot, not a general SOTA claim.
