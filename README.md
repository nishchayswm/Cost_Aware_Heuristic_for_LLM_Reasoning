# Cost-Aware Heuristic for LLM Reasoning

LLMs solves problems by generating a sequence of reasoning steps. However, a mistake in 
an early step can lead to an incorrect final answer. Searching through multiple possible 
reasoning paths allows the model to recover from such mistakes. 
This search has a cost. Each request to generate a new step, judge a candidate, repair an 
answer uses tokens. A method that explores multiple paths may improve accuracy but it 
will use more computation. The problem is how to choose useful paths without spending 
too much on search. 
Our project studies a heuristic that chooses the next reasoning state using five factors: 
current quality, estimated chance of success, depth, expected token cost, and diversity. A 
reasoning state is a partial solution, such as a sequence of arithmetic steps or a partly filled 
crossword. The heuristic assigns a priority score to each available state and expands the 
state with the highest score. 

We evaluate the approach on Game24, mini crosswords(10 words), and a small GSM8K 
problemset. We measure correctness, model calls, and total input and output tokens. 
Tokens serve as a measure of inference cost. 
The strongest result comes from Game24. On 100 generated puzzles evaluated with three 
random seeds, our method used 42.14% fewer tokens than our Tree of Thoughts baseline, 
with success rates of 82.33% and 80.00%, respectively. On a separate set of 100 
original-ToT puzzles, it used 44.39% fewer tokens while solving three fewer puzzles. These 
results show a useful cost reduction in this task, with accuracy depending on the dataset.

## Quick start

Run the commands below from the `final_proj` directory in PowerShell. You need **Python 3.11 or newer**. The benchmark uses the Python standard library, and the generated Game of 24 and GSM8K datasets are included.

### Check the environment

The offline tests use synthetic questions and mocked model responses. For a live run, start Ollama, download the default model if needed, and check the connection. The default endpoint is `http://localhost:11434`.

```powershell
python --version
python -m unittest discover -s tests -v
ollama pull qwen2.5:7b-instruct
python -m costbench doctor
```

### Run the benchmarks

Choose the command for the task and split you want to evaluate. Each output directory holds one configuration, dataset selection and seed.

```powershell

# Game of 24: 10 development puzzles, or all 100 generated test puzzles
python -u -m costbench.game24 run --dataset data/game24/game24_dev.jsonl --limit 10 --output results/game24_dev10_exp64
python -u -m costbench.game24 run --seed 42 --output results/game24_test100_exp64_seed42
```

The Game of 24 commands use the report's **64-expansion limit** by default. To run all three reported seeds on the same 100 generated test puzzles:

```powershell
foreach ($seed in 42, 43, 44) {
    python -u -m costbench.game24 run --seed $seed --output "results/game24_test100_exp64_seed$seed"
    if ($LASTEXITCODE -ne 0) { break }
}
```

The GSM8K development command selects the first 30 bundled questions. The report gives the sample size but does not list their IDs. The original-ToT puzzle set and the crossword runner and dataset are outside this checkout.

### Pause, resume and regenerate reports

From a second PowerShell window, use `status` to inspect a GSM8K run or `pause` to stop either runner after the current method saves. Repeat the original run command to resume. Completed results are skipped, and replayed model calls retain their recorded costs.

```powershell
python -m costbench status --output results/game24_test100_exp64_seed42
python -m costbench pause --output results/game24_test100_exp64_seed42
python -m costbench report --output results/game24_test100_exp64_seed42
python -m costbench.game24 report --output results/game24_test100_exp64_seed42
```

The two report commands read saved results without contacting the model. If a backend request fails, restore the Ollama connection and repeat the run command with `--retry-failed`. An affected result carries `unknown_cost=true` because its recorded tokens are a lower bound. A changed model, configuration, code version or question selection requires a fresh output directory.

## Research question and approach

A correct intermediate step can be expensive to extend, and two highly rated steps can lead through nearly identical calculations. Our search maintains a *frontier* of partial solutions and ranks them by the value of expanding each one next. Early designs scored quality and word overlap with the question. That could reward a step for repeating the prompt or penalize a correct paraphrase. The final rule instead measures novelty against reasoning states the search has already considered and estimates the cost of the next expansion.

### Priority function

For a state `s`, the engine expands the candidate with the highest priority.

```text
H(s) = 0.35 Q(s) + 0.30 P(s) − 0.05 D(s) − 0.15 C(s) + 0.15 V(s)

D(s) = min(1, depth / maximum_depth) × (1 − P(s))
C(s) = min(1, estimated_next_tokens / reference_tokens)
V(s) = novelty(s) × √(Q(s) × P(s))
```

| Factor | Interpretation |
|---|---|
| **Q · Quality** | The model judges whether the reasoning so far is sound and useful. |
| **P · Success estimate** | The model estimates whether the state can reach a correct answer within the remaining budget. These scores are uncalibrated. |
| **D · Depth penalty** | A deep path incurs a smaller penalty when the model still expects it to succeed. |
| **C · Expected token cost** | The engine estimates the next round of work. Evaluation uses recorded input and output tokens. |
| **V · Diversity** | A new direction earns a larger bonus when both its quality and its estimated chance of success are high. |

The report describes exploring 15 weight settings on ten development problems. The coefficients above are fixed design choices. Comparing the heuristic with **quality-only best-first search** tests their combined effect while keeping the search engine and task checks in common.

### How the search works on each task

**Game of 24.** A state holds the remaining rational values and the expressions that produced them. Python enumerates legal pairwise operations and merges equivalent number multisets. The model rates a shortlist of successors. Novelty measures multiset Jaccard distance from expanded states, so repeated values count. The cost estimate uses recent response length and the number of remaining terms, normalized by 4,000 tokens. At two values, all four search methods try the final operations locally. A solution must use each original number occurrence exactly once and evaluate exactly to 24. Up to two model calls per method and puzzle can probe uncertain states. The symbolic baseline searches independently and consumes CPU time without model calls.

**GSM8K.** Each search method first generates and judges a complete candidate, then explores alternative one-step continuations from the root. Python evaluates numeric expressions with exact arithmetic. The judge assesses interpretation, units, coverage and completion. Explicit rubric failures reduce Q and P, and a sound subtotal can be reopened if the question asks for a further calculation. Novelty uses TF-IDF cosine distance from expanded reasoning paths. Estimated cost includes prompt tokens, expected output and a share of the sibling evaluation call, normalized by 1,200 tokens. Search selects the evaluated completion with the highest `√(Q × P)`. The evaluator then compares its numeric answer with the reference label.

**Mini crosswords (report experiment).** States are partially filled grids. Word lengths and crossing letters constrain candidate entries. Novelty measures how many word slots differ from other frontier states. The cost estimate uses the number of unfilled slots and the average tokens per call. Word accuracy, letter accuracy and complete-grid success capture different degrees of progress.

### Baselines

| Method | Comparison |
|---|---|
| Chain of Thought (`cot`) | Generate one complete solution. |
| Self-Consistency (`self_consistency`) | On GSM8K, vote over six normalized numeric answers. On Game of 24, score success if any of six expressions verifies. |
| Tree of Thoughts (`tot_bfs` in this repository) | Use bounded breadth or beam search for the arithmetic tasks. The crossword experiment uses depth-first search. |
| Self-evaluation beam (`self_eval_beam`) | Select among candidate states stochastically using prompted model ratings. |
| Quality-only best-first (`quality_bestfirst`) | Expand the highest-Q state using the same task-specific engine as the five-factor method. |
| Verified sampling and symbolic search (Game of 24 only) | Stop sampling at the first verified expression, or solve the puzzle with deterministic arithmetic search. |

The five-factor method appears as `heuristic_bestfirst` in reports. All LLM comparisons use local adaptations of the baseline algorithms. On Game of 24, the shared legal-move generator and exact endgame checks also contribute to the difference between state search and full-expression sampling.

## Experimental results

The following tables give the aggregates reported in *Cost Aware Heuristic for LLM Reasoning*, §§4.2–4.5, by Nishchay Gupta, Dhruv Gupta and Abhishek Vekariya (10 November 2026). The experiments used Qwen2.5-7B through Ollama. **Mean tokens** counts input plus output per attempt, including generation, evaluation and recorded retries. Percentage savings use `100 × (1 − heuristic mean tokens / baseline mean tokens)`. The raw predictions and call logs for these tables are unavailable in this checkout.

### Generated Game of 24

Each method attempted the **same 100 unique puzzles at three seeds** (42, 43 and 44), giving 300 attempts per method. Five-factor search used **42.14% fewer tokens** than ToT BFS and solved 247 attempts versus 240. The report gives an approximate 95% interval of **37.08%–46.47%** for token savings. The experiment did not establish an accuracy advantage.

| Method | Solved / 300 | Success | Mean calls | Mean tokens |
|---|---:|---:|---:|---:|
| Chain of Thought | 14 | 4.67% | 1.00 | 238.66 |
| Self-Consistency (success@6) | 53 | 17.67% | 6.00 | 1,441.30 |
| ToT BFS | 240 | 80.00% | 10.85 | 6,814.72 |
| Self-evaluation beam | 238 | 79.33% | 11.50 | 7,043.84 |
| Quality-only best-first | 243 | 81.00% | 6.96 | 4,655.15 |
| **Five-factor best-first** | **247** | **82.33%** | **5.79** | **3,943.12** |

Against the quality-only control, five-factor search used **15.30% fewer tokens**. The deterministic arithmetic solver solved every puzzle with zero model calls.

### Original-ToT Game of 24

On a separate 100-puzzle set at seed 42, five-factor search used **44.39% fewer tokens** than ToT BFS and solved **three fewer puzzles**. A dataset audit found 19 starting-number multisets shared with the prepared generated splits. Across the other 81 puzzles, five-factor search solved 66 and ToT BFS solved 71.

| Method | Solved / 100 | Mean calls | Mean tokens |
|---|---:|---:|---:|
| ToT BFS | 81 | 12.28 | 7,757.32 |
| Quality-only best-first | 75 | 8.03 | 5,364.56 |
| **Five-factor best-first** | **78** | **6.24** | **4,313.82** |

### Mini crosswords

Both methods attempted the same ten puzzles under two budgets. Five-factor search improved word and letter accuracy at each budget. **Neither method completed a grid.** At the higher budget their mean token use differed by fewer than 360 tokens per puzzle.

| Token budget | Method | Word accuracy | Letter accuracy | Mean tokens |
|---|---|---:|---:|---:|
| 30,000 | ToT DFS | 3.0% | 19.2% | 28,195.2 |
| 30,000 | Five-factor best-first | 10.0% | 32.4% | 28,916.8 |
| 120,000 | ToT DFS | 6.0% | 22.0% | 114,485.2 |
| 120,000 | Five-factor best-first | 11.0% | 34.8% | 114,126.9 |

The lower budget allowed 60 calls and 30 expansions per puzzle. The higher budget allowed 240 calls and 120 expansions.

### GSM8K development pilot

All six methods answered **27 of 30 development questions** correctly. Five-factor search used **11.4% fewer tokens** than ToT BFS. A single chain of thought reached the same aggregate accuracy at about one tenth of the five-factor method's token cost.

| Method | Correct / 30 | Mean calls | Mean tokens |
|---|---:|---:|---:|
| Chain of Thought | 27 | 1.07 | 418.1 |
| Self-Consistency | 27 | 6.23 | 2,435.6 |
| ToT BFS | 27 | 9.00 | 4,610.4 |
| Self-evaluation beam | 27 | 8.77 | 4,424.8 |
| Quality-only best-first | 27 | 8.73 | 4,493.1 |
| **Five-factor best-first** | **27** | **8.10** | **4,085.7** |

### What the earlier GSM8K pilot revealed

Two earlier runs informed implementation changes. In the V2 pilot on 18 test questions, five-factor search scored 9/18 and returned no prediction eight times. All eight failures occurred at the root, where invalid assignments or references blocked the first usable step. CoT returned no prediction 16 times. These questions informed later revisions and count as development-exposed. [Bundled pilot statistics](analysis/v2_diagnosis.json)

A subsequent V6 run on ten development questions recorded 9/10 for both five-factor search and self-consistency, at 3,351 and 2,655 mean tokens respectively. CoT and quality-only search each scored 8/10. These figures describe a separate development run from the report's 30-question pilot. Neither pilot provides a held-out estimate for the full dataset.

## Data, configuration and outputs

| Resource | Contents |
|---|---|
| [GSM8K manifest](data/manifest.json) | Source hashes and sampled indices for 100 development questions from the official training split and 100 test questions. Both samples use seed 42. |
| [Reviewed labels](data/label_corrections.json) | An audited correction is scored separately from the original GSM8K label. The evaluator keeps both scores. |
| [Game of 24 manifest](data/game24/game24_manifest.json) | Three disjoint splits of 100 locally generated solvable puzzles. Permutations of a starting multiset stay in the same split. |
| [GSM8K configuration](configs/default.json) | Depth 8, branch factor 2, beam width 2, 12 expansions and a best-first frontier of 12. |
| [Game of 24 configuration](configs/game24.json) | Depth 3 and up to 64 expansions per model-guided search. Both task configurations allow up to 32 model calls, 6,144 output tokens and 30,000 total tokens per method and question. |
| [V6 linear control](configs/v6_linear_control.json) | Uses plain normalized depth and ungated novelty. Pass it with `--config` on the same development questions and seed to compare the two scoring rules. |

The runners rotate method order by question. Aggregate reports include only questions with results from every method (six for GSM8K, eight for Game of 24). For an individual run, open `report.html` for an overview, then use the files below to inspect a comparison.

| Run output | Use |
|---|---|
| `summary.csv`, `report.md` | Compare accuracy, mean model calls and recorded token costs. |
| `per_question.csv`, `results_long.csv`, `results.jsonl` | Inspect predictions and costs by question and method. |
| `calls/`, `traces/` | Inspect prompts, responses, state priorities, expansion order and stopping decisions. |
| `manifest.json` | Identify the configuration, model digest, code and data hashes, and selected question IDs. |
| `paired_comparisons.csv` (GSM8K) | Read paired accuracy differences and question-bootstrap intervals. GSM8K also records warmup costs separately in `warmups.jsonl`. |

For implementation details, see the [methodology](METHODOLOGY.md), [GSM8K search policies](costbench/methods.py), [Game of 24 engine](costbench/game24.py) and [budget and replay logic](costbench/backend.py).
