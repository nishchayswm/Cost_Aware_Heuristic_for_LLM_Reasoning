# Mini crosswords: 50 puzzles, two search policies

## Structured output repair (v2)

Use the regenerated notebook and restart the kernel after installing dependencies.
The v2 backend uses `lm-format-enforcer==0.11.3` to constrain JSON during decoding:
move slots follow h1-h5/v1-v5 and each word contains exactly five English letters.
Completed JSON cannot have trailing markdown or a fabricated conversation.
Local validation also checks numeric scores and complete, unique evaluation IDs.
Truncated or invalid responses have up to two retries within the same call/token
budgets; exhausted retries report `invalid_model_response` explicitly.

Run the new preflight cell before setting `RUN_BENCHMARK=True`. Its model calls
and costs are saved separately from the measured 50-puzzle experiment.
The project is restored under `/kaggle/working/crosswords50_schema_v2`, with
results in `results/schema_v2_seed42/`. Existing dataset snapshots and model pins
are copied from the original project when available. Old result files are preserved.
The export is `crosswords50_schema_v2_results.zip`. Restore this v2 project folder
when resuming a v2 run. Output constraints enforce format and length, not correct answers.
This repair has offline validation; a live Kaggle GPU run is still required.

Upload `Crosswords50_Heuristic_vs_ToT_Kaggle.ipynb` to Kaggle. It embeds all necessary source and offline tests. Enable GPU and Internet, run setup cells, then set `RUN_BENCHMARK=True` in the run cell. No Ollama or paid API is needed. Default model: Qwen/Qwen2.5-7B-Instruct, 4-bit NF4 on one GPU. The notebook has not been executed on Kaggle by its authoring agent.

Exactly 50 unique grids are selected without replacement from the original ToT `mini0505.json`, using selection seed 2026. The upstream commit, data hash, selected IDs and source license are saved. This is a custom random split, not ToT's published crossword test subset. No evaluation puzzle is used for tuning. Answer keys are passed to the scoring function only after search returns; the policy receives only the ten clues. Dataset availability does not establish absence from model pretraining.

## Methods and shared controls

- `tot_dfs`: ToT-style depth-first search, confidence-ordered children, model feasibility pruning and backtracking through retained alternatives. This is a local-model adaptation, not an exact reproduction of ToT's published prompts/evaluation protocol.
- `heuristic_bestfirst`: same candidates and evaluations, with best-first selection by the five-factor heuristic.

Each state holds ten explicit word assignments. Revising a word clears conflicting crossing assignments. Both methods use this same revision rule, duplicate-state filter, batch evaluation, candidate generation, shared feasibility threshold and final-output ranking. Full grids can still be revised; filling the grid is not a correctness certificate. The final grid maximizes `coverage * Q` among evaluated candidates, with coverage, P and smaller depth as tiebreakers. Ground truth never chooses the winning state or triggers early stopping.

Default limits per method and puzzle: 30,000 total input+output tokens, 40 calls, 20 expansions, depth 20, branching 3 and frontier 60. Exact tokenizer counts reserve the full completion allowance before each call. A limited budget may end with unused tokens. Both generation and batch judging are charged. Invalid JSON consumes its real cost and is never silently repaired.

## Heuristic

`H = .35 Q + .30 P - .05 D - .15 C + .15 V`

- Q: model-rated average plausibility of completely filled clue slots, including words implied by crossings.
- P: model-rated feasibility of satisfying the whole crossword. This is not a calibrated probability.
- D: normalized action depth, multiplied by `(1-P)`; revisions count as actions.
- C: estimated future-work proxy: unassigned-slot fraction times two average-cost calls, divided by total token budget, clipped to [0,1]. It is not a trained cost forecast. Averages include prompt and completion tokens.
- V: minimum normalized assignment distance from another retained frontier state, multiplied by `sqrt(Q*P)`. Singleton frontier diversity is zero.

Weights are carried over as provisional weights; no crossword weight fitting is performed. Features are task adaptations, not a claim of a learned cross-task policy. The notebook computes per-puzzle paired differences in word accuracy, with a bootstrap interval; treat small differences cautiously with only 50 puzzles.

## Output and resume

Both methods run on each puzzle before moving to the next; their execution order alternates. Sampling seeds are derived from puzzle, trial seed and prompt, so identical prompts use the same randomness in both methods. Calls are recorded/charged separately per method.

Output under `/kaggle/working/crosswords50_schema_v2/results/schema_v2_seed42/` includes individual result JSON files, `results_long.csv`, `per_question.csv`, `summary.csv`, `summary.json`, raw model calls and search traces. Summary metrics include word accuracy, letter accuracy, full-puzzle solved rate, tokens, calls, runtime, expansions and invalid responses (JSON/schema/truncation failures). Summaries use only puzzles completed by both methods. Inference time excludes model load and does not represent total session runtime.

Rerun the same run cell to resume. Completed methods are skipped and successful calls are replayed with their original costs. Source, model/runtime, dataset and config changes require a fresh output directory. Download the final ZIP/checkpoint using the export cell. To recover after a Kaggle reset, attach your own ZIP to a new session and extract it under `/kaggle/working` before setup. Use the same package versions and GPU/runtime for an exact resume; the notebook saves `pip_freeze.txt`. After a hard crash, remove a stale `run.lock` only when no benchmark is active. A call interrupted before its response is saved has unknown cost; for cost comparisons, start a fresh trial rather than silently ignoring interrupted GPU work.

Sources: https://github.com/princeton-nlp/tree-of-thought-llm and https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/scripts/crosswords/search_crosswords-dfs.ipynb
