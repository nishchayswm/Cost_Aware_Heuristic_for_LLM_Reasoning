# Mini crosswords: search v5

Upload `Crosswords50_Heuristic_vs_ToT_Kaggle.ipynb` into a fresh Kaggle GPU session with Internet enabled. Run setup and the synthetic preflight, then set `RUN_BENCHMARK=True`. Model inference was not run locally while preparing this update.

## Choose the puzzle count

Set `NUM_PUZZLES=15` in the dataset cell. Any integer from 1 to 50 is supported. This selects the first N from the fixed, randomly selected 50-puzzle pool, using selection seed 2026. Changing N automatically produces a separate result folder. No source patch is needed. Source revision, pool IDs and selected-run data hash are saved. This custom split is not the published ToT evaluation subset.

## Changes after reviewing the critique

- Global confidence-first proposals across incomplete clues replace mandatory most-constrained-clue selection. The model can choose an easier anchor clue instead of always starting at h1. Up to three moves are proposed per expansion.
- Synthetic few-shot examples demonstrate exact spelling positions, clue meaning, invalid patterns and abstention. Examples are authored independently, not copied from benchmark solutions.
- Schema decoding checks JSON structure and five-letter word length, but no longer forces all words to fit one selected pattern during decoding. Python checks crossing consistency and discards conflicts; an empty moves list is valid.
- The minimum of individual slot feasibility values controls normal expansion. A value below 0.15 blocks normal expansion even if geometric-mean P is high. Such states can only take a bounded explicit repair transition; otherwise they are discarded. P remains an uncalibrated heuristic feature.
- Final selection excludes states with an infeasible slot or a completed word rated below 0.5. Among eligible states, rank summed completed-word fit. An empty or partial grid is allowed when no plausible full grid is found. These model-based thresholds can falsely reject rare valid answers and are provisional, not calibrated guarantees.
- Budgets increase equally to 120 calls, 80,000 total tokens and 60 expansions per method/puzzle. Larger budgets are a separate experiment, not evidence of improved efficiency over prior 40-call runs.

The existing per-clue-pattern evaluation cache is retained: unchanged ratings were already reused. Both generation and evaluation retries are charged. The critique's detailed accuracy/retry figures were not independently verified from a new trace archive. Its claim that completed slots automatically receive 1.0 is incorrect: our previous version also used model-assigned scores. Current fixes target reliance on those scores, not a nonexistent unconditional 1.0 assignment.

## Shared engine and heuristic

ToT-style DFS and heuristic best-first use the same generator, judge, cache rules, checks, repair transitions, budget and final selector. Only frontier selection differs. This remains a local-model adaptation, not an exact replication of the official GPT-4 experiment. Matching final answers do not mean the policies are identical.

`H = .35 Q + .30 P - .05 D - .15 C + .15 V`

Q is completed-clue quality scaled by sqrt(completed fraction); P is geometric-mean slot feasibility with a 0.5 prior for unconstrained slots; D is normalized depth times (1-P); C estimates remaining incomplete slots times two average calls, normalized by token budget; V is assignment diversity gated by sqrt(raw quality * P). Weights are unchanged. Hard local feasibility checks are separate from this score. Repair removes suspect supporting assignments explicitly, is logged, and is bounded to three transitions per path.

A small 7B model may still lack rare clue knowledge or misjudge candidate words. These changes cannot guarantee good scores or a win over DFS. Do not tune to specific reference answers. The synthetic preflight validates formatting and a legal placement, not real crossword accuracy.

## Upgrade, outputs and resume

New project: `/kaggle/working/crosswords50_search_v5`. Default result folder: `results/search_v5_n15_seed42`. Old v4/v3/v2 dataset selections and model pins may be carried forward in the same session; previous scores and calls are never merged. Restart the kernel before loading the model to release older instances.

Both methods run on each puzzle before moving on, with alternating order. Printed output includes grids, accuracy, fill, calls, tokens and stop reasons. Paired CSVs include `same_prediction` and `same_search_path`; traces retain selected states, available clues, per-clue scores, frontier features and repair removals. Raw responses, individual results, summaries and plots are saved. Export `crosswords50_results.zip` and save Kaggle outputs before ending a session.

Resume only with identical source, dataset, model/runtime and config. Successful calls replay with original costs. An interrupted call with unknown cost blocks a clean-cost trial. Source or budget changes require fresh results. Tests are offline mocks; the grammar integration test runs on Kaggle after dependency installation and may be skipped locally if lm-format-enforcer is absent.

Official reference: https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/scripts/crosswords/search_crosswords-dfs.ipynb . The official notebook proposes across clues and prunes using impossible ratings; its exact prompts, model, sampling and budget differ from this adaptation.
