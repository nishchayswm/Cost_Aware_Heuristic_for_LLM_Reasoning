# Game of 24: corrected engine (v2)

No new model benchmark has been run for this revision. 42 offline tests pass.
Previous results in results/game24_dev10 are preserved. Use a fresh output folder;
code/config hashes intentionally prevent resuming an older experiment with new code.

## Run the same 10 development puzzles

```powershell
cd "C:\Users\dhruv\OneDrive\Documents\COST\final_proj"
python -u -m costbench.game24 run --dataset data/game24/game24_dev.jsonl --limit 10 --output results/game24_dev10_v2
```

Use your existing dataset. If installing into a fresh project without prepared data,
run `python -m costbench.game24 prepare` first; it creates separate train/dev/test
sets of 100 locally generated solvable multisets. It refuses to overwrite data.
The generator uses an exact solver offline; the proposed policy never calls it.
These are locally generated puzzles, not the original ToT paper's benchmark.

After settings are frozen, run the test split:

```powershell
python -u -m costbench.game24 run --output results/game24_test100_v2
```

All eight methods finish on each puzzle before the next starts. Requests run
sequentially on the local model. Each terminal row prints the verified expression,
solved/failed status, calls and tokens. No results are promised in advance.

## Fixes supported by the previous traces

1. A globally valid solution is accepted even when it changes the current prefix.
   Prefix consistency is tracked separately and cannot give false success evidence
   to the selected branch. No arithmetic or input-usage rule has been relaxed.
2. Python enumerates all unique legal one-step states with exact Fraction arithmetic.
   The LLM ranks candidate IDs; it no longer invents operands or reconstructs the
   expression forest. Equivalent remaining-number multisets are deduplicated.
3. One model call produces a shortlist of scored candidates per expansion. Invalid
   IDs, duplicate IDs, nonnumeric/nonfinite/out-of-range scores are logged/rejected.
   Unrated states retain zero default scores and remain eligible for later expansion.
   Q assesses usefulness of a reduction, rather than merely output syntax. P is an
   explicit model estimate of eventual success. Both remain uncalibrated judgments.
4. Auxiliary probes choose legal move IDs; Python constructs every expression.
   They apply only to scored states with P between .35 and .65, and consume at most
   TWO LLM calls TOTAL per puzzle/method, rather than two completions for every node.
   A finished probe updates that state's P as (2*P+outcome)/3. An unfinished probe
   does not fabricate an outcome. Actual calls/tokens remain fully charged.
5. At two remaining values, Python tries only the possible FINAL operations.
   No LLM call is needed. All four search controls receive this exact same aid.
   This is a one-operation endgame check, not a full reachability oracle in ranking.
6. Frontier overflow is deferred instead of permanently discarded. When the active
   frontier empties, deferred states are restored; there are no repeated identical
   root restarts. Full search exhaustion and configured resource limits stop the run.
7. The same global call/token ceilings remain. max_expansions is now 32, including
   local endgame visits; a local visit can cost zero LLM calls. Comparisons between
   revisions therefore cover multiple infrastructure changes, not only weights.

The score remains:

H=.35 Q+.30 P-.05 (depth/3)(1-P)-.15 C+.15 V sqrt(Q*P)

C is a rough next-ranking-call estimate, zero for a two-value local endgame. The
report uses measured cost, not this estimate. V uses rational-number multiset
Jaccard novelty against previously expanded states. There is no learned-weight
or calibrated-probability claim, and no assumption that larger Q/P implies truth.

## Eight comparisons

| Method | Behavior |
|---|---|
| heuristic_bestfirst | Five-factor ranking over legal intermediate states |
| quality_bestfirst | Same engine, Q-only priority |
| tot_bfs | Shallowest active depth first, P-ranked bounded frontier |
| self_eval_beam | Shallowest active depth, stochastic P-based selection |
| cot | One natural-language solution in a JSON envelope plus final expression |
| self_consistency | Six samples, success if any expression verifies (success@6) |
| verified_sampling | Up to six samples, stop at first verified expression |
| symbolic_search | Separate complete deterministic search, zero LLM calls |

These are local adaptations, not published SOTA reproductions. The beam variants
use a deferred-state reserve, so they can revisit previously deferred candidates.
Fixed-budget sampling is not standard majority-vote SC: all valid answers are 24,
so expressions must be checked instead of voting on that number. The sampling
baselines generate full expressions; deterministic endgame aid applies to the four
state-search methods. Do not attribute that aid's benefit to the heuristic alone.

The symbolic reference exposes how much this small task can be solved by exact
arithmetic search alone. Its zero token count does not mean zero compute: reports
include mean_execution_seconds alongside inference wall time. It does not use a
model or supply answers to any other method. The proposed policy has no access to
its results, training labels, or the dataset-construction oracle.

## Verification and logs

A successful expression uses exactly the four original integer occurrences once
and evaluates exactly to 24. Negative/fractional intermediates are legal. No extra
constants, concatenation, powers, factorials or generated code execution is allowed.
An expression with valid syntax but the wrong numeric result is still incorrect.

Each output folder contains summary.csv, results_long.csv, per_question.csv,
report.md, report.html, manifest.json, results.jsonl, calls/ and traces/.
Summary statistics use only complete EIGHT-method puzzle blocks. New metrics record
rating failures, rollout calls, endgame checks, off-branch solutions, actual tokens,
inference wall time and method execution wall time. Traces preserve candidate
IDs, scores, expansion features, legal transitions, rollouts and final checks.

No warmup is excluded; startup may affect request wall time. Model order rotates
by puzzle. Backend failures stop the run, and --retry-failed explicitly retries
with unknown cost recorded. Compare costs only with that flag in mind.

## Resume, pause, reports and tests

Repeat the exact run command to resume saved work. Changed code/config/model/data
or question limit requires a new output directory. One output folder holds one
seed. Use --seed 43 / --seed 44 with different folders for repeated experiments.
Never run two runners into the same folder.

```powershell
python -m costbench pause --output results/game24_dev10_v2
python -m costbench.game24 report --output results/game24_dev10_v2
python -m unittest discover -s tests -v
```

Use the Game of 24 report command, not the generic GSM8K report command.
Offline tests cover exact arithmetic, duplicate-number usage, previously discarded
solutions, complete-solution/branch separation, legal move enumeration, no runtime
oracle use, zero-call endgames, score validation, bounded move probes, deferred
states, budgets, all-method output and resume. They do not establish live accuracy.

Freeze settings on train/dev before using untouched test puzzles. Starting-number
multisets (including their permutations) never cross the generated split boundaries.
