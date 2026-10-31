# V5: working judge signals and reliability-aware search

The V4 diagnostic was paused after 17 method results (two complete six-method
blocks and five results on the third question). All search judges inspected there
returned zero Q/P scores, preventing meaningful state ranking. The raw results
remain in results/dev_v4_10; they are not a completed 10-question experiment.

Two diagnostic calls using the first development question are retained separately
in results/judge_diagnostics_v5.jsonl, including actual token counts and timing.
Changing to JSON mode produced nonzero scores and identified a missing contribution.
This implicates the previous constrained judge protocol on this local backend;
it does not prove which individual prompt/schema feature caused the collapse.

## Current changes

- Generation still uses a constrained arithmetic schema and AST checking.
- Judging uses JSON mode, asks for a short mathematical reason before its rubric
  and scores, and validates required fields and finite scores in [0,1] in Python.
  The per-judge output cap is 600; the global call/token ceilings are unchanged.
- The judge distinguishes claimed-complete answers from unfinished paths.
- A claimed completion can reopen if interpretation passes, units do not fail,
  and completion is fail/pending or coverage fails. It retains the existing
  reasoning and consumes normal expansion budgets. Wrong prefixes are not repaired.
- Q/P are attenuated by explicit rubric failures (see CHANGES_V4.md).
- The proposed priority is

  H = .35 Q + .30 P - .05 (depth/max_depth)(1-P)
      - .15 C + .15 V sqrt(Q P)

  C includes estimated next-generation tokens and amortized judging tokens.
  Weights remain fixed. Confidence and lexical diversity are still proxies.
- All search baselines receive the common judge, recovery and final selector.
  Quality-only best-first differs in state ranking, providing a same-engine control.
  CoT/self-consistency use the shared generation protocol without a search judge.
- configs/v5_linear_control.json restores ungated diversity and plain depth while
  keeping V5 shared fixes, for a future controlled ablation on the same questions.

## Experiment

26 synthetic tests passed before running. All six methods run on the same first
10 training/development questions, seed 42, in results/dev_v5_10. No test-set run
is performed. Code and config are frozen during this final comparison. Both the
preliminary run and diagnostic probes are exploratory development work and must
be disclosed; their cost is separate from the final benchmark's per-method costs.
These ten questions cannot establish SOTA or generalization, and V2 used different
questions so its percentages are not a valid before/after accuracy comparison.

```powershell
cd "C:\Users\dhruv\OneDrive\Documents\COST\final_proj"
python -u -m costbench run --dataset data/gsm8k_dev_100.jsonl --limit 10 --output results/dev_v5_10
```

This exact command resumes the recorded experiment. The default full test output
is now results/gsm8k100_v5. Preserve previous outputs and do not mix code versions.
