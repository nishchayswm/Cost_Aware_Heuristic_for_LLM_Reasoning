# V4: reliability-aware five-factor search

## Formula

H(s) = .35 Q + .30 P - .05 [depth/max_depth * (1-P)]
       - .15 C + .15 [V * sqrt(Q*P)]

Q and P are the same model's estimates, adjusted for contradictions in its rubric.
Reliability R is one minus the fraction of relevant checks marked fail. The
relevant checks are interpretation and units for partial paths, plus coverage and
completion for completed paths. Q=raw_Q*R and P=raw_P*R. Pending and uncertain
checks do not count as explicit failures. This consistency rule is a heuristic,
not an empirical probability calibration.

The depth term reduces penalties for deeper paths the judge considers promising.
The gated diversity term rewards novelty in plausible states. C now estimates
next generation input/output tokens plus a share of observed judge tokens,
normalized and capped as before. Judge cost uses an EWMA (0.7 previous + 0.3 new),
initially the configured output-token cap. It is amortized by branch factor;
invalid or deduplicated siblings can make that estimate optimistic. Actual costs
always include every request, including retries and judges.

## Shared search fixes and controlled comparison

All four search methods use the reliability adjustment, judging, cost accounting,
and final selector. The selector ranks evaluated completed candidates by sqrt(Q*P)
with generation-order tie breaking. If the judge marks coverage or completion
fail but both interpretation and units pass, a completed state below the depth
limit is reopened for continuation. It is removed from final-answer candidates.
Incorrect prefixes are not rewritten and gold answers never enter this decision.
There is no extra unbudgeted inference. A false judge failure can reopen a correct
answer; this risk must be measured, not assumed away.

CoT and self-consistency retain their usual no-judge selection and share the same
arithmetic format, validation, retries, model and budget ceilings. Quality-only
best-first controls for the common search fixes: only its state ranking differs.
The config v4_linear_control.json additionally restores ungated novelty and the
original depth term while keeping shared fixes. It can isolate scoring changes
on the SAME development questions and seeds. The older v3 control does not isolate
v4 changes and should not be used for that purpose.

## Validation

25 synthetic tests passed before inference. The authorized live experiment runs
all six methods on the first 10 fixed TRAIN/development questions, seed 42, output
results/dev_v4_10. Each question finishes across methods before moving on. The
code/config are frozen for this experiment; no changes are made based on intermediate
scores. These are diagnostic development outcomes, not held-out SOTA evidence.
Earlier v2 scores used different questions and cannot establish before/after gains.

Run from final_proj:

```powershell
python -u -m costbench run --dataset data/gsm8k_dev_100.jsonl --limit 10 --output results/dev_v4_10
```

Repeating the exact command resumes or refreshes this experiment. Preserve its
code/config; a different setup requires a new output folder. The original 100-question
benchmark is not run as part of this change.
