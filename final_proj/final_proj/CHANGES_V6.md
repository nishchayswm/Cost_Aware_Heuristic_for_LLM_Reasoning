# V6: numeric reasoning states and a complete-solution incumbent

## Motivation and implementation

The completed V5 development run scored 4/10 for five-factor best-first, 7/10 for
CoT and 8/10 for self-consistency. It exposed two major shortcomings: confusing
step identifiers with the wrong quantities, and returning a subtotal when a final
conversion was still needed. Better weights alone could not fix these issues.

V6 keeps the reliability-adjusted five-factor formula from V5:

H = .35 Q + .30 P - .05 (depth/max_depth)(1-P) - .15 C + .15 V sqrt(Q*P)

It adds the following shared infrastructure changes:

1. **Numeric expressions.** The model writes numbers from the problem or values
   already computed, rather than symbolic s1/s2 references. Python still stores
   sequential step IDs and reasoning paths. Dependencies are implicit in quantity
   descriptions and substituted values; this is not a symbolic dependency proof.
   A schema and runtime checks prohibit generated variable references. Old symbolic
   traces remain readable by the calculator for compatibility.
2. **Complete-solution incumbent.** Every search method first generates and judges
   one complete solution inside the same call/token budget. Search still explores
   intermediate paths, but has an answer candidate available. It may replace this
   candidate according to the same shared final selector sqrt(Q*P). It is not a
   trusted oracle and does not guarantee that a correct initial answer is retained.
   This applies to quality-only and both beam controls as well as the proposed
   method, and must be described as an adaptation when comparing published methods.
3. **Alternative sibling generation.** Later siblings see earlier proposed next
   steps as untrusted data and are asked for another valid derivation if possible.
   This reduces redundant branches without requiring artificial differences.
4. **Bounded powers.** Exact Fraction arithmetic supports integer exponents from
   -32 to 32, subject to the existing intermediate magnitude limits. No generated
   code is executed; huge/fractional exponents and invalid arithmetic are rejected.

Weights, total token/call caps, datasets and seeds remain unchanged. Shared fixes
must not be attributed solely to the heuristic. Quality-only best-first is the
same-engine control. configs/v6_linear_control.json additionally disables the
confidence gates on diversity and depth with all shared V6 fixes retained.

## Evaluation and reproducibility

29 synthetic tests passed before inference. All six methods run on the same first
10 development/train questions, seed 42, in results/dev_v6_10, with code/config
frozen throughout. The completed V5 run remains in results/dev_v5_10, and the
abandoned V4 diagnostic plus two judge probes are also retained. Source snapshots
and exact configs accompany the V5 and V6 results. Repeated development inspection
and revision mean these are exploratory outcomes, not unbiased test estimates.
An unseen, frozen holdout is required before claiming general accuracy gains.

The run logs every method's own calls and tokens. Incumbent generation and judging
are included. Development probes and earlier revisions are separately recorded
and are additional project expenditure, not hidden inside final inference costs.

```powershell
cd "C:\Users\dhruv\OneDrive\Documents\COST\final_proj"
python -u -m costbench run --dataset data/gsm8k_dev_100.jsonl --limit 10 --output results/dev_v6_10
```

The original 100-question test benchmark was not run. Its fresh default output
would be results/gsm8k100_v6. Never resume older experiments under changed code.


## Completed development results

All 60 results were saved for 10 questions and six methods. Proposed: 9/10,
6.6 mean calls, 3351 mean tokens. Self-consistency: 9/10, 6.3 calls, 2655 tokens.
Quality-only: 8/10, 7.5 calls, 3810 tokens. CoT: 8/10, 1 call, 416 tokens.
The proposed system improved from V5 4/10 to 9/10 on these exposed development
questions, with five fixes and no regressions. Shared generation and incumbent
changes contributed; this does not isolate the formula's causal contribution.
It matches self-consistency accuracy here but costs MORE tokens, so it has not
achieved the desired accuracy/cost advantage over self-consistency. The remaining
proposed failure is the compound-interest question. No test labels were changed.

Results: results/dev_v6_10/report.html, summary.csv, revision_comparison.md,
diagnostics.json. All results meet configured budgets, code hash matches the
frozen manifest, and no unknown-cost backend failures were recorded.
