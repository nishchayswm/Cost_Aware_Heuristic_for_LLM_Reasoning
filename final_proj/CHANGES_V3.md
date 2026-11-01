# V3: observed failures and targeted changes

## Observed v2 pilot (18 complete questions, 108 method results)

| Method | Correct | No prediction | Mean tokens |
|---|---:|---:|---:|
| heuristic_bestfirst | 9/18 | 8 | 3141 |
| cot | 2/18 | 16 | 841 |
| self_consistency | 7/18 | 6 | 4754 |
| tot_bfs | 7/18 | 7 | 2850 |
| self_eval_beam | 9/18 | 7 | 3428 |
| quality_bestfirst | 9/18 | 6 | 3582 |

Original and reviewed accuracy are identical for this pilot. The proposed method
has eight missing predictions and one numeric wrong answer. In those eight cases
all generated paths fail at the root. Most failures are assignment syntax or
nonexistent references. Weight tuning cannot recover a state that never entered
the search. Low CoT accuracy is dominated by formatting rejection (16 missing
answers), so these results do not establish superiority over a healthy baseline.

## Implemented changes

1. All methods receive expression character constraints, a small synthetic format
   example, and clearer retry feedback. Single-step prompts explicitly list
   available references; root expressions cannot use step variables in the schema.
   Runtime AST validation remains authoritative; schema constraints alone cannot
   prove dependency or semantic correctness.
2. A redundant `s1 = 7*3` or `s1 := 7*3` can become `7*3` only at step s1.
   Original expression and normalization are logged. Wrong IDs, unknown aliases,
   forward/self references, executable code and division by zero remain invalid.
3. The judge prompt no longer contains a literal zero-score output example.
   This removes a possible copying/anchoring cue; its actual effect is untested.
4. Proposed state selection now uses:

   H = .35 Q + .30 P - .05 D - .15 C + .15 V sqrt(Q P)

   Q, P, D, C and raw lexical V keep their previous definitions. This reduces
   novelty rewards for paths assessed as poor. It adds no LLM call and leaves
   budgets, weights and common final-answer selection unchanged. It is a hypothesis,
   not a calibrated or proven optimal policy. Erroneously low judge scores can
   suppress useful exploration; the ungated control is essential.
5. New output defaults to results/gsm8k100_v3; previous experiments are preserved.

The remaining numeric failure involved an incorrect interpretation, not just
arithmetic. These changes do not guarantee recovery of such errors. Existing rubric
checks are still model judgments and do not impose a hard score gate.

## Validate without tuning to these test outcomes

Run from final_proj. First compare all six methods on separate development data:

```powershell
python -u -m costbench run --dataset data/gsm8k_dev_100.jsonl --limit 10 --output results/dev_v3_gated
python -u -m costbench run --config configs/v3_ungated_control.json --dataset data/gsm8k_dev_100.jsonl --limit 10 --output results/dev_v3_ungated
```

Compare accuracy, missing predictions, schema failures, tokens and calls. Both runs
have the same generation fixes; the configuration changes only the proposed
ranking gate. Replicate with multiple seeds before selecting the setting. Judges,
final-answer selectors and shared fixes are identical across search baselines.

To rerun the 100-question diagnostic comparison when ready:

```powershell
python -u -m costbench run --output results/gsm8k100_v3
```

Because the first 18 test questions were inspected to inform this revision,
report them as development-exposed; improvements on them are not independent
confirmation. Do not claim SOTA or a guaranteed gain. Use untouched held-out
questions for a confirmatory evaluation after freezing settings.

Validation: 23 synthetic offline tests passed. No v3 model or GSM8K inference was
performed. Offline tests verify protocol, safety, accounting and score behavior;
they cannot establish model accuracy or live Ollama grammar compatibility.
