# Reliability fixes, v2

- Generation JSON contains only steps (claim/expression) and complete.
- Python assigns IDs, derives dependencies and calculates exact arithmetic. It never guesses unknown references.
- Search judges check coverage, interpretation, units and completion before Q/P scoring.
- One structural regeneration is allowed for every method, inside existing budgets. Traces record invalid attempts and retry costs.
- The five-factor formula, weights, traversal rules and final-answer selection are unchanged.
- Original and reviewed label scores appear separately in terminal output, CSV, Markdown and HTML. The dataset remains unchanged.
- Runs default to results/gsm8k100_v2. Earlier pilot results remain in results/gsm8k100. Code changes prevent mixing or resuming an old experiment.

These changes address observed formatting and evaluation weaknesses. They do not establish better model accuracy or superiority to baselines. Only synthetic offline tests were run for this revision. Two inspected test questions informed the diagnosis; disclose this exposure when reporting results.

Run from final_proj:

```powershell
python -m unittest discover -s tests -v
python -u -m costbench run --dataset data/gsm8k_dev_100.jsonl --limit 1 --output results/dev_check_v2
python -u -m costbench run --output results/gsm8k100_v2
```

The last two commands invoke the local model; run them yourself when ready.
