# Original ToT Game of 24 — local runner

This converts the dataset preparation, overlap audit and reporting from `kaggle/Game24_Original_ToT_Kaggle.ipynb` into a local command-line workflow. All eight existing methods are reused unchanged. Their source files were verified to match the notebook snapshot at conversion time. Python 3.11+ and an installed/running Ollama are required; no third-party Python packages are needed.

From `C:\Users\dhruv\OneDrive\Documents\COST\final_proj`:

```text
ollama pull qwen2.5:7b-instruct
python local_game24_original/run.py prepare
python -u local_game24_original/run.py run --limit 100 --seed 42
```

If Ollama is not already running, run `ollama serve` in a separate terminal. If the model is already installed, the pull command can be skipped. Commands work in CMD and PowerShell; no shell-variable syntax is used.

For a smaller pilot:

```text
python -u local_game24_original/run.py run --limit 15 --seed 42
```

The full run writes to `results/game24_original_n100_seed42_ollama`; the 15-puzzle run uses its own `n15` directory. Change `--seed` to 43 or 44 to repeat the same problems with different model sampling. The limit takes the first N rows of the official fixed evaluation split, not a new random subset.

## Data and methods

The runner downloads the original ToT repository's `24.csv` and license, pins the Git revision, verifies the CSV hash, and selects zero-based rows 900–999. Downloads require Internet on first preparation. Later runs reuse the local files without network downloads. Supply `prepare --commit <40-character-revision>` in a fresh dataset directory to match an existing Kaggle source manifest exactly, or copy its `data/original_tot` folder into this project's `data/original_tot` before running. Existing changed artifacts are rejected instead of overwritten.

Methods: five-factor heuristic best-first, quality-only best-first, bounded ToT-style breadth search, self-evaluation stochastic beam, CoT, fixed-budget verified sampling (legacy `self_consistency` ID), early-stopping verified sampling, and symbolic search. Every method runs on one problem before moving to the next. The fixed sampling comparator is success@6, not majority-vote self-consistency. These are the project's adaptations rather than exact reproductions of published SOTA systems.

The same expansion-64 config and default Qwen2.5 7B model family are used. Local Ollama/GGUF and JSON handling differ from the Kaggle Transformers/NF4 backend. Compare methods within each run; do not pool those backends as identical experiments. No model inference was executed during conversion.

## Output and resume

Per-problem results are printed to the terminal. Saved files include `results.jsonl`, `results_long.csv`, `per_question.csv`, `summary.csv`, HTML/Markdown reports, raw calls, search traces, the run config, provenance and overlap audit. A supplementary `disjoint_subset_summary.csv` excludes puzzles overlapping any prepared generated train/dev/test split; the full official split remains unchanged.

Rerun the identical command to resume. Changed code/config/model/dataset requires a new `--output` directory. The runner records its own hash as well as the engine's hash. If you change the model, specify both `--model NAME` and a fresh `--output` path. A failed inference with unknown cost is flagged by the existing engine; do not silently merge it into a clean-cost experiment. Do not delete an active run lock.

Regenerate reports without calling the model:

```text
python local_game24_original/run.py report --output results/game24_original_n100_seed42_ollama
```

Original reference: https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/scripts/game24/bfs.sh
