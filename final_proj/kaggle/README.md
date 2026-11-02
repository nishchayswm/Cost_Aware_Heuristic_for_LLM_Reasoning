# Kaggle Game of 24 benchmark

Import `Game24_Original_ToT_Kaggle.ipynb` into Kaggle. The notebook embeds the project: no separate code upload or Ollama installation is required.

1. Enable a GPU accelerator and Internet in notebook settings.
2. Run the setup, offline tests, dataset download, overlap audit, and model-loading cells in order.
3. In the run cell set `RUN_BENCHMARK=True`. Keep `LIMIT=100`, `SEEDS=[42]` initially.
4. Run the benchmark cell. It runs all eight methods for each puzzle before proceeding to the next puzzle, printing each result in the cell output.
5. Run the summary and export cells. Download `/kaggle/working/game24_kaggle_results.zip` and save the notebook outputs.

The original dataset is the Tree of Thoughts repository's `24.csv`, rows 900 through 999 (zero-based), matching its official evaluation script. Repository commit and CSV hash are recorded. This is the official fixed split, not a random sample. No solutions or human difficulty ratings are passed to the model.

Methods: heuristic best-first, quality-only best-first, bounded ToT-style breadth search, stochastic self-evaluation beam, CoT, fixed-budget verified sampling (`self_consistency` legacy identifier), early-stopping verified sampling, and symbolic search. These are the existing project comparisons; they are not exact reproductions of published SOTA configurations. Fixed-budget verified sampling is success@6, not majority-vote self-consistency.

Optional: `INCLUDE_GENERATED=True` also evaluates your generated test dataset under this same backend. Additional seeds `[42,43,44]` repeat the same puzzles with different sampling randomness. Other tasks such as GSM8K are not wired into this notebook.

Saved outputs include per-problem scores for every method, summaries, HTML reports, raw responses, token/call costs, search traces, dataset provenance, package versions and model revision. The overlap audit checks the original puzzles against all three existing generated splits; a supplementary disjoint-subset summary is produced when available.

## Resume

Within the same session, rerun the benchmark cell with identical settings. Completed results and saved calls are reused. After an interruption, run the export cell to save a checkpoint. For a new session, attach that ZIP as Kaggle input and extract it into `/kaggle/working` before running setup; it contains the `cost_game24` directory. Use only your own checkpoint archive. Existing bundled source is checked before reuse. Resume requires the same model revision, package/runtime fingerprint, config and dataset; otherwise start a fresh experiment. A hard session shutdown may leave a stale `run.lock`; remove it only when no benchmark is running.

## Interpretation and validation

The model is `Qwen/Qwen2.5-7B-Instruct`, locally generated on one GPU using Transformers and 4-bit NF4. JSON is requested by prompting and checked by Python; constrained grammar decoding is not used. This backend differs from Windows Ollama/GGUF, so compare methods within this experiment. Model download and inference require Kaggle resources; actual runtime depends on accelerator and search size.

Compare the heuristic primarily against quality-only best-first, which shares the search engine. Symbolic search is a separate zero-LLM reference. Search methods have legal-move and endgame assistance that full-solution sampling does not. Do not attribute that entire advantage to the heuristic or claim a SOTA improvement solely from this experiment.

Local checks compile all notebook cells, exercise bundle extraction/dataset conversion with synthetic input, mock the model adapter, and run the existing offline test suite. No benchmark inference or Kaggle GPU run was performed while creating this notebook.
