# Parallel crossword benchmark

Upload one different notebook to each of five Kaggle GPU sessions. Enable GPU
and Internet, run installation, restart the kernel, then continue with the next
cell. Each notebook embeds its source, offline tests, the same frozen dataset,
and the exact Qwen/Qwen2.5-7B-Instruct model revision. After preflight passes,
the run cell starts its assigned two puzzles with both methods by default.

| Notebook | Global puzzle indices | Puzzle IDs |
|---|---|---|
| `Crosswords10_Batch01_of_05.ipynb` | 1-2 | crossword-0030, crossword-0081 |
| `Crosswords10_Batch02_of_05.ipynb` | 3-4 | crossword-0128, crossword-0131 |
| `Crosswords10_Batch03_of_05.ipynb` | 5-6 | crossword-0026, crossword-0057 |
| `Crosswords10_Batch04_of_05.ipynb` | 7-8 | crossword-0142, crossword-0107 |
| `Crosswords10_Batch05_of_05.ipynb` | 9-10 | crossword-0146, crossword-0140 |

The batches cover ten unique puzzles exactly once. Methods are `tot_dfs` and
`heuristic_bestfirst`, with generation seed 42 and identical limits per method
and puzzle: **30,000 input+output tokens, 60 calls including retries, and 30
expansions**. There are four solver runs per notebook and twenty in total.
Preflight costs are recorded separately in each session. Output constraints
enforce format and five-letter word length; they do not prove clue correctness.

Download each notebook's uniquely named ZIP, or its named result JSONL and
matching manifest JSON. The export cell can discover saved results after a
kernel restart. Store downloads in `results/`; it starts empty except for
`.gitkeep`, which lets Git track the directory. Raw result ZIPs, prompts,
traces, runtime and dataset provenance are preserved in the exports.

To merge, extract each ZIP once under a downloads directory, then run from the
repository root:

```powershell
python benchmark_crossword/merge_crossword_batches.py downloads --output crossword_merged
```

Alternatively place the five named result JSONL files and their matching
manifest JSON files in that directory. Avoid including both those files and
another extracted copy of the same batch. The merger requires five complete
batches and rejects missing/duplicate results or mismatched data, model,
code, package versions, settings or seeds. It produces merged JSONL,
per-question comparisons and summary CSV/JSON files. Each batch's runtime
and GPU are retained; timing across different GPU types is not controlled.

These notebooks passed offline checks, including partitioning, resume, merge
validation and restart-safe exports. A live Kaggle GPU run is still required.
