# Game of 24 benchmark

Run each of the six notebooks separately in Kaggle with Internet and a GPU enabled.
Each notebook runs one method on the same 100 fixed test puzzles with seed 42
and model `qwen2.5:7b-instruct`.

- `Game24_heuristic_bestfirst.ipynb`
- `Game24_without_quality.ipynb`
- `Game24_without_success.ipynb`
- `Game24_without_depth.ipynb`
- `Game24_without_cost.ipynb`
- `Game24_without_diversity.ipynb`

Download the named result files and place them in `results/`. The directory is
initially empty except for `.gitkeep`, which allows Git to track it.
Result filenames include the method, model, seed, split and number of puzzles.
Rows also include puzzle IDs and dataset/configuration hashes for merging.
