# Self-Consistency Chain-of-Thought Evaluation Results (`k=10` & Multi-`k` Replay)

- **Generator Model:** `qwen2.5:7b-instruct` (Local Ollama on NVIDIA GeForce RTX 4060 8GB VRAM)
- **Total Benchmark Problems:** `150` (`GSM8K=30`, `MATH-500=20`, `Game of 24=20`, `HotpotQA=25`, `ASQA=55`)
- **Local Monetary Cost:** `$0.00` (Free local execution; simulated cloud cost: `$0.00259` / problem)

## 1. Per-Subset Results at Fixed `k=10`

| Subset | Group | Problems (N) | Correct / Solved | Mean Quality Score | Mean API Calls (10 x CoT steps) | Total Subset Calls | Mean Tokens / Problem |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** | Group A | 30 | 26 / 30 (86.7%) | **0.8667** | 18.83 | 565 | 3667.0 |
| **MATH-500** | Group A | 20 | 6 / 20 (30.0%) | **0.3000** | 36.95 | 739 | 6927.6 |
| **Game of 24** | Group A | 20 | 0 / 20 (0.0%) | **0.0000** | 32.95 | 659 | 5967.2 |
| **HotpotQA** | Group A | 25 | 14 / 25 (56.0%) | **0.7461** | 13.52 | 338 | 17767.8 |
| **ASQA** | Group B | 55 | 0 / 55 (0.0%) | **0.1806** | 24.11 | 1326 | 4386.3 |

## 2. Pooled Macro-Average Quality (`k=10`)

| Evaluation Group | Included Subsets | Pooled Macro Quality | Mean API Calls / Problem | Mean Tokens / Problem |
| :--- | :--- | :---: | :---: | :---: |
| **Group A (Exact / Short)** | GSM8K, MATH-500, Game of 24, HotpotQA | **0.4782** | 24.22 | 8548.4 |
| **Group B (Long-Form)** | ASQA (DR = sqrt(STR-EM * ROUGE-L)) | **0.1806** | 24.11 | 4386.3 |
| **Overall Macro (All 5 Subsets)** | Equal weight across all 5 subsets | **0.4187** | 24.18 | 7022.3 |

## 3. Part 3 Replay Pareto Frontier (`k = 1, 3, 5, 8, 10`)

| Config (k) | Group A Macro Quality | Group B (ASQA) Quality | Overall Macro Quality | Mean API Calls / Problem | Total Dataset API Calls | Mean Tokens / Problem |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `k=1` | 0.4173 | 0.1565 | **0.3651** | 2.43 | 364 | 704.2 |
| `k=3` | 0.4284 | 0.1627 | **0.3753** | 7.30 | 1095 | 2110.5 |
| `k=5` | 0.4409 | 0.1669 | **0.3861** | 12.09 | 1813 | 3510.5 |
| `k=8` | 0.4500 | 0.1683 | **0.3937** | 19.34 | 2901 | 5617.4 |
| `k=10` | 0.4782 | 0.1681 | **0.4162** | 24.18 | 3627 | 7022.3 |
