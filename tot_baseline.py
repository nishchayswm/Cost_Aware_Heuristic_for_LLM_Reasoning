"""ToT using the shared generator plus an LLM judge for candidate ranking."""
from benchmark_engine import METHODS
from solver_adapters import solve, batch

K_BEAM, N_CANDIDATES, D_MAX = 3, 2, 10

def tot_search(question, problem_id=None, expected_answer=None, d_max=D_MAX, k=K_BEAM,
               n_candidates=N_CANDIDATES, verbose=True, seed=11):
    return solve(question, METHODS[2], problem_id, expected_answer, d_max=d_max,
                 k=k, n_candidates=n_candidates, verbose=verbose, seed=seed)

def run_batch(problems, log_path='logs/tot_baseline_v3.jsonl', verbose=True):
    return batch(problems, tot_search, log_path, verbose)

if __name__ == '__main__':
    from problems import PROBLEMS
    run_batch(PROBLEMS[:1])
