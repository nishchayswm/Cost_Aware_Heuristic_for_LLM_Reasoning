"""Cost-aware priority search, using the shared benchmark engine."""
from benchmark_engine import METHODS
from solver_adapters import solve, batch

K_BEAM, N_CANDIDATES, D_MAX = 3, 2, 10

def search(question, problem_id=None, expected_answer=None, d_max=D_MAX, k=K_BEAM,
           n_candidates=N_CANDIDATES, w1=0.5, w3=0.1, verbose=True, seed=11):
    return solve(question, METHODS[3], problem_id, expected_answer, d_max, k,
                 n_candidates, w1, w3, verbose, seed)

def run_batch(problems, log_path='logs/priority_search_v3.jsonl', verbose=True):
    return batch(problems, search, log_path, verbose)

if __name__ == '__main__':
    from problems import PROBLEMS
    run_batch(PROBLEMS[:3])
