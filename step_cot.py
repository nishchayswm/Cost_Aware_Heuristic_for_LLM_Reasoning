"""Greedy step-wise CoT with the common structured generation policy."""
from benchmark_engine import METHODS
from solver_adapters import solve, batch

def solve_step_cot(question, problem_id=None, expected_answer=None, d_max=10, verbose=True, seed=11):
    return solve(question, METHODS[1], problem_id, expected_answer, d_max=d_max,
                 k=1, n_candidates=1, verbose=verbose, seed=seed)

def run_batch(problems, log_path='logs/step_cot_v3.jsonl', verbose=True):
    return batch(problems, solve_step_cot, log_path, verbose)

if __name__ == '__main__':
    from problems import PROBLEMS
    run_batch(PROBLEMS[:2])
