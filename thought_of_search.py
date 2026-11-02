"""Local entry point for the documented Thought of Search arithmetic adaptation."""
from benchmark_engine import METHODS
from solver_adapters import solve, batch


def tos_search(question, problem_id=None, expected_answer=None, d_max=10, verbose=True, seed=11):
    return solve(question, METHODS[5], problem_id, expected_answer, d_max=d_max,
                 verbose=verbose, seed=seed)


def run_batch(problems, log_path='logs/thought_of_search_v1.jsonl', verbose=True):
    return batch(problems, tos_search, log_path, verbose)


if __name__ == '__main__':
    from problems import STANDARD_PROBLEMS, HARD_PROBLEMS
    run_batch(STANDARD_PROBLEMS + HARD_PROBLEMS)
