"""One complete generation; explicit final marker, no guessed numeric fallback."""
from benchmark_engine import METHODS, extract_explicit_answer
from solver_adapters import solve, batch

extract_final_answer = extract_explicit_answer

def solve_standard_cot(question, problem_id=None, expected_answer=None, verbose=True, seed=11):
    return solve(question, METHODS[0], problem_id, expected_answer, verbose=verbose, seed=seed)

def run_batch(problems, log_path='logs/standard_cot_v3.jsonl', verbose=True):
    return batch(problems, solve_standard_cot, log_path, verbose)

if __name__ == '__main__':
    from problems import PROBLEMS
    run_batch(PROBLEMS[:3])
