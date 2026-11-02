"""Compatibility entry points for local scripts; all use the shared v3 engine."""
import json
from pathlib import Path
from benchmark_engine import Config, METHODS, Solver
from benchmark_runtime import ollama_backend


def solve(question, method, problem_id=None, expected_answer=None, d_max=10,
          k=3, n_candidates=2, w1=0.5, w3=0.1, verbose=True, seed=11):
    if method == METHODS[3] and w3 == 0:
        method = METHODS[4]
    elif method == METHODS[3] and w3 != 0.1:
        raise ValueError('Use w3=0.1 or w3=0 for the documented comparison')
    result = Solver(ollama_backend, Config(seed=seed, max_depth=d_max, beam=k,
                    candidates=n_candidates, w1=w1)).solve(question, method)
    ans = result['ans']
    correct = (ans is not None and abs(ans - expected_answer) < 1e-4) if expected_answer is not None else None
    result.update(problem_id=problem_id, question=question, expected=expected_answer,
                  expected_answer=expected_answer, final_answer=ans, correct=correct,
                  outcome='missing' if ans is None else 'unscored' if correct is None else
                          'correct' if correct else 'wrong')
    # Preserve counters used by older callers without using their cached transport.
    from llm import stats
    stats.update(calls=result['calls'], tokens=result['tokens'], hits=0)
    if verbose:
        print(f'{method}: answer={ans}, status={result["status"]}, calls={result["calls"]}')
    return result


def batch(problems, solver, log_path, verbose=True):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to destroy a previous experiment.
    rows = []
    with path.open('x', encoding='utf-8') as handle:
        for pid, (question, expected) in enumerate(problems):
            row = solver(question, problem_id=pid, expected_answer=expected, verbose=verbose)
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            rows.append(row)
    return rows
