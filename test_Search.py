"""
Quick test script for Priority Search.
Validates the end-to-end execution of Search.py on a benchmark problem.
"""

from Search import search
from problems import PROBLEMS

if __name__ == "__main__":
    q, expected = PROBLEMS[0]
    print("Testing Priority Search on Problem 0:")
    print("Question:", q)
    print("Expected Answer:", expected)
    res = search(q, problem_id=0, expected_answer=expected, d_max=5, k=3, n_candidates=2, verbose=True)
    print("\nResult summary:")
    print("Status:", res["status"])
    print("Final Answer:", res["final_answer"])
    print("Expected:", res["expected_answer"])
    print("Correct:", res["correct"])
    print("Calls:", res["calls"])
    print("Tokens:", res["tokens"])
