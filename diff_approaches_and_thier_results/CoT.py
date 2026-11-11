import json, re, os
from datetime import datetime
from llm import llm_call, stats
from validator import check_equation
from heuristic import R

def extract_json(raw):
    raw = raw.strip()
    matches = re.findall(r"\{.*?\}", raw, re.DOTALL)
    parsed = []
    for m in matches:
        try:
            parsed.append(json.loads(m))
        except json.JSONDecodeError:
            continue
    return parsed if parsed else None

def is_final(text):
    t = text.upper()
    return "FINAL ANSWER" in t or "FINAL" in t or "ANSWER IS" in t

def get_step(question, steps_so_far):
    if steps_so_far:
        formatted = []
        for i, s in enumerate(steps_so_far):
            eq_str = f" [Equation: {s['equation']}]" if s.get('equation') and s['equation'].lower() != 'none' else ""
            formatted.append(f"{i+1}. {s['step']}{eq_str}")
        steps_text = "\n".join(formatted)
    else:
        steps_text = "none yet"

    prompt = f"""Problem: {question}

Steps completed so far:
{steps_text}

Instructions:
1. Review the steps above. If the calculations already answer the question, your step MUST be "FINAL ANSWER: <number>":
   {{"step": "FINAL ANSWER: <number>", "equation": "none", "confidence": 1.0}}
2. Otherwise, give ONLY the next NEW arithmetic step. Never repeat any calculation already completed above.
3. The "equation" field MUST be a complete arithmetic calculation with computed result (e.g. "10 + 5 = 15"). If finished, set to "none".
4. "confidence" must be a float between 0.0 and 1.0.
5. Some problems require MULTIPLE conversions chained together. Before answering FINAL ANSWER, check: does your last calculated value directly answer the question, in the units the question asks for? If not, continue.

Reply ONLY as JSON, no other text:
{{"step": "<description of step or FINAL ANSWER: number>", "equation": "<evaluated equation or none>", "confidence": 0.95}}"""
    raw = llm_call(prompt, temp=0.1)
    return extract_json(raw)

def plain_cot(question, problem_id=None, expected_answer=None, d_max=8, verbose=True):
    steps_so_far = []
    final_answer = None
    status = "incomplete"
    depth = 0

    while depth < d_max:
        step_jsons = get_step(question, steps_so_far)
        if step_jsons is None:
            if verbose:
                print("Stopping: could not parse model output.")
            status = "parse_error"
            break

        stop = False
        for step_json in step_jsons:
            step_text = step_json.get("step", "").strip()
            equation = str(step_json.get("equation", "")).strip()

            if is_final(step_text):
                if verbose:
                    print(f"\n--- Depth {depth} ---\nStep: {step_text}\nDONE: Solved successfully!")
                steps_so_far.append({"step": step_text, "equation": equation, "confidence": R(step_json)})
                match = re.search(r"(-?\d+\.?\d*)", step_text)
                final_answer = float(match.group(1)) if match else None
                status = "done"
                stop = True
                break

            if any(s["step"].lower() == step_text.lower() for s in steps_so_far) or \
               (equation.lower() != "none" and any(s["equation"].lower() == equation.lower() for s in steps_so_far)):
                if verbose:
                    print(f"\nSTUCK: repeated equation '{equation}'. Marking as unsolved.")
                status = "stuck"
                stop = True
                break

            R_ = R(step_json)
            valid = check_equation(equation) if (equation and equation.lower() != "none") else "N/A"
            if verbose:
                print(f"\n--- Depth {depth} ---\nStep: {step_text}\nEquation: {equation} | Valid: {valid}\nR(s): {R_}")
            steps_so_far.append({"step": step_text, "equation": equation, "confidence": R_})
            depth += 1

        if stop:
            break
    else:
        status = "depth_limit"
        if verbose:
            print("\nHit depth limit without a final answer.")

    result = {
        "method": "plain_cot",
        "problem_id": problem_id,
        "question": question,
        "steps": steps_so_far,
        "final_answer": final_answer,
        "expected_answer": expected_answer,
        "correct": (final_answer == expected_answer) if expected_answer is not None else None,
        "status": status,
        "calls": stats["calls"],
        "tokens": stats["tokens"],
        "timestamp": datetime.now().isoformat(),
    }

    if verbose:
        print(f"\nCalls used: {stats['calls']} | Tokens used: {stats['tokens']}")

    return result

def run_batch(problems, log_path="logs/plain_cot.jsonl", verbose=True):
    os.makedirs("logs", exist_ok=True)
    open(log_path, "w").close()  # clear previous results for a clean run

    for i, (question, answer) in enumerate(problems):
        stats["calls"] = 0
        stats["tokens"] = 0
        if verbose:
            print(f"\n{'='*50}\nPROBLEM {i}: {question}\n{'='*50}")
        result = plain_cot(question, problem_id=i, expected_answer=answer, verbose=verbose)
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    from problems import PROBLEMS, HARD_PROBLEMS
    run_batch(PROBLEMS + HARD_PROBLEMS)