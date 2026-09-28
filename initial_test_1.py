import json, re, os
from datetime import datetime
from llm import llm_call, stats
from validator import check_equation
from heuristic import R, T, final

def extract_json(raw):
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

QUESTION = "You have 30 coins in dimes and quarters. Their total value is $5.10. Find the number of dimes you have."
D_MAX = 10 #guessed for now

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
2. Otherwise, give ONLY the next NEW arithmetic step (one at a time). Never repeat any calculation already completed above.
3. The "equation" field MUST be a complete arithmetic calculation with computed result (e.g. "10 + 5 = 15"). If finished, set to "none".
4. "confidence" must be a float between 0.0 and 1.0.

Reply ONLY as JSON, no other text:
{{"step": "<description of step or FINAL ANSWER: number>", "equation": "<evaluated equation or none>", "confidence": 0.95}}"""
    raw = llm_call(prompt, temp=0.1)
    print("\n[RAW MODEL OUTPUT]:", raw, "\n")
    parsed = extract_json(raw)
    if parsed is None:
        print("Bad JSON from model:", raw)
    return parsed

def solve(question, problem_id=None, expected_answer=None):
    steps_so_far = []
    final_answer = None

    for depth in range(D_MAX):
        step_json = get_step(question, steps_so_far)
        if step_json is None:
            print("Stopping: could not parse model output.")
            break

        step_text = step_json.get("step", "").strip()
        equation = str(step_json.get("equation", "")).strip()

        if is_final(step_text):
            print(f"\n--- Depth {depth} ---")
            print("Step:", step_text)
            print("Equation:", equation)
            print("\nDONE: Solved successfully!")
            steps_so_far.append({"step": step_text, "equation": equation, "confidence": R(step_json)})
            match = re.search(r"(-?\d+\.?\d*)", step_text)
            final_answer = float(match.group(1)) if match else None
            break

        if any(s["step"].lower() == step_text.lower() for s in steps_so_far) or \
           (equation.lower() != "none" and any(s["equation"].lower() == equation.lower() for s in steps_so_far)):
            print(f"\nCalculations complete; model reached final state on '{step_text}'.")
            break

        R_ = R(step_json)
        valid = check_equation(equation) if (equation and equation.lower() != "none") else "N/A"

        print(f"\n--- Depth {depth} ---")
        print("Step:", step_text)
        print("Equation:", equation, "| Valid:", valid)
        print("R(s):", R_)

        steps_so_far.append({"step": step_text, "equation": equation, "confidence": R_})
    else:
        print("\nHit depth limit without a final answer.")

    print("\nCalls used:", stats["calls"], "| Tokens used:", stats["tokens"])

    # --- silent logging, no printing ---
    result = {
        "problem_id": problem_id,
        "question": question,
        "steps": steps_so_far,
        "final_answer": final_answer,
        "expected_answer": expected_answer,
        "correct": (final_answer == expected_answer) if expected_answer is not None else None,
        "calls": stats["calls"],
        "tokens": stats["tokens"],
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/m1_results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    return result

solve(QUESTION, problem_id=0, expected_answer=21)