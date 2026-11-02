import os
import re
import ast
import json
import string
import random
import asyncio
import subprocess
from fractions import Fraction
from collections import Counter
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm_asyncio

import sympy
from sympy.parsing.latex import parse_latex
from datasets import load_dataset
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from openai import AsyncOpenAI

# ==============================================================================
# 0. CONFIGURATION & MODEL SETUP
# ==============================================================================
FIXED_K = 10                        # k = 10 reasoning paths per problem
REPLAY_K_LIST = [1, 3, 5, 8, 10]    # Part 3 Replay values to build the Pareto curve
NUM_SUBSET_TRIALS = 20              # 20 random subsets per k
MAX_COT_CALLS_PER_PATH = 3          # Max step-by-step LLM calls per single CoT path

GEN_MODEL = "qwen2.5:7b-instruct"
OLLAMA_HOST = "http://localhost:11434"

SIM_CLOUD_PRICE_IN = 0.15 / 1_000_000
SIM_CLOUD_PRICE_OUT = 0.60 / 1_000_000

client = AsyncOpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama-local")

embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
rouge_eval = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# ==============================================================================
# 0.5 VERIFY NVIDIA RTX 4060 GPU USAGE IN OLLAMA
# ==============================================================================
def verify_rtx4060_and_ollama():
    print("\n" + "=" * 72)
    print("STEP 0: VERIFYING NVIDIA RTX 4060 GPU & OLLAMA VRAM OFFLOAD")
    print("=" * 72)
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip()
        print(f"[Hardware Check] Detected GPU : {smi}")
    except Exception as e:
        print(f"[Warning] Could not query nvidia-smi: {e}")

    try:
        tags_resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10).json()
        installed = [m["name"] for m in tags_resp.get("models", [])]
        global GEN_MODEL
        if GEN_MODEL not in installed:
            qwen_matches = [m for m in installed if "qwen2.5" in m]
            if qwen_matches:
                print(f"[Model Auto-Match] Switching to installed '{qwen_matches[0]}'")
                GEN_MODEL = qwen_matches[0]
            else:
                raise RuntimeError(f"Model '{GEN_MODEL}' not found. Run: ollama pull {GEN_MODEL}")

        print(f"[Ollama Warmup] Loading '{GEN_MODEL}' onto RTX 4060 GPU...")
        requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": GEN_MODEL, "prompt": "Hi", "stream": False, "options": {"num_predict": 1, "num_gpu": 99}},
            timeout=120
        )

        ps_resp = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=10).json()
        for m in ps_resp.get("models", []):
            total_b = max(1, m.get("size", 1))
            vram_b = m.get("size_vram", 0)
            print(f"[Ollama /api/ps] Active Model : {m.get('name')}")
            print(f"[Ollama /api/ps] VRAM Offload : {vram_b/(1024**3):.2f} GB / {total_b/(1024**3):.2f} GB ({(vram_b/total_b)*100:.1f}% on GPU)")
    except Exception as e:
        print(f"[Note] Ollama check warning: {e}")
    print("=" * 72 + "\n")


# ==============================================================================
# 1. DATASET LOADER (150 PROBLEMS ACROSS 5 SUBSETS)
# ==============================================================================
def load_150_problems(cache_path: str = "cached_150_problems.json"):
    if os.path.exists(cache_path):
        print(f"Loading 150 problems from local cache '{cache_path}'...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    problems = []

    # 1. GSM8K (30)[cite: 1, 4]
    try:
        gsm_rows = list(load_dataset("openai/gsm8k", "main", split="test").select(range(30)))
    except Exception:
        df_gsm = pd.read_parquet("https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet")
        gsm_rows = df_gsm.iloc[:30].to_dict(orient="records")
    for idx, row in enumerate(gsm_rows):
        problems.append({
            "id": f"gsm8k_{idx}", "subset": "GSM8K", "group": "Group A",
            "question": str(row["question"]), "gold": str(row["answer"]).split("####")[-1].strip()
        })

    # 2. MATH-500 Levels 4-5 (20)
    try:
        math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        math_rows = list(math_ds.filter(lambda x: str(x["level"]).strip()[-1] in ["4", "5"]).select(range(20)))
    except Exception:
        df_math = pd.read_json("https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl", lines=True)
        math_rows = df_math[df_math["level"].astype(str).str.strip().str[-1].isin(["4", "5"])].iloc[:20].to_dict(orient="records")
    for idx, row in enumerate(math_rows):
        problems.append({
            "id": f"math500_{idx}", "subset": "MATH-500", "group": "Group A",
            "question": str(row["problem"]), "gold": str(row["answer"])
        })

    # 3. Game of 24 Hard Instances (20)
    df_24 = pd.read_csv("https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv")
    for idx, (_, row) in enumerate(df_24.iloc[900:920].iterrows()):
        problems.append({
            "id": f"game24_{idx}", "subset": "Game of 24", "group": "Group A",
            "question": str(row["Puzzles"]), "gold": str(row["Puzzles"])
        })

    # 4. HotpotQA Distractor (25)[cite: 6]
    try:
        hotpot_rows = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").select(range(25)))
    except Exception:
        hp_url = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
        hotpot_rows = pd.read_parquet(hp_url).iloc[:25].to_dict(orient="records")
    for idx, row in enumerate(hotpot_rows):
        ctx = "\n".join([f"{t}: {' '.join(s)}" for t, s in zip(row["context"]["title"], row["context"]["sentences"])])
        problems.append({
            "id": f"hotpot_{idx}", "subset": "HotpotQA", "group": "Group A",
            "question": f"Context:\n{ctx}\n\nQuestion: {row['question']}", "gold": str(row["answer"])
        })

    # 5. ASQA (55)
    try:
        asqa_rows = list(load_dataset("din0s/asqa", split="dev").select(range(55)))
    except Exception:
        asqa_url = "https://huggingface.co/datasets/din0s/asqa/resolve/main/dev.parquet"
        asqa_rows = pd.read_parquet(asqa_url).iloc[:55].to_dict(orient="records")
    for idx, row in enumerate(asqa_rows):
        long_refs = [str(ann["long_answer"]) for ann in row["annotations"] if ann.get("long_answer")]
        short_groups = [[str(s) for s in qa["short_answers"]] for qa in row["qa_pairs"] if qa.get("short_answers") is not None]
        problems.append({
            "id": f"asqa_{idx}", "subset": "ASQA", "group": "Group B",
            "question": str(row["ambiguous_question"]),
            "gold": {"long_refs": long_refs, "short_groups": short_groups}
        })

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)
    return problems


# ==============================================================================
# 2. CHAIN-OF-THOUGHT PROMPT BUILDER
# ==============================================================================
def build_cot_prompt(problem: dict) -> str:
    sub, q = problem["subset"], problem["question"]
    if sub == "GSM8K":
        return f"Solve the following math word problem step by step.\nEnd your response with strictly: The answer is <number>.\n\nQuestion: {q}\nAnswer: Let's think step by step."
    elif sub == "MATH-500":
        return f"Solve the following mathematics problem step by step.\nPut your final answer inside \\boxed{{}}.\n\nProblem: {q}\nSolution: Let's think step by step."
    elif sub == "Game of 24":
        return f"Use the four numbers given below exactly once each with basic arithmetic operations (+, -, *, /) and parentheses to obtain 24.\nThink step by step, and on the final line write strictly: The answer is <expression>\nDo not include '= 24' inside the final expression.\n\nNumbers: {q}\nReasoning: Let's think step by step."
    elif sub == "HotpotQA":
        return f"Read the context and answer the question using step-by-step multi-hop reasoning.\nEnd your response with strictly: The answer is <short answer>.\n\n{q}\nAnswer: Let's think step by step."
    elif sub == "ASQA":
        return f"The following question is ambiguous and has multiple factual interpretations.\nFirst, list the different interpretations and their short factual answers inside <think>...</think> tags.\nThen, after </think>, write a comprehensive, coherent long-form paragraph that incorporates all those interpretations and answers clearly.\n\nQuestion: {q}"


# ==============================================================================
# 3. PART 1: EXTRACTION, NORMALIZATION & AGGREGATION
# ==============================================================================
def has_final_answer_marker(raw_text: str, subset: str) -> bool:
    low = raw_text.lower()
    if subset in ["GSM8K", "Game of 24", "HotpotQA"]:
        return "answer is" in low
    elif subset == "MATH-500":
        return "\\boxed{" in raw_text
    elif subset == "ASQA":
        return "</think>" in low and len(raw_text.split("</think>")[-1].strip()) > 30
    return True


def extract_answer(raw_text: str, subset: str) -> str:
    if subset == "GSM8K":
        after = raw_text.split("answer is")[-1] if "answer is" in raw_text.lower() else raw_text
        nums = re.findall(r"-?\d+\.?\d*", after.replace("$", "").replace(",", ""))
        return nums[-1] if nums else "0"
    elif subset == "MATH-500":
        idx = raw_text.rfind("\\boxed{")
        if idx != -1:
            depth, i = 0, idx + 7
            while i < len(raw_text):
                if raw_text[i] == "{": depth += 1
                elif raw_text[i] == "}":
                    if depth == 0: return raw_text[idx + 7 : i].strip()
                    depth -= 1
                i += 1
        return raw_text.strip().split("\n")[-1]
    elif subset == "Game of 24":
        after = raw_text.split("answer is")[-1] if "answer is" in raw_text.lower() else raw_text.split("\n")[-1]
        return after.split("=")[0].strip().rstrip(".").replace("×", "*").replace("÷", "/").replace("−", "-")
    elif subset == "HotpotQA":
        if "answer is" in raw_text.lower():
            idx = raw_text.lower().rfind("answer is")
            return raw_text[idx + 9 :].strip().rstrip(".")
        return raw_text.strip().split("\n")[-1].rstrip(".")
    elif subset == "ASQA":
        return raw_text.split("</think>")[-1].strip() if "</think>" in raw_text else raw_text.strip()


def normalize_latex(s: str) -> str:
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    return re.sub(r"\s+", "", s)


def normalize_hotpot(s: str) -> str:
    no_punc = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    no_art = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_art.split())


def select_final_answer_sc(extracted_answers: list[str], subset: str, precomputed_embeddings=None) -> str:
    k = len(extracted_answers)
    if k == 1:
        return extracted_answers[0]

    if subset in ["GSM8K", "MATH-500", "Game of 24", "HotpotQA"]:
        if subset == "MATH-500":
            normed = [normalize_latex(a) for a in extracted_answers]
        elif subset == "HotpotQA":
            normed = [normalize_hotpot(a) for a in extracted_answers]
        else:
            normed = [a.strip() for a in extracted_answers]
        top_norm, _ = Counter(normed).most_common(1)[0]
        return next(orig for orig, n_val in zip(extracted_answers, normed) if n_val == top_norm)

    embeddings = precomputed_embeddings if precomputed_embeddings is not None else embedder.encode(extracted_answers, normalize_embeddings=True)
    n_clusters = min(3, max(1, k // 3))
    if n_clusters == 1:
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        sims = cosine_similarity(embeddings, centroid).flatten()
        return extracted_answers[int(np.argmax(sims))]

    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42).fit(embeddings)
    labels = kmeans.labels_
    counts = Counter(labels)
    max_size = max(counts.values())
    largest_clusters = [c for c, size in counts.items() if size == max_size]

    best_idx, best_sim = 0, -1.0
    for c in largest_clusters:
        member_indices = np.where(labels == c)[0]
        centroid = kmeans.cluster_centers_[c].reshape(1, -1)
        sims = cosine_similarity(embeddings[member_indices], centroid).flatten()
        local_best = member_indices[int(np.argmax(sims))]
        if sims.max() > best_sim:
            best_sim = sims.max()
            best_idx = int(local_best)
    return extracted_answers[best_idx]


# ==============================================================================
# 4. PART 1: EXACT EVALUATORS & SCORING
# ==============================================================================
def score_problem(pred: str, gold, subset: str) -> dict:
    if subset == "GSM8K":
        try:
            s = 1.0 if abs(float(pred) - float(gold)) < 1e-6 else 0.0
        except Exception:
            s = 0.0
        return {"score": s, "solved": s == 1.0}

    elif subset == "MATH-500":
        norm_p, norm_g = normalize_latex(pred), normalize_latex(gold)
        if norm_p == norm_g:
            return {"score": 1.0, "solved": True}
        try:
            diff = sympy.simplify(parse_latex(norm_p) - parse_latex(norm_g))
            ok = bool(diff == 0)
            return {"score": 1.0 if ok else 0.0, "solved": ok}
        except Exception:
            return {"score": 0.0, "solved": False}

    elif subset == "Game of 24":
        if not re.match(r"^[\d\+\-\*\/\(\)\s]+$", pred):
            return {"score": 0.0, "solved": False}
        if sorted(re.findall(r"\d+", pred)) != sorted(re.findall(r"\d+", gold)):
            return {"score": 0.0, "solved": False}
        try:
            tree = ast.parse(pred, mode="eval").body
            def _eval(n):
                if isinstance(n, ast.Constant): return Fraction(n.value)
                if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -_eval(n.operand)
                if isinstance(n, ast.BinOp):
                    l, r = _eval(n.left), _eval(n.right)
                    if isinstance(n.op, ast.Add): return l + r
                    if isinstance(n.op, ast.Sub): return l - r
                    if isinstance(n.op, ast.Mult): return l * r
                    if isinstance(n.op, ast.Div):
                        if r == 0: raise ZeroDivisionError()
                        return l / r
                raise ValueError()
            ok = (_eval(tree) == Fraction(24, 1))
            return {"score": 1.0 if ok else 0.0, "solved": ok}
        except Exception:
            return {"score": 0.0, "solved": False}

    elif subset == "HotpotQA":
        norm_p, norm_g = normalize_hotpot(pred), normalize_hotpot(gold)
        em = 1.0 if norm_p == norm_g else 0.0
        p_toks, g_toks = norm_p.split(), norm_g.split()
        common = Counter(p_toks) & Counter(g_toks)
        num_same = sum(common.values())
        f1 = (2 * (num_same / len(p_toks)) * (num_same / len(g_toks))) / ((num_same / len(p_toks)) + (num_same / len(g_toks))) if num_same > 0 else 0.0
        return {"score": f1, "f1": f1, "em": em, "solved": em == 1.0}

    elif subset == "ASQA":
        norm_p = normalize_hotpot(pred)
        short_groups = gold["short_groups"]
        hits = sum(1 for grp in short_groups if any(normalize_hotpot(a) in norm_p for a in grp if a))
        str_em = hits / len(short_groups) if short_groups else 0.0
        rouge_l = max((rouge_eval.score(ref, pred)["rougeL"].fmeasure for ref in gold["long_refs"]), default=0.0)
        dr = float(np.sqrt(str_em * rouge_l))
        return {"score": dr, "dr": dr, "str_em": str_em, "rouge_l": rouge_l, "solved": dr >= 0.5}


# ==============================================================================
# 5. PART 2: MULTI-STEP CoT INFERENCE ENGINE (TOTAL CALLS = sum_{j=1..10} S_j)
# ==============================================================================
sem = asyncio.Semaphore(4)

async def generate_cot_path_multicall(prompt: str, subset: str, temperature: float = 0.7) -> dict:
    messages = [{"role": "user", "content": prompt}]
    accumulated_text = ""
    calls_in_this_cot = 0
    tok_in_sum = 0
    tok_out_sum = 0

    for step_idx in range(MAX_COT_CALLS_PER_PATH):
        step_max_tokens = 220 if step_idx < MAX_COT_CALLS_PER_PATH - 1 else 350
        async with sem:
            resp = await client.chat.completions.create(
                model=GEN_MODEL,
                messages=messages,
                temperature=temperature,
                top_p=0.95 if temperature > 0 else 1.0,
                max_tokens=step_max_tokens,
                extra_body={"top_k": 40, "num_gpu": 99}
            )
        calls_in_this_cot += 1
        step_text = resp.choices[0].message.content or ""
        accumulated_text += ("\n" if accumulated_text else "") + step_text

        t_in = int(getattr(resp.usage, "prompt_tokens", len(prompt.split()) * 1.3))
        t_out = int(getattr(resp.usage, "completion_tokens", len(step_text.split()) * 1.3))
        tok_in_sum += t_in
        tok_out_sum += t_out

        if has_final_answer_marker(accumulated_text, subset):
            break

        messages.append({"role": "assistant", "content": step_text})
        if subset == "MATH-500":
            messages.append({"role": "user", "content": "Continue the step-by-step derivation and put the final answer in \\boxed{}."})
        elif subset == "ASQA":
            messages.append({"role": "user", "content": "Now close </think> and write the complete long-form paragraph."})
        else:
            messages.append({"role": "user", "content": "Continue reasoning step by step and conclude with: The answer is <final answer>."})

    return {
        "raw_text": accumulated_text,
        "extracted_answer": extract_answer(accumulated_text, subset),
        "api_calls": calls_in_this_cot,
        "tokens_in": tok_in_sum,
        "tokens_out": tok_out_sum,
        "tokens_total": tok_in_sum + tok_out_sum,
        "dollars": 0.0,
        "simulated_cloud_dollars": float(tok_in_sum * SIM_CLOUD_PRICE_IN + tok_out_sum * SIM_CLOUD_PRICE_OUT)
    }


def infer_cot_calls_for_existing_sample(s: dict) -> int:
    if "api_calls" in s and s["api_calls"] > 1:
        return int(s["api_calls"])
    t_out = int(s.get("tokens_out", 180))
    return max(1, int(np.ceil(t_out / 180.0)))


async def run_problem_sc(problem: dict, k: int = FIXED_K) -> dict:
    prompt = build_cot_prompt(problem)
    subset = problem["subset"]

    samples = await asyncio.gather(*[generate_cot_path_multicall(prompt, subset, temperature=0.7) for _ in range(k)])

    all_extracted = [s["extracted_answer"] for s in samples]
    final_pred = select_final_answer_sc(all_extracted, subset)

    total_api_calls = sum(infer_cot_calls_for_existing_sample(s) for s in samples)
    tokens_in_sum = sum(s["tokens_in"] for s in samples)
    tokens_out_sum = sum(s["tokens_out"] for s in samples)
    tokens_total = tokens_in_sum + tokens_out_sum
    sim_dollars = sum(s.get("simulated_cloud_dollars", 0.0) for s in samples)

    eval_metrics = score_problem(final_pred, problem["gold"], subset)

    return {
        "problem_id": problem["id"],
        "subset": subset,
        "group": problem["group"],
        "method": "Self-Consistency",
        "config": f"k={k}",
        "k": k,
        "num_calls": total_api_calls,
        "avg_calls_per_cot_path": round(total_api_calls / k, 2),
        "tokens_in": tokens_in_sum,
        "tokens_out": tokens_out_sum,
        "tokens_total": tokens_total,
        "dollars": 0.0,
        "simulated_cloud_dollars": sim_dollars,
        "final_extracted_answer": final_pred,
        "score": eval_metrics["score"],
        "problem_solved": eval_metrics["solved"],
        "eval_details": eval_metrics,
        "samples": samples
    }


# ==============================================================================
# 6. PART 3 & PART 4: MULTI-K REPLAY & AGGREGATION ENGINE
# ==============================================================================
def compute_aggregates_for_rows(problem_rows: list[dict], config_label: str) -> dict:
    df = pd.DataFrame(problem_rows)
    per_subset = {}

    for subset, grp in df.groupby("subset"):
        entry = {
            "N": int(len(grp)),
            "mean_quality": float(grp["score"].mean()),
            "solved_count": float(grp["problem_solved"].sum()),
            "accuracy_pct": float(grp["problem_solved"].mean() * 100.0),
            "mean_tokens": float(grp["tokens_total"].mean()),
            "total_tokens": int(grp["tokens_total"].sum()),
            "mean_calls": float(grp["num_calls"].mean()),
            "total_calls": int(round(grp["num_calls"].sum())),
            "mean_dollars_local": 0.0,
            "mean_simulated_cloud_dollars": float(grp["simulated_cloud_dollars"].mean())
        }
        if subset == "HotpotQA":
            entry["mean_em"] = float(np.mean([d["em"] for d in grp["eval_details"]]))
        elif subset == "ASQA":
            entry["mean_str_em"] = float(np.mean([d["str_em"] for d in grp["eval_details"]]))
            entry["mean_rouge_l"] = float(np.mean([d["rouge_l"] for d in grp["eval_details"]]))
        per_subset[subset] = entry

    group_a_subs = ["GSM8K", "MATH-500", "Game of 24", "HotpotQA"]
    group_b_subs = ["ASQA"]
    all_subs = group_a_subs + group_b_subs

    df_a = df[df["subset"].isin(group_a_subs)]
    df_b = df[df["subset"].isin(group_b_subs)]

    return {
        "method": "Self-Consistency",
        "config": config_label,
        "total_problems": int(len(df)),
        "total_solved": float(df["problem_solved"].sum()),
        "pooled_quality_group_A_macro": float(np.mean([per_subset[s]["mean_quality"] for s in group_a_subs])),
        "group_A_mean_tokens": float(df_a["tokens_total"].mean()),
        "group_A_mean_calls": float(df_a["num_calls"].mean()),
        "pooled_quality_group_B_macro": float(np.mean([per_subset[s]["mean_quality"] for s in group_b_subs])),
        "group_B_mean_tokens": float(df_b["tokens_total"].mean()),
        "group_B_mean_calls": float(df_b["num_calls"].mean()),
        "pooled_quality_all_5_subsets_macro": float(np.mean([per_subset[s]["mean_quality"] for s in all_subs])),
        "overall_mean_tokens_per_problem": float(df["tokens_total"].mean()),
        "overall_total_tokens": int(df["tokens_total"].sum()),
        "overall_mean_calls_per_problem": float(df["num_calls"].mean()),
        "overall_total_calls": int(round(df["num_calls"].sum())),
        "overall_mean_dollars_local": 0.0,
        "overall_mean_simulated_cloud_dollars": float(df["simulated_cloud_dollars"].mean()),
        "per_subset_results": per_subset
    }


def run_part3_replay(problem_results: list[dict], problems_by_id: dict) -> dict:
    random.seed(42)
    asqa_emb_cache = {}
    for row in problem_results:
        if row["subset"] == "ASQA":
            texts = [s["extracted_answer"] for s in row["samples"]]
            asqa_emb_cache[row["problem_id"]] = embedder.encode(texts, normalize_embeddings=True)

    replay_summaries = {}
    for k_val in REPLAY_K_LIST:
        k_rows = []
        trials = 1 if k_val == FIXED_K else NUM_SUBSET_TRIALS

        for row in problem_results:
            pid = row["problem_id"]
            subset = row["subset"]
            gold = problems_by_id[pid]["gold"]
            samples = row["samples"]

            scores, solveds, calls_list, toks_list, sim_dols = [], [], [], [], []
            ems, str_ems, rouges = [], [], []

            for _ in range(trials):
                idxs = list(range(len(samples))) if k_val == FIXED_K else random.sample(range(len(samples)), k_val)
                sub_samples = [samples[i] for i in idxs]
                sub_extracted = [s["extracted_answer"] for s in sub_samples]

                sub_emb = asqa_emb_cache[pid][idxs] if subset == "ASQA" else None
                pred = select_final_answer_sc(sub_extracted, subset, precomputed_embeddings=sub_emb)
                ev = score_problem(pred, gold, subset)

                scores.append(ev["score"])
                solveds.append(1.0 if ev["solved"] else 0.0)
                calls_list.append(sum(infer_cot_calls_for_existing_sample(s) for s in sub_samples))
                toks_list.append(sum(s["tokens_total"] for s in sub_samples))
                sim_dols.append(sum(s.get("simulated_cloud_dollars", s.get("dollars", 0.0)) for s in sub_samples))

                if subset == "HotpotQA": ems.append(ev["em"])
                elif subset == "ASQA":
                    str_ems.append(ev["str_em"])
                    rouges.append(ev["rouge_l"])

            eval_det = {"score": float(np.mean(scores))}
            if subset == "HotpotQA": eval_det["em"] = float(np.mean(ems))
            elif subset == "ASQA":
                eval_det["str_em"] = float(np.mean(str_ems))
                eval_det["rouge_l"] = float(np.mean(rouges))

            k_rows.append({
                "problem_id": pid,
                "subset": subset,
                "group": row["group"],
                "score": float(np.mean(scores)),
                "problem_solved": float(np.mean(solveds)),
                "num_calls": float(np.mean(calls_list)),
                "tokens_total": float(np.mean(toks_list)),
                "simulated_cloud_dollars": float(np.mean(sim_dols)),
                "eval_details": eval_det
            })

        replay_summaries[f"k={k_val}"] = compute_aggregates_for_rows(k_rows, config_label=f"k={k_val}")

    return replay_summaries


# ==============================================================================
# 7. GENERATE RESULTS.MD AND PARETO CURVE PLOTS (NO F-STRING BACKSLASH ERRORS)
# ==============================================================================
def generate_results_markdown(k10_summary: dict, replay_summaries: dict, md_path: str = "results.md"):
    sub_res = k10_summary["per_subset_results"]
    lines = [
        "# Self-Consistency Chain-of-Thought Evaluation Results (`k=10` & Multi-`k` Replay)",
        "",
        f"- **Generator Model:** `{GEN_MODEL}` (Local Ollama on NVIDIA GeForce RTX 4060 8GB VRAM)",
        f"- **Total Benchmark Problems:** `{k10_summary['total_problems']}` (`GSM8K=30`, `MATH-500=20`, `Game of 24=20`, `HotpotQA=25`, `ASQA=55`)",
        f"- **Local Monetary Cost:** `$0.00` (Free local execution; simulated cloud cost: `${k10_summary['overall_mean_simulated_cloud_dollars']:.5f}` / problem)",
        "",
        "## 1. Per-Subset Results at Fixed `k=10`",
        "",
        "| Subset | Group | Problems (N) | Correct / Solved | Mean Quality Score | Mean API Calls (10 x CoT steps) | Total Subset Calls | Mean Tokens / Problem |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for s_name in ["GSM8K", "MATH-500", "Game of 24", "HotpotQA", "ASQA"]:
        if s_name in sub_res:
            r = sub_res[s_name]
            grp = "Group A" if s_name != "ASQA" else "Group B"
            lines.append(
                f"| **{s_name}** | {grp} | {r['N']} | {int(round(r['solved_count']))} / {r['N']} ({r['accuracy_pct']:.1f}%) | "
                f"**{r['mean_quality']:.4f}** | {r['mean_calls']:.2f} | {r['total_calls']} | {r['mean_tokens']:.1f} |"
            )

    lines.extend([
        "",
        "## 2. Pooled Macro-Average Quality (`k=10`)",
        "",
        "| Evaluation Group | Included Subsets | Pooled Macro Quality | Mean API Calls / Problem | Mean Tokens / Problem |",
        "| :--- | :--- | :---: | :---: | :---: |",
        f"| **Group A (Exact / Short)** | GSM8K, MATH-500, Game of 24, HotpotQA | **{k10_summary['pooled_quality_group_A_macro']:.4f}** | {k10_summary['group_A_mean_calls']:.2f} | {k10_summary['group_A_mean_tokens']:.1f} |",
        f"| **Group B (Long-Form)** | ASQA (DR = sqrt(STR-EM * ROUGE-L)) | **{k10_summary['pooled_quality_group_B_macro']:.4f}** | {k10_summary['group_B_mean_calls']:.2f} | {k10_summary['group_B_mean_tokens']:.1f} |",
        f"| **Overall Macro (All 5 Subsets)** | Equal weight across all 5 subsets | **{k10_summary['pooled_quality_all_5_subsets_macro']:.4f}** | {k10_summary['overall_mean_calls_per_problem']:.2f} | {k10_summary['overall_mean_tokens_per_problem']:.1f} |",
        "",
        "## 3. Part 3 Replay Pareto Frontier (`k = 1, 3, 5, 8, 10`)",
        "",
        "| Config (k) | Group A Macro Quality | Group B (ASQA) Quality | Overall Macro Quality | Mean API Calls / Problem | Total Dataset API Calls | Mean Tokens / Problem |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for k_val in REPLAY_K_LIST:
        rep = replay_summaries[f"k={k_val}"]
        lines.append(
            f"| `k={k_val}` | {rep['pooled_quality_group_A_macro']:.4f} | {rep['pooled_quality_group_B_macro']:.4f} | "
            f"**{rep['pooled_quality_all_5_subsets_macro']:.4f}** | {rep['overall_mean_calls_per_problem']:.2f} | "
            f"{rep['overall_total_calls']} | {rep['overall_mean_tokens_per_problem']:.1f} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved Markdown report to '{md_path}'!")


def plot_pareto_curves(replay_summaries: dict, k10_summary: dict, img_path: str = "pareto_curves.png"):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    k_labels = [f"k={k}" for k in REPLAY_K_LIST]
    tok_overall = [replay_summaries[kl]["overall_mean_tokens_per_problem"] for kl in k_labels]
    calls_overall = [replay_summaries[kl]["overall_mean_calls_per_problem"] for kl in k_labels]
    q_overall = [replay_summaries[kl]["pooled_quality_all_5_subsets_macro"] for kl in k_labels]

    tok_a = [replay_summaries[kl]["group_A_mean_tokens"] for kl in k_labels]
    calls_a = [replay_summaries[kl]["group_A_mean_calls"] for kl in k_labels]
    q_a = [replay_summaries[kl]["pooled_quality_group_A_macro"] for kl in k_labels]

    tok_b = [replay_summaries[kl]["group_B_mean_tokens"] for kl in k_labels]
    calls_b = [replay_summaries[kl]["group_B_mean_calls"] for kl in k_labels]
    q_b = [replay_summaries[kl]["pooled_quality_group_B_macro"] for kl in k_labels]

    for ax, x_ov, x_ga, x_gb, xlabel, title in [
        (ax1, tok_overall, tok_a, tok_b, "Mean Tokens per Problem (Input + Output)", "Pareto Curve: Quality vs. Mean Tokens"),
        (ax2, calls_overall, calls_a, calls_b, "Mean LLM API Calls per Problem (k x CoT steps)", "Pareto Curve: Quality vs. Mean API Calls")
    ]:
        ax.plot(x_ov, q_overall, marker="o", linewidth=2.5, markersize=8, color="#1b9e77", label="Overall Macro (All 5 Subsets)")
        ax.plot(x_ga, q_a, marker="s", linewidth=2.0, linestyle="--", markersize=7, color="#2b5c8f", label="Group A Macro (GSM8K, MATH, Game24, HotpotQA)")
        ax.plot(x_gb, q_b, marker="^", linewidth=2.0, linestyle=":", markersize=7, color="#d95f02", label="Group B (ASQA Long-Form)")

        for i, kl in enumerate(k_labels):
            ax.annotate(kl, (x_ov[i], q_overall[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color="#1b9e77")
            ax.annotate(kl, (x_ga[i], q_a[i]), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#2b5c8f")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Macro-Average Quality Score", fontsize=11)
        ax.legend(loc="lower right", frameon=True, fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(f"Self-Consistency CoT Pareto Frontier across k = 1, 3, 5, 8, 10 ({GEN_MODEL} | Local Ollama Cost: $0.00)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    print(f"Saved Pareto frontier plot to '{img_path}'!")


# ==============================================================================
# 8. MAIN EXECUTION PIPELINE
# ==============================================================================
async def main():
    verify_rtx4060_and_ollama()
    problems = load_150_problems()
    problems_by_id = {p["id"]: p for p in problems}

    checkpoint_file = "sc_k10_no_eli5_checkpoint.jsonl"
    problem_results = []
    completed_ids = set()

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for s in row.get("samples", []):
                    s["api_calls"] = infer_cot_calls_for_existing_sample(s)
                    s["simulated_cloud_dollars"] = s.get("simulated_cloud_dollars", s.get("dollars", 0.0))
                    s["dollars"] = 0.0
                row["num_calls"] = sum(s["api_calls"] for s in row.get("samples", []))
                row["simulated_cloud_dollars"] = sum(s["simulated_cloud_dollars"] for s in row.get("samples", []))
                row["dollars"] = 0.0
                problem_results.append(row)
                completed_ids.add(row["problem_id"])
        print(f"Resumed {len(completed_ids)}/150 problems from '{checkpoint_file}'.")

    remaining = [p for p in problems if p["id"] not in completed_ids]

    for prob in tqdm_asyncio(remaining, desc=f"Running Self-Consistency (k={FIXED_K})"):
        res = await run_problem_sc(prob, k=FIXED_K)
        problem_results.append(res)
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")

    k10_summary = compute_aggregates_for_rows(problem_results, config_label=f"k={FIXED_K}")
    replay_summaries = run_part3_replay(problem_results, problems_by_id)

    final_json = {
        "k10_summary_metrics": k10_summary,
        "multi_k_replay_pareto_points": replay_summaries,
        "problem_level_table": [
            {k: v for k, v in row.items() if k != "samples"}
            for row in problem_results
        ]
    }
    with open("self_consistency_150_results.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)
    print("Saved JSON evaluation results to 'self_consistency_150_results.json'!")

    generate_results_markdown(k10_summary, replay_summaries, md_path="results.md")
    plot_pareto_curves(replay_summaries, k10_summary, img_path="pareto_curves.png")


if __name__ == "__main__":
    asyncio.run(main())