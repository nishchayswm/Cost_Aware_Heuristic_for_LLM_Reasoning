import os
import re
import ast
import json
import string
import asyncio
import subprocess
from fractions import Fraction
from collections import Counter
import requests
import numpy as np
import pandas as pd
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
FIXED_K = 10  # Sample k=10 paths per problem
GEN_MODEL = "qwen2.5:7b-instruct"  # Change to "qwen2.5:7b" if that is the tag in `ollama list`
OLLAMA_HOST = "http://localhost:11434"

PRICE_IN = 0.15 / 1_000_000
PRICE_OUT = 0.60 / 1_000_000

client = AsyncOpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama-local")

# Force SentenceTransformer onto CPU so 100% of your 8GB RTX 4060 VRAM stays free for Ollama
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
rouge_eval = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# ==============================================================================
# 0.5 VERIFY NVIDIA RTX 4060 GPU USAGE IN OLLAMA
# ==============================================================================
def verify_rtx4060_and_ollama():
    """
    1. Checks hardware GPU via nvidia-smi.
    2. Warms up Ollama with GEN_MODEL.
    3. Queries Ollama's /api/ps endpoint to verify 100% GPU VRAM offloading.
    """
    print("\n" + "=" * 70)
    print("STEP 0: VERIFYING NVIDIA RTX 4060 GPU & OLLAMA VRAM OFFLOAD")
    print("=" * 70)

    # 1. Check nvidia-smi
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip()
        print(f"[Hardware Check] Detected GPU : {smi}")
    except Exception as e:
        print(f"[Warning] Could not run nvidia-smi: {e}")

    # 2. Check installed Ollama models & warm up the model so it loads into VRAM
    try:
        tags_resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10).json()
        installed = [m["name"] for m in tags_resp.get("models", [])]
        global GEN_MODEL
        if GEN_MODEL not in installed:
            # Auto-fallback if user pulled "qwen2.5:7b" instead of "qwen2.5:7b-instruct"
            qwen_matches = [m for m in installed if "qwen2.5" in m]
            if qwen_matches:
                print(f"[Model Auto-Match] '{GEN_MODEL}' not found, switching to installed '{qwen_matches[0]}'")
                GEN_MODEL = qwen_matches[0]
            else:
                raise RuntimeError(
                    f"Model '{GEN_MODEL}' is not pulled in Ollama! Installed models: {installed}.\n"
                    f"Run: ollama pull {GEN_MODEL}"
                )

        print(f"[Ollama Warmup] Loading '{GEN_MODEL}' into RTX 4060 VRAM...")
        requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": GEN_MODEL, "prompt": "Hi", "stream": False, "options": {"num_predict": 1, "num_gpu": 99}},
            timeout=120
        )

        # 3. Query /api/ps to inspect active VRAM allocation
        ps_resp = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=10).json()
        running_models = ps_resp.get("models", [])
        for m in running_models:
            total_bytes = m.get("size", 1)
            vram_bytes = m.get("size_vram", 0)
            gpu_pct = (vram_bytes / total_bytes) * 100.0
            vram_gb = vram_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            print(f"[Ollama /api/ps] Active Model : {m.get('name')}")
            print(f"[Ollama /api/ps] VRAM Offload : {vram_gb:.2f} GB / {total_gb:.2f} GB ({gpu_pct:.1f}% on GPU)")
            if gpu_pct > 90:
                print("[SUCCESS] Ollama is running primarily on your RTX 4060 GPU!")
            elif gpu_pct > 0:
                print("[PARTIAL GPU] Model is partially split between RTX 4060 VRAM and System RAM.")
            else:
                print("[WARNING] 0% VRAM used! Ollama is running on CPU. Check NVIDIA drivers / restart Ollama.")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Ollama at {OLLAMA_HOST}. Is `ollama serve` running? Error: {e}")
    print("=" * 70 + "\n")


# ==============================================================================
# 1. DATASET LOADER (150 PROBLEMS ACROSS 5 SUBSETS - BULLETPROOF)
# ==============================================================================
def load_150_problems(cache_path: str = "cached_150_problems.json"):
    """
    Loads the exact 150 problems (GSM8K=30, MATH-500=20, Game24=20, HotpotQA=25, ASQA=55)
    and caches them locally so HuggingFace Hub is only queried once.
    """
    if os.path.exists(cache_path):
        print(f"Loading 150 problems from local cache '{cache_path}'...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    problems = []

    # 1. GSM8K (N = 30)
    print(" -> Downloading GSM8K (30 problems)...")
    try:
        gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(30))
        gsm_rows = list(gsm)
    except Exception:
        df_gsm = pd.read_parquet("https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet")
        gsm_rows = df_gsm.iloc[:30].to_dict(orient="records")

    for idx, row in enumerate(gsm_rows):
        gold = str(row["answer"]).split("####")[-1].strip()
        problems.append({
            "id": f"gsm8k_{idx}", "subset": "GSM8K", "group": "Group A",
            "question": str(row["question"]), "gold": gold
        })

    # 2. MATH-500 Levels 4-5 (N = 20)
    print(" -> Downloading MATH-500 Levels 4-5 (20 problems)...")
    try:
        math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        math_hard = math_ds.filter(lambda x: str(x["level"]).strip()[-1] in ["4", "5"]).select(range(20))
        math_rows = list(math_hard)
    except Exception:
        df_math = pd.read_json("https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl", lines=True)
        df_math = df_math[df_math["level"].astype(str).str.strip().str[-1].isin(["4", "5"])].iloc[:20]
        math_rows = df_math.to_dict(orient="records")

    for idx, row in enumerate(math_rows):
        problems.append({
            "id": f"math500_{idx}", "subset": "MATH-500", "group": "Group A",
            "question": str(row["problem"]), "gold": str(row["answer"])
        })

    # 3. Game of 24 Hard Instances (N = 20, ranks 901-920 from ToT repo)
    print(" -> Downloading Game of 24 Hard Instances (20 problems)...")
    tot_url = "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv"
    df_24 = pd.read_csv(tot_url)
    for idx, (_, row) in enumerate(df_24.iloc[900:920].iterrows()):
        problems.append({
            "id": f"game24_{idx}", "subset": "Game of 24", "group": "Group A",
            "question": str(row["Puzzles"]), "gold": str(row["Puzzles"])
        })

    # 4. HotpotQA Distractor (N = 25) - Direct Parquet fallback bypasses HfUriError completely!
    print(" -> Downloading HotpotQA Distractor (25 problems)...")
    try:
        hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").select(range(25))
        hotpot_rows = list(hotpot)
    except Exception:
        hp_url = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
        df_hp = pd.read_parquet(hp_url).iloc[:25]
        hotpot_rows = df_hp.to_dict(orient="records")

    for idx, row in enumerate(hotpot_rows):
        titles = row["context"]["title"]
        sents = row["context"]["sentences"]
        ctx = "\n".join([f"{t}: {' '.join(s)}" for t, s in zip(titles, sents)])
        problems.append({
            "id": f"hotpot_{idx}", "subset": "HotpotQA", "group": "Group A",
            "question": f"Context:\n{ctx}\n\nQuestion: {row['question']}",
            "gold": str(row["answer"])
        })

    # 5. ASQA (N = 55)
    print(" -> Downloading ASQA (55 problems)...")
    try:
        asqa = load_dataset("din0s/asqa", split="dev").select(range(55))
        asqa_rows = list(asqa)
    except Exception:
        asqa_url = "https://huggingface.co/datasets/din0s/asqa/resolve/main/dev.parquet"
        df_asqa = pd.read_parquet(asqa_url).iloc[:55]
        asqa_rows = df_asqa.to_dict(orient="records")

    for idx, row in enumerate(asqa_rows):
        anns = row["annotations"]
        qas = row["qa_pairs"]
        long_refs = [str(ann["long_answer"]) for ann in anns if ann.get("long_answer")]
        short_groups = [[str(s) for s in qa["short_answers"]] for qa in qas if qa.get("short_answers") is not None]
        problems.append({
            "id": f"asqa_{idx}", "subset": "ASQA", "group": "Group B",
            "question": str(row["ambiguous_question"]),
            "gold": {"long_refs": long_refs, "short_groups": short_groups}
        })

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)
    print(f"Successfully loaded and cached {len(problems)} problems to '{cache_path}'!")
    return problems


# ==============================================================================
# 2. CHAIN-OF-THOUGHT PROMPT BUILDER
# ==============================================================================
def build_cot_prompt(problem: dict) -> str:
    """Builds Chain-of-Thought prompts with strict formatting targets for extraction."""
    sub = problem["subset"]
    q = problem["question"]

    if sub == "GSM8K":
        return (
            "Solve the following math word problem step by step.\n"
            "End your response with strictly: The answer is <number>.\n\n"
            f"Question: {q}\nAnswer: Let's think step by step."
        )
    elif sub == "MATH-500":
        return (
            "Solve the following mathematics problem step by step.\n"
            "Put your final answer inside \\boxed{}.\n\n"
            f"Problem: {q}\nSolution: Let's think step by step."
        )
    elif sub == "Game of 24":
        return (
            "Use the four numbers given below exactly once each with basic arithmetic operations "
            "(+, -, *, /) and parentheses to obtain 24.\n"
            "Think step by step, and on the final line write strictly: The answer is <expression>\n"
            "Do not include '= 24' inside the final expression.\n\n"
            f"Numbers: {q}\nReasoning: Let's think step by step."
        )
    elif sub == "HotpotQA":
        return (
            "Read the context and answer the question using step-by-step multi-hop reasoning.\n"
            "End your response with strictly: The answer is <short answer>.\n\n"
            f"{q}\nAnswer: Let's think step by step."
        )
    elif sub == "ASQA":
        return (
            "The following question is ambiguous and has multiple factual interpretations.\n"
            "First, list the different interpretations and their short factual answers "
            "inside <think>...</think> tags.\n"
            "Then, after </think>, write a comprehensive, coherent long-form paragraph that "
            "incorporates all those interpretations and answers clearly.\n\n"
            f"Question: {q}"
        )


# ==============================================================================
# 3. PART 1: ANSWER EXTRACTION & SELF-CONSISTENCY AGGREGATION
# ==============================================================================
def extract_answer(raw_text: str, subset: str) -> str:
    """Extracts the final answer from the raw CoT output according to Part 1 rules."""
    if subset == "GSM8K":
        after = raw_text.split("answer is")[-1] if "answer is" in raw_text.lower() else raw_text
        after_clean = after.replace("$", "").replace(",", "")
        nums = re.findall(r"-?\d+\.?\d*", after_clean)
        return nums[-1] if nums else "0"

    elif subset == "MATH-500":
        idx = raw_text.rfind("\\boxed{")
        if idx != -1:
            depth, i = 0, idx + 7
            while i < len(raw_text):
                if raw_text[i] == "{":
                    depth += 1
                elif raw_text[i] == "}":
                    if depth == 0:
                        return raw_text[idx + 7 : i].strip()
                    depth -= 1
                i += 1
        return raw_text.strip().split("\n")[-1]

    elif subset == "Game of 24":
        after = raw_text.split("answer is")[-1] if "answer is" in raw_text.lower() else raw_text.split("\n")[-1]
        expr = after.split("=")[0].strip().rstrip(".")
        expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-")
        return expr

    elif subset == "HotpotQA":
        if "answer is" in raw_text.lower():
            idx = raw_text.lower().rfind("answer is")
            return raw_text[idx + 9 :].strip().rstrip(".")
        return raw_text.strip().split("\n")[-1].rstrip(".")

    elif subset == "ASQA":
        if "</think>" in raw_text:
            return raw_text.split("</think>")[-1].strip()
        return raw_text.strip()


def normalize_latex(s: str) -> str:
    """Normalises LaTeX (\dfrac -> \frac, drop \left/\right, strip spaces)."""
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    return re.sub(r"\s+", "", s)


def normalize_hotpot(s: str) -> str:
    """Official HotpotQA normalisation: lowercase, strip punctuation and articles."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def select_final_answer_sc(extracted_answers: list[str], subset: str) -> str:
    """
    - Group A (GSM8K, MATH-500, Game of 24, HotpotQA): Unweighted majority vote.
    - Group B (ASQA): Embed all k answers, cluster them, and return the branch
      closest to the centroid of the largest cluster.
    """
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
        for orig, n_val in zip(extracted_answers, normed):
            if n_val == top_norm:
                return orig

    # ASQA: Embed -> K-Means Cluster -> Nearest to Centroid of Largest Cluster
    embeddings = embedder.encode(extracted_answers, normalize_embeddings=True)
    n_clusters = min(3, max(1, k // 3))
    if n_clusters == 1:
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        sims = cosine_similarity(embeddings, centroid).flatten()
        return extracted_answers[int(np.argmax(sims))]

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(embeddings)
    labels = kmeans.labels_

    counts = Counter(labels)
    max_size = max(counts.values())
    largest_clusters = [c for c, size in counts.items() if size == max_size]

    best_idx = 0
    best_sim = -1.0
    for c in largest_clusters:
        member_indices = np.where(labels == c)[0]
        centroid = kmeans.cluster_centers_[c].reshape(1, -1)
        sims = cosine_similarity(embeddings[member_indices], centroid).flatten()
        local_best_idx = member_indices[int(np.argmax(sims))]
        if sims.max() > best_sim:
            best_sim = sims.max()
            best_idx = int(local_best_idx)

    return extracted_answers[best_idx]


# ==============================================================================
# 4. PART 1: EXACT EVALUATORS & METRICS BY SUBSET
# ==============================================================================
def eval_fraction_tree(node) -> Fraction:
    """Evaluates an AST arithmetic expression using exact fractions.Fraction."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(node.value)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_fraction_tree(node.operand)
    elif isinstance(node, ast.BinOp):
        left = eval_fraction_tree(node.left)
        right = eval_fraction_tree(node.right)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div):
            if right == 0: raise ZeroDivisionError()
            return left / right
    raise ValueError("Invalid AST node")


def score_problem(pred: str, gold, subset: str) -> dict:
    """Computes the exact Part 1 score for GSM8K, MATH-500, Game of 24, HotpotQA, and ASQA."""
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
            is_zero = (diff == 0)
            return {"score": 1.0 if is_zero else 0.0, "solved": bool(is_zero)}
        except Exception:
            return {"score": 0.0, "solved": False}

    elif subset == "Game of 24":
        if not re.match(r"^[\d\+\-\*\/\(\)\s]+$", pred):
            return {"score": 0.0, "solved": False}
        used_nums = sorted(re.findall(r"\d+", pred))
        gold_nums = sorted(re.findall(r"\d+", gold))
        if used_nums != gold_nums:
            return {"score": 0.0, "solved": False}
        try:
            tree = ast.parse(pred, mode="eval")
            val = eval_fraction_tree(tree.body)
            ok = (val == Fraction(24, 1))
            return {"score": 1.0 if ok else 0.0, "solved": ok}
        except Exception:
            return {"score": 0.0, "solved": False}

    elif subset == "HotpotQA":
        norm_p, norm_g = normalize_hotpot(pred), normalize_hotpot(gold)
        em = 1.0 if norm_p == norm_g else 0.0
        p_toks, g_toks = norm_p.split(), norm_g.split()
        common = Counter(p_toks) & Counter(g_toks)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            prec = num_same / len(p_toks)
            rec = num_same / len(g_toks)
            f1 = (2 * prec * rec) / (prec + rec)
        return {"score": f1, "f1": f1, "em": em, "solved": em == 1.0}

    elif subset == "ASQA":
        norm_p = normalize_hotpot(pred)
        short_groups = gold["short_groups"]
        hits = 0
        for group in short_groups:
            if any(normalize_hotpot(alias) in norm_p for alias in group if alias):
                hits += 1
        str_em = hits / len(short_groups) if short_groups else 0.0

        rouge_l = max(
            rouge_eval.score(ref, pred)["rougeL"].fmeasure
            for ref in gold["long_refs"]
        ) if gold["long_refs"] else 0.0
        dr = float(np.sqrt(str_em * rouge_l))
        return {"score": dr, "dr": dr, "str_em": str_em, "rouge_l": rouge_l, "solved": dr >= 0.5}


# ==============================================================================
# 5. PART 2: INFERENCE ENGINE & COST LOGGING (WITH GPU OFFLOAD ENFORCED)
# ==============================================================================
sem = asyncio.Semaphore(4)

async def generate_single_call(prompt: str, temperature: float = 0.7) -> dict:
    """Runs a single local LLM call with num_gpu=99 to ensure all layers run on RTX 4060."""
    async with sem:
        resp = await client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            max_tokens=600,
            extra_body={
                "top_k": 40,       # Matches k=40 top-k sampling in the paper
                "num_gpu": 99      # Forces Ollama to offload all layers to RTX 4060 GPU
            }
        )
        msg = resp.choices[0].message.content
        tok_in = int(getattr(resp.usage, "prompt_tokens", len(prompt.split()) * 1.3))
        tok_out = int(getattr(resp.usage, "completion_tokens", len(msg.split()) * 1.3))
        return {
            "raw_text": msg,
            "tokens_in": tok_in,
            "tokens_out": tok_out,
            "tokens_total": tok_in + tok_out,
            "dollars": float(tok_in * PRICE_IN + tok_out * PRICE_OUT)
        }


async def run_problem_sc(problem: dict, k: int = FIXED_K) -> dict:
    """Runs k=10 Self-Consistency paths on one problem, aggregates, scores, and logs cost."""
    prompt = build_cot_prompt(problem)
    subset = problem["subset"]

    samples = await asyncio.gather(*[generate_single_call(prompt, temperature=0.7) for _ in range(k)])

    for s in samples:
        s["extracted_answer"] = extract_answer(s["raw_text"], subset)

    all_extracted = [s["extracted_answer"] for s in samples]
    final_pred = select_final_answer_sc(all_extracted, subset)

    tokens_in_sum = sum(s["tokens_in"] for s in samples)
    tokens_out_sum = sum(s["tokens_out"] for s in samples)
    tokens_total = tokens_in_sum + tokens_out_sum
    dollars_total = sum(s["dollars"] for s in samples)

    eval_metrics = score_problem(final_pred, problem["gold"], subset)

    return {
        "problem_id": problem["id"],
        "subset": subset,
        "group": problem["group"],
        "method": "Self-Consistency",
        "config": f"k={k}",
        "k": k,
        "num_calls": k,
        "tokens_in": tokens_in_sum,
        "tokens_out": tokens_out_sum,
        "tokens_total": tokens_total,
        "dollars": dollars_total,
        "final_extracted_answer": final_pred,
        "score": eval_metrics["score"],
        "problem_solved": eval_metrics["solved"],
        "eval_details": eval_metrics,
        "samples": samples
    }


# ==============================================================================
# 6. PART 4: AGGREGATION & JSON EXPORT
# ==============================================================================
def compute_part4_aggregates(problem_rows: list[dict], config_label: str = f"k={FIXED_K}") -> dict:
    """Aggregates per-subset quality, Group A/B macro-averages, and mean tokens/calls."""
    df = pd.DataFrame(problem_rows)

    per_subset = {}
    for subset, grp in df.groupby("subset"):
        entry = {
            "N": int(len(grp)),
            "mean_quality": float(grp["score"].mean()),
            "solved_count": int(grp["problem_solved"].sum()),
            "mean_tokens": float(grp["tokens_total"].mean()),
            "mean_dollars": float(grp["dollars"].mean()),
            "mean_calls": float(grp["num_calls"].mean())
        }
        if subset == "HotpotQA":
            entry["mean_em"] = float(np.mean([d["em"] for d in grp["eval_details"]]))
        elif subset == "ASQA":
            entry["mean_str_em"] = float(np.mean([d["str_em"] for d in grp["eval_details"]]))
            entry["mean_rouge_l"] = float(np.mean([d["rouge_l"] for d in grp["eval_details"]]))
        per_subset[subset] = entry

    group_a_subsets = ["GSM8K", "MATH-500", "Game of 24", "HotpotQA"]
    group_b_subsets = ["ASQA"]
    all_subsets = group_a_subsets + group_b_subsets

    pooled_group_a = float(np.mean([per_subset[s]["mean_quality"] for s in group_a_subsets if s in per_subset]))
    pooled_group_b = float(np.mean([per_subset[s]["mean_quality"] for s in group_b_subsets if s in per_subset]))
    pooled_all_macro = float(np.mean([per_subset[s]["mean_quality"] for s in all_subsets if s in per_subset]))

    return {
        "method": "Self-Consistency",
        "config": config_label,
        "total_problems": int(len(df)),
        "pooled_quality_group_A_macro": pooled_group_a,
        "pooled_quality_group_B_macro": pooled_group_b,
        "pooled_quality_all_5_subsets_macro": pooled_all_macro,
        "overall_mean_tokens_per_problem": float(df["tokens_total"].mean()),
        "overall_mean_dollars_per_problem": float(df["dollars"].mean()),
        "overall_mean_calls_per_problem": float(df["num_calls"].mean()),
        "per_subset_results": per_subset
    }


async def main():
    # Step 0: Verify RTX 4060 GPU is active in Ollama before starting
    verify_rtx4060_and_ollama()

    print("Loading 150-problem dataset (GSM8K=30, MATH-500=20, Game24=20, HotpotQA=25, ASQA=55)...")
    problems = load_150_problems()

    problem_results = []
    checkpoint_file = "sc_k10_no_eli5_checkpoint.jsonl"

    completed_ids = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                problem_results.append(row)
                completed_ids.add(row["problem_id"])
        print(f"Resumed {len(completed_ids)}/150 problems from checkpoint.")

    remaining = [p for p in problems if p["id"] not in completed_ids]

    for prob in tqdm_asyncio(remaining, desc=f"Running Self-Consistency (k={FIXED_K})"):
        res = await run_problem_sc(prob, k=FIXED_K)
        problem_results.append(res)
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")

    summary = compute_part4_aggregates(problem_results, config_label=f"k={FIXED_K}")
    final_output = {
        "summary_metrics": summary,
        "problem_level_table": [
            {k: v for k, v in row.items() if k != "samples"}
            for row in problem_results
        ]
    }

    with open("self_consistency_150_results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print("\nSaved complete results to 'self_consistency_150_results.json'!")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())