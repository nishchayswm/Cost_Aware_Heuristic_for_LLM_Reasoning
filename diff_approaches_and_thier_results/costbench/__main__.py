from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .backend import BackendError, Session, append_json, read_jsonl, request
from .graph import number
from .labels import load_corrections, score
from .methods import Engine, METHODS
from .report import build_report

ROOT=Path(__file__).resolve().parent.parent


def fingerprint():
    digest=hashlib.sha256()
    for p in sorted((ROOT/"costbench").glob("*.py")):
        digest.update(p.name.encode());digest.update(p.read_bytes())
    return digest.hexdigest()


def config(path):
    cfg=json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ["max_calls","max_output_tokens","max_total_tokens","solve_tokens","step_tokens",
                "judge_tokens","samples","beam_width","branch_factor","max_depth","num_ctx",
                "max_expansions","max_frontier","cost_reference_tokens"]:
        if not isinstance(cfg[key],int) or cfg[key]<1: raise ValueError(f"{key} must be a positive integer")
    if type(cfg.get("generation_retries",1)) is not int or not 0 <= cfg.get("generation_retries",1) <= 3:
        raise ValueError("generation_retries must be an integer from 0 to 3")
    if type(cfg.get("quality_gated_diversity",False)) is not bool:
        raise ValueError("quality_gated_diversity must be boolean")
    for flag in ("search_revision","progress_aware_depth","judge_json_mode","numeric_expressions","initial_solution","diverse_generation"):
        if type(cfg.get(flag,False)) is not bool:raise ValueError(f"{flag} must be boolean")
    names={"quality","success","depth","cost","diversity"}
    if set(cfg["weights"])!=names:raise ValueError("weights must contain exactly the five heuristic factors")
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in cfg["weights"].values()):
        raise ValueError("Weights must be finite and nonnegative")
    if not math.isclose(sum(cfg["weights"].values()),1,abs_tol=1e-8):raise ValueError("Weights must sum to one")
    if cfg["beam_temperature"]<=0:raise ValueError("beam_temperature must be positive")
    return cfg


def doctor(cfg):
    tags=request(cfg["base_url"],"/api/tags")
    model=next((m for m in tags["models"] if m["name"]==cfg["model"]),None)
    if model is None:
        raise RuntimeError(f'Model missing: {cfg["model"]}. Run: ollama pull {cfg["model"]}')
    print(f'Python {platform.python_version()} | Ollama model: {model["name"]}',flush=True)
    print(f'Quantization: {model.get("details",{}).get("quantization_level")} | digest: {model["digest"]}',flush=True)
    return {"model":model,"ollama":request(cfg["base_url"],"/api/version"),
            "python":platform.python_version(),"platform":platform.platform()}


def run(args):
    cfg=config(args.config)
    if args.model: cfg["model"]=args.model
    if args.seed is not None: cfg["seed"]=args.seed
    dataset=Path(args.dataset)
    if args.limit<1:raise ValueError("limit must be positive")
    questions=read_jsonl(dataset)[:args.limit]
    if not questions: raise ValueError("Empty dataset")
    if len({q["id"] for q in questions})!=len(questions): raise ValueError("Duplicate question IDs")
    if any(number(q["answer"]) is None for q in questions): raise ValueError("Invalid dataset answer")
    corrections=load_corrections(getattr(args,"label_corrections",ROOT/"data/label_corrections.json"))
    for q in questions: score(q,None,corrections)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    env=doctor(cfg)
    manifest={"label_corrections":corrections,"config":cfg,"dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest(),
              "code_sha256":fingerprint(),"question_ids":[q["id"] for q in questions],
              "methods":METHODS,"model_digest":env["model"]["digest"]}
    meta=root/"manifest.json"
    if meta.exists():
        saved=json.loads(meta.read_text(encoding="utf-8"))
        if saved["experiment"]!=manifest:
            raise ValueError("Config/model/code/dataset changed. Use a NEW output directory to avoid mixing experiments.")
    else:
        meta.write_text(json.dumps({"experiment":manifest,"environment":env,
                                   "created_utc":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
    (root/"label_corrections.json").write_text(json.dumps(list(corrections.values()),indent=2),encoding="utf-8")
    lock=root/"run.lock"
    try:
        handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Run lock exists: {lock}. If no runner is active, remove this stale lock before resuming.")
    os.write(handle,str(os.getpid()).encode());os.close(handle)
    try:
        build_report(root)
        saved_rows=read_jsonl(root/"results.jsonl")
        existing={(r["question_id"],r["seed"],r["method"]):r for r in saved_rows}
        if all((q["id"],cfg["seed"],m) in existing for q in questions for m in METHODS):
            build_report(root);print("Already complete; reports refreshed.");return
        # One warmup excluded from method costs and saved separately.
        start=time.perf_counter()
        warm=request(cfg["base_url"],"/api/chat",{"model":cfg["model"],"messages":[{"role":"user","content":"Reply OK."}],
                     "stream":False,"keep_alive":"30m","options":{"num_ctx":cfg["num_ctx"],"num_predict":8}},cfg["timeout_seconds"])
        append_json(root/"warmups.jsonl",{"seconds":time.perf_counter()-start,"response":warm})
        for index,q in enumerate(questions):
            # One model on one GPU; rotate policy order to reduce systematic order bias.
            methods=list(METHODS);shift=index%len(methods);methods=methods[shift:]+methods[:shift]
            print(f'\nQUESTION {index+1}/{len(questions)}: {q["id"]}',flush=True)
            print(q["question"],flush=True)
            print(f'Ground truth (evaluator only): {q["answer"]}',flush=True)
            labels=score(q,None,corrections)
            if labels["label_correction_applied"]:
                print(f'Reviewed label (evaluator only): {labels["gold_reviewed"]}; original retained.',flush=True)
            for method in methods:
                if (root/"pause.request").exists():
                    (root/"pause.request").unlink()
                    build_report(root)
                    print("Paused cleanly between methods. Rerun the same command to resume.",flush=True)
                    return
                key=(q["id"],cfg["seed"],method)
                if key in existing:
                    print(f"  {method}: resumed (already saved)",flush=True);continue
                session=Session(cfg,q["id"],method,cfg["seed"],root/"calls"/f'{q["id"]}_{cfg["seed"]}_{method}.jsonl',
                                retry_failed=getattr(args,"retry_failed",False))
                start=time.perf_counter()
                try:
                    # Only the question text reaches the policy. Gold stays in this evaluator.
                    result=Engine(session,q["question"]).run(method)
                    result["status"]="ok"
                except BackendError:
                    # Stop rather than silently score infrastructure failures as model mistakes.
                    build_report(root)
                    raise
                elapsed=time.perf_counter()-start
                prediction=number(result["prediction"]) if result["prediction"] is not None else None
                row={"question_id":q["id"],"question":q["question"],"seed":cfg["seed"],"method":method,
                     "label":METHODS[method],"gold":number(q["answer"]),**result,**session.metrics(),
                     "execution_seconds_this_attempt":round(elapsed,4),
                     **score(q,prediction,corrections)}
                # Trace is separate so wide result tables stay small.
                trace=root/"traces"/f'{q["id"]}_{cfg["seed"]}_{method}.json'
                trace.parent.mkdir(exist_ok=True)
                trace.write_text(json.dumps(row,indent=2),encoding="utf-8")
                row.pop("candidates",None);row.pop("events",None)
                row["trace"]=str(trace.relative_to(root))
                append_json(root/"results.jsonl",row)
                existing[key]=row
                print(f'  {method:20s} {"CORRECT" if row["correct"] else "WRONG":7s} answer={prediction} gold={row["gold"]} '
                      f'reviewed_correct={row["correct_reviewed"]} reviewed_gold={row["gold_reviewed"]} calls={row["calls"]} tokens={row["total_tokens"]} seconds={row["inference_seconds"]:.1f}',flush=True)
                # Save the partial question table too; summary waits for equal coverage.
                build_report(root)
            print("\n"+"Method".ljust(24)+"Prediction".ljust(16)+"Original".ljust(10)+"Reviewed".ljust(10)+"Calls".ljust(8)+"Tokens".ljust(10)+"Seconds",flush=True)
            for m in METHODS:
                r=existing[(q["id"],cfg["seed"],m)]
                print(f'{m:24s}{str(r["prediction"]):16s}{str(r["correct"]):10s}{str(r.get("correct_reviewed",r["correct"])):10s}{r["calls"]:<8}{r["total_tokens"]:<10}{r["inference_seconds"]:.1f}',flush=True)
            print(f'All {len(METHODS)} method results saved. Report: {root/"report.html"}',flush=True)
        print("\nCOMPLETE: "+str(root/"report.html"),flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description="Question-major local LLM reasoning benchmark")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("doctor");p.add_argument("--config",default=str(ROOT/"configs/default.json"))
    p=sub.add_parser("report");p.add_argument("--output",required=True)
    p=sub.add_parser("status");p.add_argument("--output",default=str(ROOT/"results/gsm8k100_v6"))
    p=sub.add_parser("pause");p.add_argument("--output",default=str(ROOT/"results/gsm8k100_v6"))
    p=sub.add_parser("run")
    p.add_argument("--config",default=str(ROOT/"configs/default.json"))
    p.add_argument("--dataset",default=str(ROOT/"data/gsm8k_test_100.jsonl"))
    p.add_argument("--output",default=str(ROOT/"results/gsm8k100_v6"))
    p.add_argument("--label-corrections",default=str(ROOT/"data/label_corrections.json"))
    p.add_argument("--limit",type=int,default=100)
    p.add_argument("--seed",type=int)
    p.add_argument("--model")
    p.add_argument("--retry-failed",action="store_true",help="Retry a failed backend request on resume; its unknown cost remains flagged")
    args=parser.parse_args()
    try:
        if args.command=="doctor": doctor(config(args.config))
        elif args.command=="report": build_report(args.output)
        elif args.command=="status":
            root=Path(args.output);rows=read_jsonl(root/"results.jsonl")
            groups={}
            for r in rows:groups.setdefault((r["question_id"],r["seed"]),set()).add(r["method"])
            done=sum(len(methods)==len(METHODS) for methods in groups.values())
            print(f"Saved: {len(rows)} method results; {done} complete all-method blocks.")
            print("Run lock: "+((root/"run.lock").read_text() if (root/"run.lock").exists() else "none"))
            for method,label in METHODS.items():
                group=[r for r in rows if r["method"]==method]
                print(f'{label}: {sum(r["correct"] for r in group)}/{len(group)} correct (coverage may differ until block completes)')
        elif args.command=="pause":
            root=Path(args.output)
            if not (root/"run.lock").exists():
                print("No run lock found; no pause request created.")
            else:
                (root/"pause.request").write_text("Pause after current method completes.\n")
                print("Pause requested. The current method will finish and save before stopping.")
        else: run(args)
    except KeyboardInterrupt:
        print("\nStopped. Completed results and successful calls are saved; rerun the identical command to resume.",file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        return 1
    return 0


if __name__=="__main__": raise SystemExit(main())
