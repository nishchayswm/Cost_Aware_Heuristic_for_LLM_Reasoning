"""Dependency-free CSV, Markdown and HTML reporting; only paired questions compared."""
import csv
import html
import json
import math
import random
import statistics
from pathlib import Path

from .backend import read_jsonl
from .methods import METHODS


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def wilson(k, n):
    if not n: return 0, 0
    z=1.96; p=k/n; d=1+z*z/n
    mid=(p+z*z/(2*n))/d
    width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return mid-width,mid+width


def build_report(directory):
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(root/"results.jsonl")
    for r in rows:
        r.setdefault("correct_original",r["correct"])
        r.setdefault("correct_reviewed",r["correct"])
        r.setdefault("gold_reviewed",r["gold"])
    by_key={(r["question_id"],r["seed"],r["method"]):r for r in rows}
    rows=list(by_key.values())
    cases=sorted({(r["question_id"],r["seed"]) for r in rows})
    complete=[key for key in cases if all((*key,m) in by_key for m in METHODS)]
    # Comparisons use only complete all-method blocks; failures remain incorrect.
    paired=[by_key[(*key,m)] for key in complete for m in METHODS]
    summary=[]
    for method,label in METHODS.items():
        group=[r for r in paired if r["method"]==method]
        n=len(group); k=sum(r["correct"] for r in group)
        reviewed=sum(r["correct_reviewed"] for r in group)
        rlow,rhigh=wilson(reviewed,n)
        low,high=wilson(k,n)
        def avg(field): return round(statistics.mean(r.get(field,0) for r in group),3) if group else 0
        summary.append({"method":method,"label":label,"n":n,"correct":k,
                        "accuracy":round(k/n,4) if n else None,
                        "correct_reviewed":reviewed,"accuracy_reviewed":round(reviewed/n,4) if n else None,
                        "reviewed_wilson_low":round(rlow,4),"reviewed_wilson_high":round(rhigh,4),
                        "mean_generation_retries":avg("generation_retries"),
                        "wilson_low":round(low,4),"wilson_high":round(high,4),
                        "mean_calls":avg("calls"),"mean_input_tokens":avg("input_tokens"),
                        "mean_output_tokens":avg("output_tokens"),"mean_total_tokens":avg("total_tokens"),
                        "mean_seconds":avg("inference_seconds"),"mean_judge_calls":avg("judge_calls"),
                        "mean_parse_failures":avg("parse_failures"),
                        "mean_schema_failures":avg("candidate_schema_failures"),
                        "mean_truncated_calls":avg("truncated_calls"),
                        "mean_expanded_states":avg("expanded_states"),
                        "mean_duplicate_states":avg("duplicate_states"),
                        "mean_max_depth":avg("max_reached_depth"),
                        "mean_evaluation_failures":avg("evaluation_failures"),
                        "unknown_cost_results":sum(bool(r.get("unknown_cost")) for r in group),
                        "errors":sum(r["status"]!="ok" for r in group)})
    write_csv(root/"summary.csv",summary,list(summary[0]))
    wide=[]
    for qid,seed in cases:
        found=next(r for r in rows if r["question_id"]==qid and r["seed"]==seed)
        row={"question_id":qid,"seed":seed,"gold":found["gold"],"gold_reviewed":found["gold_reviewed"],"question":found["question"]}
        for method in METHODS:
            r=by_key.get((qid,seed,method),{})
            for field in ["prediction","correct","correct_original","correct_reviewed","generation_retries","calls","input_tokens","output_tokens","total_tokens","inference_seconds","status","stop_reason","parse_failures","candidate_schema_failures","expanded_states","duplicate_states","max_reached_depth","unknown_cost","failed_attempts"]:
                row[f"{method}__{field}"]=r.get(field,"")
        wide.append(row)
    fields=list(wide[0]) if wide else ["question_id","seed","gold","question"]
    write_csv(root/"per_question.csv",wide,fields)
    if rows: write_csv(root/"results_long.csv",rows,["question_id","seed","method","prediction","gold","gold_reviewed","correct","correct_original","correct_reviewed","label_correction_reason","calls","input_tokens","output_tokens","total_tokens","inference_seconds","status","stop_reason"])
    comparisons=[]
    # Bootstrap question clusters, keeping seeds together; no false independence.
    qids=sorted({k[0] for k in complete})
    for method in METHODS:
        if method=="heuristic_bestfirst" or not qids: continue
        deltas=[]
        reviewed_deltas=[]
        token_deltas=[]
        for qid in qids:
            pairs=[(by_key[(qid,s,"heuristic_bestfirst")],by_key[(qid,s,method)]) for q,s in complete if q==qid]
            deltas.append(statistics.mean(int(a["correct"])-int(b["correct"]) for a,b in pairs))
            reviewed_deltas.append(statistics.mean(int(a["correct_reviewed"])-int(b["correct_reviewed"]) for a,b in pairs))
            token_deltas.append(statistics.mean(a["total_tokens"]-b["total_tokens"] for a,b in pairs))
        rng=random.Random(73)
        samples=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(2000))
        lo,hi=samples[49],samples[1949]
        rng=random.Random(73)
        reviewed_samples=sorted(statistics.mean(rng.choices(reviewed_deltas,k=len(reviewed_deltas))) for _ in range(2000))
        comparisons.append({"baseline":method,"questions":len(qids),
                            "accuracy_difference":statistics.mean(deltas),"bootstrap_low":lo,"bootstrap_high":hi,
                            "reviewed_accuracy_difference":statistics.mean(reviewed_deltas),
                            "reviewed_bootstrap_low":reviewed_samples[49],"reviewed_bootstrap_high":reviewed_samples[1949],
                            "mean_token_difference":statistics.mean(token_deltas)})
    write_csv(root/"paired_comparisons.csv",comparisons,["baseline","questions","accuracy_difference","bootstrap_low","bootstrap_high","reviewed_accuracy_difference","reviewed_bootstrap_low","reviewed_bootstrap_high","mean_token_difference"])
    lines=["# Local reasoning benchmark", "",f"Complete all-method blocks: {len(complete)}. Saved method results: {len(rows)}.","",
           "Accuracy and costs below use only completed all-method blocks. Errors count as incorrect; inspect status and unknown-cost failures.","",
           "| Method | Original correct / N | Original accuracy | Reviewed correct / N | Reviewed accuracy | Calls | Input tokens | Output tokens | Seconds |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        acc=f'{r["accuracy"]:.1%}' if r["accuracy"] is not None else "pending"
        reviewed_acc=f'{r["accuracy_reviewed"]:.1%}' if r["accuracy_reviewed"] is not None else "pending"
        lines.append(f'| {r["label"]} | {r["correct"]}/{r["n"]} | {acc} | {r["correct_reviewed"]}/{r["n"]} | {reviewed_acc} | {r["mean_calls"]} | {r["mean_input_tokens"]} | {r["mean_output_tokens"]} | {r["mean_seconds"]} |')
    lines += ["", "Original metrics retain dataset labels. Reviewed metrics use only the frozen evaluator-only label corrections saved with this run. Legacy runs without reviewed fields retain their original scores; no retrospective correction is applied.", "These are calculator-assisted local adaptations, not reproductions of published SOTA scores. Self-evaluation beam uses prompted scores, not the original probability-based objective.",
              "Five-factor best-first ranks intermediate states using quality, uncalibrated success confidence, depth, estimated additional token cost and lexical diversity. Weights are provisional, not learned.",
              "Wilson intervals are descriptive for single-seed runs; use question-cluster bootstrap differences for repeated seeds.",
              "Inference seconds include recorded request wall time. Warmup is excluded and separately recorded. Timing depends on caching and background load.",
              "A 100-question pilot cannot by itself establish non-inferiority or SOTA. Do not tune using these test results."]
    (root/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    header="""<!doctype html><meta charset="utf-8"><title>Local reasoning benchmark</title>
<style>body{font:15px system-ui;background:#f5f7fb;color:#17233b;margin:40px auto;max-width:1250px;padding:0 24px}table{border-collapse:collapse;background:white;width:100%;margin:24px 0}th,td{padding:12px;border-bottom:1px solid #dbe1eb;text-align:left}th{background:#e8eef8}h1{font-size:28px}.note{color:#57657c}svg{background:white;border-radius:12px}pre{white-space:pre-wrap}a{color:#2455ad}</style>"""
    body=f"<h1>Local reasoning benchmark</h1><p>{len(complete)} complete all-method question/seed blocks; {len(rows)} saved results.</p>"
    body+="<p class='note'>Question-major execution. All methods use one local model. Local adaptations; no established SOTA claim.</p>"
    if any(r.get("unknown_cost") for r in rows):
        body+="<p><strong>Some backend failures were retried. Their token costs are unknown; reported costs are lower bounds. Inspect unknown_cost_results in summary.csv before comparing efficiency.</strong></p>"
    body+="<table><tr><th>Method</th><th>Original correct / N</th><th>Original accuracy</th><th>Reviewed correct / N</th><th>Reviewed accuracy</th><th>Calls</th><th>Total tokens</th><th>Seconds</th></tr>"
    for r in summary:
        acc=f'{r["accuracy"]:.1%}' if r["accuracy"] is not None else "pending"
        reviewed_acc=f'{r["accuracy_reviewed"]:.1%}' if r["accuracy_reviewed"] is not None else "pending"
        body+=f'<tr><td>{html.escape(r["label"])}</td><td>{r["correct"]}/{r["n"]}</td><td>{acc}</td><td>{r["correct_reviewed"]}/{r["n"]}</td><td>{reviewed_acc}</td><td>{r["mean_calls"]}</td><td>{r["mean_total_tokens"]}</td><td>{r["mean_seconds"]}</td></tr>'
    body+="</table><h2>Original-label accuracy versus tokens</h2><svg viewBox='0 0 900 430' width='100%' role='img' aria-label='Accuracy versus mean total tokens'>"
    body+="<path d='M70 20V360H870' stroke='#789' fill='none'/><text x='350' y='410'>Mean total tokens per question (lower is cheaper)</text>"
    maxcost=max([r["mean_total_tokens"] for r in summary]+[1])*1.1
    for tick in range(0,101,20):
        y=360-3.2*tick
        body+=f"<text x='15' y='{y}'>{tick}%</text><path d='M70 {y}H870' stroke='#edf0f5'/>"
    colors=["#065f46","#2563eb","#d97706","#9333ea","#dc2626","#475569"]
    for i,r in enumerate(summary):
        if r["accuracy"] is None: continue
        x=70+780*r["mean_total_tokens"]/maxcost;y=360-320*r["accuracy"]
        body+=f'<circle cx="{x}" cy="{y}" r="7" fill="{colors[i]}"><title>{html.escape(r["label"])}: {r["accuracy"]:.1%}, {r["mean_total_tokens"]} tokens</title></circle>'
        body+=f'<text x="{x+9}" y="{y+15*i-25}" font-size="11" fill="{colors[i]}">{html.escape(r["method"])}</text>'
    body+="</svg><p><a href='per_question.csv'>Per-question scores for all methods</a> Â· <a href='summary.csv'>Summary CSV</a> Â· <a href='paired_comparisons.csv'>Paired comparisons</a></p>"
    body+="<h2>Individual questions</h2><table><tr><th>Question / seed</th><th>Gold: original / reviewed</th>"+"".join("<th>"+html.escape(m)+"</th>" for m in METHODS)+"</tr>"
    for row in wide:
        body+="<tr><td>"+html.escape(row["question_id"])+" / "+str(row["seed"])+"<br>"+html.escape(row["question"])+"</td><td>"+html.escape(str(row["gold"]))+" / "+html.escape(str(row["gold_reviewed"]))+"</td>"
        for method in METHODS:
            r=by_key.get((row["question_id"],row["seed"],method))
            body+=("<td>pending</td>" if r is None else "<td>"+html.escape(str(r["prediction"]))+"<br>"+("original correct" if r["correct"] else "original incorrect")+"<br>"+("reviewed correct" if r["correct_reviewed"] else "reviewed incorrect")+f'<br>{r.get("calls",0)} calls<br>{r.get("total_tokens",0)} tokens</td>')
        body+="</tr>"
    body+="</table>"
    body+="<h2>Methodology and limitations</h2><pre>"+html.escape("\n".join(lines[-5:]))+"</pre>"
    (root/"report.html").write_text(header+body,encoding="utf-8")
    return summary
