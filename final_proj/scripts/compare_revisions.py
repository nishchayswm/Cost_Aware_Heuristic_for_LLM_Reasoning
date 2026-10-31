"""Compare two complete development runs; never invokes a model."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--before',required=True);p.add_argument('--after',required=True)
a=p.parse_args()
def rows(path):
    return {(r['question_id'],r['seed'],r['method']):r for r in (json.loads(l) for l in (Path(path)/'results.jsonl').read_text(encoding='utf-8').splitlines() if l.strip())}
before,after=rows(a.before),rows(a.after)
if set(before)!=set(after):raise SystemExit('Run coverage differs; wait for both runs to finish.')
methods=list(dict.fromkeys(r['method'] for r in after.values()))
lines=['# Paired development revision comparison','','Same questions and seed. Revisions were informed by these development examples; this is not an unbiased held-out improvement estimate. Shared generator/search changes must not be credited solely to the heuristic.','','| Method | Before | After | Fixed | Regressed | Before mean calls | After mean calls | Before mean tokens | After mean tokens |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
summary=[]
for method in methods:
    pairs=[(before[k],r) for k,r in after.items() if r['method']==method]
    for b,r in pairs:
        if b['question']!=r['question'] or b.get('gold_reviewed',b['gold'])!=r.get('gold_reviewed',r['gold']):raise SystemExit('Question/label mismatch')
    n=len(pairs)
    def correct(r):return r.get('correct_reviewed',r['correct'])
    item=dict(method=method,n=n,before_correct=sum(correct(b) for b,r in pairs),after_correct=sum(correct(r) for b,r in pairs),fixed=sum(not correct(b) and correct(r) for b,r in pairs),regressed=sum(correct(b) and not correct(r) for b,r in pairs))
    for field in ('calls','total_tokens'):
        item['before_'+field]=sum(b[field] for b,r in pairs)/n
        item['after_'+field]=sum(r[field] for b,r in pairs)/n
    summary.append(item)
    lines.append(f"| {method} | {item['before_correct']}/{n} | {item['after_correct']}/{n} | {item['fixed']} | {item['regressed']} | {item['before_calls']:.1f} | {item['after_calls']:.1f} | {item['before_total_tokens']:.0f} | {item['after_total_tokens']:.0f} |")
root=Path(a.after)
(root/'revision_comparison.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
(root/'revision_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
