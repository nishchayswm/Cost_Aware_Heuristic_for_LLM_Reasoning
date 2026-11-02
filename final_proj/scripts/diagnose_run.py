"""Read-only diagnostics of a saved run. No model access; writes analysis JSON."""
import argparse
import json
from collections import Counter
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.output)
rows=[json.loads(line) for line in (root/'results.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
summary={}
for method in dict.fromkeys(r['method'] for r in rows):
    group=[r for r in rows if r['method']==method]
    counts=Counter();examples=[]
    for r in group:
        trace=json.loads((root/r['trace']).read_text(encoding='utf-8'))
        for event in trace['events']:
            if event['event']=='evaluation':
                counts['judged_nodes']+=1
                counts['zero_score_nodes']+=int(event['quality']==0 and event['success']==0)
                counts['pending_completion_nodes']+=int(event.get('rubric',{}).get('completion')=='pending')
            if event['event']=='reopen_incomplete_answer':counts['reopened_answers']+=1
            if event['event']=='invalid_generation':counts['invalid_generations']+=1
        if not r.get('correct_reviewed',r['correct']):
            examples.append({'question_id':r['question_id'],'prediction':r['prediction'],'gold':r.get('gold_reviewed',r['gold']),'stop_reason':r['stop_reason'],'trace':r['trace']})
    summary[method]={'n':len(group),'correct':sum(r.get('correct_reviewed',r['correct']) for r in group),'no_prediction':sum(r['prediction'] is None for r in group),'calls':sum(r['calls'] for r in group),'tokens':sum(r['total_tokens'] for r in group),**counts,'failures':examples}
(root/'diagnostics.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
for method,row in summary.items():print(method,{k:v for k,v in row.items() if k!='failures'})
