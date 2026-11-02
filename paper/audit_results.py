"""Recompute manuscript evidence from immutable snapshots of existing logs; no inference."""
from pathlib import Path
import sys,json,hashlib,math,csv
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from costbench.game24 import inspect_expression,METHODS
from costbench.__main__ import fingerprint

RUNS=['game24_original_n100_seed42_ollama','game24_test100_exp64_seed42',
      'game24_test100_exp64_seed43_v2','game24_test100_exp64_seed44']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
evidence={};loaded={}
for name in RUNS:
    path=ROOT/'results'/name;rows=read(path/'results.jsonl');meta=json.loads((path/'manifest.json').read_text())
    assert len(rows)==800 and len({(r['question_id'],r['method']) for r in rows})==800
    assert set(r['question_id'] for r in rows)==set(meta['ids'])
    assert meta['code_sha256']==fingerprint()
    verified=0;failures=0;logged_calls=0;truncated=0;parse=0;errors=0
    for r in rows:
        if r['prediction'] is not None:
            value,correct=inspect_expression(r['prediction'],r['numbers'])
            assert correct==r['correct'];verified+=correct
        else:assert not r['correct'];failures+=1
        journal=read(path/'calls'/f'{r["question_id"]}_{r["method"]}.jsonl') if r['calls'] else []
        responses=[x for x in journal if 'response' in x]
        assert len(responses)==r['calls']
        assert sum(x['response']['prompt_eval_count']+x['response']['eval_count'] for x in responses)==r['total_tokens']
        assert sum(x['response']['eval_count'] for x in responses)==r['output_tokens']
        assert r['total_tokens']<=meta['config']['max_total_tokens'] and r['calls']<=meta['config']['max_calls']
        assert r['output_tokens']<=meta['config']['max_output_tokens']
        errors+=sum('error' in x for x in journal)
        logged_calls+=len(responses);truncated+=r.get('truncated_calls',0);parse+=r.get('parse_failures',0)
    summary={}
    for m in METHODS:
        rr=[r for r in rows if r['method']==m]
        summary[m]=dict(n=len(rr),solved=sum(r['correct'] for r in rr),
           **{k:float(np.mean([r.get(k,0) for r in rr])) for k in ('correct','calls','total_tokens','input_tokens','output_tokens','inference_seconds','execution_seconds','expanded_states','rollout_calls','rating_failures','parse_failures','truncated_calls','endgame_checks')},
           stop_reasons=dict(Counter(r['stop_reason'] for r in rr)))
    evidence[name]=dict(rows=len(rows),verified_successes=verified,failures=failures,logged_calls=logged_calls,
        journal_errors=errors,unknown_cost=sum(bool(r.get('unknown_cost')) for r in rows),
        parse_failures=parse,truncated_calls=truncated,results_sha256=sha(path/'results.jsonl'),manifest=meta,summary=summary)
    loaded[name]=rows
    print(name, 'audited',len(rows),'records and',logged_calls,'calls',flush=True)

orig=loaded[RUNS[0]];gen=[r for name in RUNS[1:] for r in loaded[name]]
refs={s:{tuple(sorted(r['numbers'])) for r in read(ROOT/f'data/game24/game24_{s}.jsonl')} for s in ('train','dev','test')}
orig_nums={r['question_id']:tuple(sorted(r['numbers'])) for r in orig}
overlap={s:sum(n in refs[s] for n in orig_nums.values()) for s in refs}
clean={qid for qid,n in orig_nums.items() if not any(n in ref for ref in refs.values())}
evidence['overlap']=dict(**overlap,disjoint_count=len(clean),disjoint_ids=sorted(clean))
evidence['disjoint_summary']={m:dict(n=len(clean),solved=sum(r['correct'] for r in orig if r['method']==m and r['question_id'] in clean),
  tokens=float(np.mean([r['total_tokens'] for r in orig if r['method']==m and r['question_id'] in clean])),
  calls=float(np.mean([r['calls'] for r in orig if r['method']==m and r['question_id'] in clean]))) for m in METHODS}
evidence['generated_summary']={m:dict(n=300,solved=sum(r['correct'] for r in gen if r['method']==m),
    seed_successes=[evidence[name]['summary'][m]['solved'] for name in RUNS[1:]],
    **{k:float(np.mean([r[k] for r in gen if r['method']==m])) for k in ('correct','calls','total_tokens','execution_seconds')}) for m in METHODS}

def paired(rows,cluster=False):
    rng=np.random.default_rng(20261102);ids=sorted({r['question_id'] for r in rows})
    index=rng.integers(0,len(ids),size=(20000,len(ids)))
    arrays={m:{k:np.array([np.mean([r[k] for r in rows if r['method']==m and r['question_id']==q]) for q in ids])
        for k in ('correct','total_tokens','calls')} for m in METHODS}
    output={}
    for m in ('quality_bestfirst','tot_bfs','self_eval_beam','self_consistency','verified_sampling'):
        h=arrays['heuristic_bestfirst'];b=arrays[m]
        delta=h['correct']-b['correct']
        diff=delta[index].mean(axis=1)
        savings=1-h['total_tokens'][index].mean(axis=1)/b['total_tokens'][index].mean(axis=1)
        call_savings=1-h['calls'][index].mean(axis=1)/b['calls'][index].mean(axis=1)
        o=dict(delta=float(delta.mean()),delta_ci=np.quantile(diff,[.025,.975]).tolist(),
               token_saving=float(1-h['total_tokens'].mean()/b['total_tokens'].mean()),
               token_saving_ci=np.quantile(savings,[.025,.975]).tolist(),
               call_saving=float(1-h['calls'].mean()/b['calls'].mean()),
               call_saving_ci=np.quantile(call_savings,[.025,.975]).tolist())
        if not cluster:
            win=int(sum(delta>0));loss=int(sum(delta<0));n=win+loss
            o.update(wins=win,losses=loss,mcnemar_p=min(1,2*sum(math.comb(n,i) for i in range(min(win,loss)+1))/2**n) if n else 1)
        output[m]=o
    return output
evidence['paired_original']=paired(orig)
evidence['paired_generated_clustered']=paired(gen,True)
evidence['analysis']=dict(bootstrap_resamples=20000,bootstrap_seed=20261102,
    generated_unit='puzzle cluster containing all three seed runs',original_unit='paired puzzle',code_sha256=fingerprint())
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
with (OUT/'audited_results.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,['dataset','run','question_id','method','seed','correct','calls','total_tokens','execution_seconds'])
    writer.writeheader()
    for name,rows in loaded.items():
        for r in rows:writer.writerow(dict(dataset='original' if name==RUNS[0] else 'generated',run=name,
            **{k:r[k] for k in ('question_id','method','seed','correct','calls','total_tokens','execution_seconds')}))
print(json.dumps({k:evidence[k] for k in ('overlap','generated_summary','paired_original','paired_generated_clustered')},indent=2))
