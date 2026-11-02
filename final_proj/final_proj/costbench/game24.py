"""Game of 24: exact states/checker, offline dataset generation, local LLM benchmark."""
from __future__ import annotations
import argparse, ast, csv, hashlib, html, itertools, json, math, os, random, time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from .backend import Session, BudgetExceeded, BackendError, append_json, read_jsonl
from .__main__ import ROOT, config, doctor, fingerprint
from .methods import METHODS as ORIGINAL_METHODS

# A verified early-stopping sampler controls for savings due only to the checker.
METHODS={**ORIGINAL_METHODS,'verified_sampling':'Verified sampling: stop at first checked solution',
         'symbolic_search':'Exact symbolic search (no LLM reference)'}


@dataclass(frozen=True)
class Term:
    value: Fraction
    expression: str


def inspect_expression(expression, numbers, target=24):
    """No eval: exact arithmetic, approved binary operators, exact leaf multiset."""
    if not isinstance(expression,str) or len(expression)>500:raise ValueError('Invalid expression length')
    tree=ast.parse(expression.strip(),mode='eval')
    if sum(1 for _ in ast.walk(tree))>40:raise ValueError('Expression too complex')
    leaves=[]
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) is int and n.value>0:
            leaves.append(n.value);return Fraction(n.value)
        if not isinstance(n,ast.BinOp) or not isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            raise ValueError('Only supplied positive integers and binary + - * / allowed')
        a,b=visit(n.left),visit(n.right)
        if isinstance(n.op,ast.Add):return a+b
        if isinstance(n.op,ast.Sub):return a-b
        if isinstance(n.op,ast.Mult):return a*b
        return a/b
    value=visit(tree.body)
    if Counter(leaves)!=Counter(numbers):raise ValueError('Must use each supplied number exactly once')
    return value, value==target


def combine(terms, i, j, op):
    if type(i) is not int or type(j) is not int or i==j or not 0<=i<len(terms) or not 0<=j<len(terms):
        raise ValueError('Distinct valid indices required')
    if op not in ('+','-','*','/'):raise ValueError('Invalid operator')
    a,b=terms[i],terms[j]
    if op=='+':v=a.value+b.value
    elif op=='-':v=a.value-b.value
    elif op=='*':v=a.value*b.value
    else:v=a.value/b.value
    return tuple(t for k,t in enumerate(terms) if k not in (i,j))+(Term(v,f'({a.expression}{op}{b.expression})'),)


def solvable(values):
    """Exact oracle ONLY for offline dataset construction, never called by policy."""
    memo=set()
    def search(vs):
        key=tuple(sorted(vs))
        if key in memo:return False
        memo.add(key)
        if len(vs)==1:return vs[0]==24
        for i in range(len(vs)):
            for j in range(i+1,len(vs)):
                a,b=vs[i],vs[j];rest=[v for k,v in enumerate(vs) if k not in (i,j)]
                options={a+b,a-b,b-a,a*b}
                if b:options.add(a/b)
                if a:options.add(b/a)
                if any(search(rest+[v]) for v in options):return True
        return False
    return search(list(map(Fraction,values)))


def prepare(directory, size=100, seed=42):
    root=Path(directory)
    if size<1:raise ValueError('size must be positive')
    paths=[root/f'game24_{s}.jsonl' for s in ('train','dev','test')]+[root/'game24_manifest.json']
    if any(p.exists() for p in paths):raise ValueError('Dataset exists; choose a fresh directory')
    # Group by unordered multiset: permutations cannot leak across splits.
    pool=list(itertools.combinations_with_replacement(range(1,14),4));random.Random(seed).shuffle(pool)
    chosen=[]
    for nums in pool:
        if solvable(nums):chosen.append(nums)
        if len(chosen)==3*size:break
    if len(chosen)<3*size:raise ValueError('Requested more solvable puzzles than available')
    root.mkdir(parents=True,exist_ok=True)
    for k,split in enumerate(('train','dev','test')):
        rows=[dict(id='game24-'+'-'.join(map(str,n)),numbers=list(n)) for n in chosen[k*size:(k+1)*size]]
        paths[k].write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    paths[-1].write_text(json.dumps(dict(source='Locally generated, not the original ToT benchmark',range=[1,13],size_per_split=size,seed=seed,grouping='unordered number multiset',selection='random order filtered for exact solvability',hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths[:-1]}),indent=2),encoding='utf-8')
    print(f'Prepared {size} solvable puzzles per split; no LLM calls. {root}')


@dataclass
class State:
    id: int
    terms: tuple
    depth: int=0
    quality: float=.5
    success: float=.5
    parent: int|None=None
    features: dict|None=None


def legal_children(terms):
    """Enumerate one-step arithmetic transitions only; no reachability oracle."""
    seen=set();result=[]
    for i in range(len(terms)):
        for j in range(len(terms)):
            if i==j:continue
            for op in ('+','-','*','/'):
                if op in ('+','*') and i>j:continue
                try:child=combine(terms,i,j,op)
                except ZeroDivisionError:continue
                key=tuple(sorted(t.value for t in child))
                if key not in seen:
                    seen.add(key);result.append((dict(i=i,j=j,op=op),child))
    return result


class GameEngine:
    def __init__(self,session,numbers):
        self.s=session;self.cfg=session.cfg;self.numbers=numbers
        self.root=State(0,tuple(Term(Fraction(n),str(n)) for n in numbers))
        self.nodes=[self.root];self.events=[];self.expanded=[];self.seen={self.key(self.root)}
        self.answer=None;self.invalid=0;self.rollouts=0;self.expansions=0
        self.endgame_checks=0;self.rating_failures=0;self.off_branch_solutions=0
        self.next_tokens=float(self.cfg['step_tokens']);self.rng=random.Random(session.seed)
        self.rollout_spent=0

    @staticmethod
    def key(state):return tuple(sorted(t.value for t in state.terms))

    def accept(self,expression,state=None):
        """Global success is independent of evidence for a selected prefix."""
        valid=goal=branch=False
        try:
            _,goal=inspect_expression(expression,self.numbers);valid=True
            if state is None:branch=True
            else:
                tree=ast.parse(expression.strip(),mode='eval')
                subtrees=Counter(ast.dump(n) for n in ast.walk(tree))
                required=Counter(ast.dump(ast.parse(t.expression,mode='eval').body) for t in state.terms)
                branch=all(subtrees[k]>=v for k,v in required.items())
            if goal:
                self.answer=expression
                if not branch:self.off_branch_solutions+=1
        except (ValueError,SyntaxError,TypeError,ZeroDivisionError,OverflowError):pass
        return valid,goal,branch

    def completion(self,state=None,temperature=None):
        state=state or self.root
        prompt=('Game of 24. Use each original integer exactly once with binary + - * / '
                'and parentheses. No extra constants, concatenation or powers. Fractions and '
                f'negative intermediate results are legal. Numbers: {self.numbers}. '
                'Explain naturally and return JSON reasoning and expression. '
                'The expression must actually evaluate to 24. Current expression forest: '+
                json.dumps([t.expression for t in state.terms]))
        out,raw,_=self.s.ask(prompt,'game24_expression',self.cfg['solve_tokens'],temperature=temperature)
        valid,goal,branch=self.accept(out.get('expression'),state)
        self.events.append(dict(event='completion',state=state.id,valid=valid,solved=goal,
                                branch_consistent=branch,expression=out.get('expression'),raw=raw))
        return valid and branch,goal and branch

    def endgame(self,state):
        if len(state.terms)>2:return False
        self.endgame_checks+=1
        candidates=[state.terms] if len(state.terms)==1 else [t for _,t in legal_children(state.terms)]
        for terms in candidates:
            if terms[0].value==24:
                self.accept(terms[0].expression,state)
                self.events.append(dict(event='endgame',state=state.id,solved=True,expression=self.answer))
                return True
        self.events.append(dict(event='endgame',state=state.id,solved=False))
        return False

    def rate(self,parent,children):
        """One batched shortlist call. All unranked legal states remain available."""
        if not children:return
        views=[dict(id=c.id,values=[str(t.value) for t in c.terms]) for c in children]
        prompt=('Game of 24: rank legal next states for reaching 24 using every remaining '
                'value once with + - * /. Python already checked arithmetic and number usage. '
                'Return JSON ratings: a shortlist of at most '+str(self.cfg.get('rating_shortlist',6))+
                ' distinct candidate IDs with reason, quality and success, both numbers in [0,1]. '
                'Quality means how useful the reduction is, NOT whether its syntax is legal. '
                'Success is your uncertain estimate that continued search can finish. '
                'Reason about possible remaining operations; do not assume proximity to 24 implies success. '
                'Candidates: '+json.dumps(views))
        out,raw,tokens=self.s.ask(prompt,'game24_rank',self.cfg.get('rank_tokens',700),temperature=0)
        self.next_tokens=.7*self.next_tokens+.3*tokens
        lookup={c.id:c for c in children};used=set();ratings=out.get('ratings',[])
        if not isinstance(ratings,list):ratings=[]
        for r in ratings[:self.cfg.get('rating_shortlist',6)]:
            if (not isinstance(r,dict) or type(r.get('id')) is not int or r['id'] not in lookup or r['id'] in used
                or any(type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) or not 0<=r[k]<=1 for k in ('quality','success'))):
                self.rating_failures+=1;continue
            used.add(r['id']);c=lookup[r['id']];c.quality=r['quality'];c.success=r['success']
            self.events.append(dict(event='rating',state=c.id,Q=c.quality,P=c.success,reason=r.get('reason','')))
        if not used:self.rating_failures+=1
        self.events.append(dict(event='rating_batch',parent=parent.id,candidates=list(lookup),rated=sorted(used),raw=raw))

    def move_rollout(self,start):
        """Bounded auxiliary probe: choose legal move IDs; Python keeps the tree."""
        state=start;remaining=self.cfg.get('max_rollout_calls',2)-self.rollout_spent
        while remaining>0 and len(state.terms)>2 and not self.answer:
            choices=legal_children(state.terms)
            prompt=('Game of 24. Choose ONE next move ID from the legal options. Return JSON '
                    'choice (integer ID) and reason. Numbers after each move are exact. '+json.dumps([
                    dict(id=i,values=[str(t.value) for t in terms]) for i,(_,terms) in enumerate(choices)]))
            out,raw,_=self.s.ask(prompt,'game24_choice',self.cfg['step_tokens'],temperature=self.cfg.get('rollout_temperature',.4))
            self.rollout_spent+=1;self.rollouts+=1;remaining-=1
            choice=out.get('choice');valid=type(choice) is int and 0<=choice<len(choices)
            self.events.append(dict(event='rollout_move',state=start.id,choice=choice,valid=valid,raw=raw))
            if not valid:self.invalid+=1;break
            state=State(-1,choices[choice][1],state.depth+1,parent=start.id)
            if len(state.terms)<=2:
                solved=self.endgame(state)
                self.events.append(dict(event='rollout_outcome',state=start.id,solved=solved))
                # Only completed probes affect this branch's confidence.
                start.success=(2*start.success+int(solved))/3
                break

    def rank(self,states,method):
        for st in states:
            bag=Counter(t.value for t in st.terms);sims=[]
            for old in self.expanded:
                other=Counter(t.value for t in old.terms)
                sims.append(sum((bag&other).values())/max(1,sum((bag|other).values())))
            novelty=1-max(sims,default=0)
            # Two-value states need no LLM call: endgame checking is local.
            estimate=0 if len(st.terms)<=2 else self.next_tokens+200*len(st.terms)
            cost=min(1,estimate/self.cfg['cost_reference_tokens'])
            f=dict(quality=st.quality,success=st.success,depth=st.depth/3*(1-st.success),cost=cost,diversity=novelty*math.sqrt(st.quality*st.success))
            h=sum(self.cfg['weights'][k]*v*(-1 if k in ('depth','cost') else 1) for k,v in f.items())
            st.features={**f,'raw_diversity':novelty,'heuristic':h}
        return sorted(states,key=lambda st:(-(st.features['heuristic'] if method=='heuristic_bestfirst' else st.quality if method=='quality_bestfirst' else st.success),st.id))

    def expand(self,state):
        self.expansions+=1;self.expanded.append(state)
        self.events.append(dict(event='expand',state=state.id,features=state.features))
        if len(state.terms)<=2:self.endgame(state);return []
        children=[]
        for move,terms in legal_children(state.terms):
            child=State(len(self.nodes),terms,state.depth+1,quality=0,success=0,parent=state.id)
            if self.key(child) in self.seen:continue
            self.seen.add(self.key(child));self.nodes.append(child);children.append(child)
        self.events.append(dict(event='legal_moves',parent=state.id,children=[c.id for c in children]))
        self.rate(state,children)
        return children

    def run(self,method):
        reason='frontier_empty'
        try:
            if method=='symbolic_search':
                # Explicit oracle baseline, never called by the proposed policy.
                stack=[self.root]
                while stack and not self.answer:
                    state=stack.pop();self.expansions+=1
                    if len(state.terms)<=2:self.endgame(state);continue
                    for _,terms in legal_children(state.terms):
                        child=State(len(self.nodes),terms,state.depth+1,parent=state.id)
                        if self.key(child) not in self.seen:
                            self.seen.add(self.key(child));self.nodes.append(child);stack.append(child)
                reason='verified_solution' if self.answer else 'search_exhausted'
            elif method in ('cot','self_consistency','verified_sampling'):
                for _ in range(1 if method=='cot' else self.cfg['samples']):
                    self.completion(temperature=self.cfg['temperature'])
                    if method=='verified_sampling' and self.answer:break
                reason='sample_limit'
            else:
                frontier=[self.root];reserve=[]
                while not self.answer and self.expansions<self.cfg['max_expansions']:
                    if not frontier:
                        if not reserve:reason='search_exhausted';break
                        frontier=reserve;reserve=[]
                        self.events.append(dict(event='restore_deferred',count=len(frontier)))
                    ranked=self.rank(frontier,method)
                    if method in ('tot_bfs','self_eval_beam'):
                        level=min(st.depth for st in ranked);eligible=[st for st in ranked if st.depth==level]
                        selected=(self.rng.choices(eligible,weights=[math.exp(st.success/self.cfg['beam_temperature']) for st in eligible],k=1)[0]
                                  if method=='self_eval_beam' else eligible[0])
                    else:selected=ranked[0]
                    frontier.remove(selected)
                    # Probes are restricted to genuinely uncertain scored states.
                    if (selected.id!=0 and len(selected.terms)>2 and .35<=selected.success<=.65
                        and self.rollout_spent<self.cfg.get('max_rollout_calls',2)):
                        self.move_rollout(selected)
                    if self.answer:break
                    frontier+=self.expand(selected)
                    limit=self.cfg['beam_width'] if method in ('tot_bfs','self_eval_beam') else self.cfg['max_frontier']
                    ordered=self.rank(frontier,method);frontier=ordered[:limit];reserve+=ordered[limit:]
                else:reason='expansion_limit'
                if self.answer:reason='verified_solution'
        except BudgetExceeded as e:reason=str(e)
        return dict(prediction=self.answer,correct=self.answer is not None,stop_reason=reason,
                    expanded_states=self.expansions,rollout_calls=self.rollouts,invalid_moves=self.invalid,
                    rating_failures=self.rating_failures,endgame_checks=self.endgame_checks,
                    off_branch_solutions=self.off_branch_solutions,
                    events=self.events,nodes=[dict(id=n.id,parent=n.parent,depth=n.depth,
                    terms=[dict(value=str(t.value),expression=t.expression) for t in n.terms],quality=n.quality,success=n.success,features=n.features) for n in self.nodes])


def report(root):
    root=Path(root);rows=read_jsonl(root/'results.jsonl');groups={}
    for r in rows:groups.setdefault(r['question_id'],set()).add(r['method'])
    complete={q for q,ms in groups.items() if ms==set(METHODS)}
    summary=[]
    for method in METHODS:
        g=[r for r in rows if r['method']==method and r['question_id'] in complete];n=len(g)
        summary.append(dict(method=method,n=n,correct=sum(r['correct'] for r in g),accuracy=sum(r['correct'] for r in g)/n if n else None,
                            **{'mean_'+k:sum(r.get(k,0) for r in g)/n if n else 0 for k in ('calls','total_tokens','inference_seconds','execution_seconds','expanded_states','rollout_calls','rating_failures','endgame_checks')},unknown_cost_results=sum(bool(r.get('unknown_cost')) for r in g)))
    for name,data in [('summary.csv',summary),('results_long.csv',rows)]:
        if data:
            fields=list(dict.fromkeys(k for r in data for k in r))
            with (root/name).open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    wide=[]
    for q in groups:
        found=next(r for r in rows if r['question_id']==q);r=dict(question_id=q,numbers=found['numbers'])
        for m in METHODS:
            item=next((x for x in rows if x['question_id']==q and x['method']==m),{})
            for k in ('prediction','correct','calls','total_tokens','stop_reason'):r[m+'__'+k]=item.get(k,'pending')
        wide.append(r)
    if wide:
        with (root/'per_question.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(wide[0]));w.writeheader();w.writerows(wide)
    lines=['# Game of 24 results','',f'Complete {len(METHODS)}-method blocks: {len(complete)}. Saved results: {len(rows)}.','',
           '| Method | Correct/N | Accuracy | Mean calls | Mean tokens |','|---|---:|---:|---:|---:|']
    for r in summary:
        acc=f'{r["accuracy"]:.1%}' if r['n'] else 'pending'
        lines.append(f'| {r["method"]} | {r["correct"]}/{r["n"]} | {acc} | {r["mean_calls"]:.2f} | {r["mean_total_tokens"]:.0f} |')
    lines+=['','Local adaptations using batched model ratings, legal moves and shared exact two-value endgames, not published SOTA reproductions. All rollout tokens count. No learned GSM8K weights are claimed. Exact dataset construction oracle is excluded from policy.','', 'All valid solutions evaluate to 24; final-answer consensus and majority voting cannot distinguish correct solutions. SC is fixed-budget verified sampling; search stops at its first verified solution.']
    lines += ['', 'The four search methods share legal move enumeration and exact two-value endgames. Compare against symbolic_search to measure deterministic assistance. Symbolic search uses zero model calls but consumes CPU time; mean_execution_seconds reports that cost. P/Q are uncalibrated model estimates; improved accuracy has not been established by offline tests.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    body='<h1>Game of 24</h1><pre>'+html.escape('\n'.join(lines))+'</pre><h2>Per puzzle</h2>'
    for r in wide:body+='<pre>'+html.escape(json.dumps(r,indent=2))+'</pre>'
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;margin:30px}pre{white-space:pre-wrap}</style>'+body,encoding='utf-8')


def run(args):
    cfg=config(args.config)
    if args.seed is not None:cfg['seed']=args.seed
    if args.limit<1:raise ValueError('limit must be positive')
    puzzles=read_jsonl(args.dataset)[:args.limit]
    if not puzzles:raise ValueError('Empty dataset: run prepare first')
    if len({p['id'] for p in puzzles})!=len(puzzles):raise ValueError('Duplicate IDs')
    for p in puzzles:
        if len(p['numbers'])!=4 or any(type(n) is not int or not 1<=n<=13 for n in p['numbers']):raise ValueError('Require four integers from 1 to 13')
        if not all(c.isalnum() or c in '-_' for c in p['id']):raise ValueError('Unsafe puzzle ID')
    if len({tuple(sorted(p['numbers'])) for p in puzzles})!=len(puzzles):raise ValueError('Duplicate multiset')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    env=doctor(cfg)
    manifest=dict(task='game24',config=cfg,code_sha256=fingerprint(),dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),ids=[p['id'] for p in puzzles],model_digest=env['model']['digest'])
    meta=root/'manifest.json'
    if meta.exists() and json.loads(meta.read_text())!=manifest:raise ValueError('Experiment changed; use a NEW output directory')
    meta.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lock=root/'run.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    try:
        saved={(r['question_id'],r['method']) for r in read_jsonl(root/'results.jsonl')}
        for index,p in enumerate(puzzles):
            print(f'\nPUZZLE {index+1}/{len(puzzles)} {p["id"]}: {p["numbers"]}',flush=True)
            methods=list(METHODS);shift=index%len(methods);methods=methods[shift:]+methods[:shift]
            for method in methods:
                if (root/'pause.request').exists():
                    (root/'pause.request').unlink();report(root);print('Paused between methods.');return
                if (p['id'],method) in saved:print(f'  {method}: already saved',flush=True);continue
                session=Session(cfg,p['id'],method,cfg['seed'],root/'calls'/f'{p["id"]}_{method}.jsonl',retry_failed=args.retry_failed)
                start=time.perf_counter();result=GameEngine(session,p['numbers']).run(method)
                row=dict(question_id=p['id'],numbers=p['numbers'],seed=cfg['seed'],method=method,status='ok',**result,**session.metrics(),execution_seconds=time.perf_counter()-start)
                trace=root/'traces'/f'{p["id"]}_{method}.json';trace.parent.mkdir(exist_ok=True);trace.write_text(json.dumps(row,indent=2),encoding='utf-8')
                row.pop('nodes');row.pop('events');row['trace']=str(trace.relative_to(root));append_json(root/'results.jsonl',row)
                print(f'  {method:23s} {"SOLVED" if row["correct"] else "FAILED":6s} expression={row["prediction"]} calls={row["calls"]} tokens={row["total_tokens"]}',flush=True)
                report(root)
        print(f'Complete: {root/"report.html"}')
    finally:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--directory',default=str(ROOT/'data/game24'));p.add_argument('--size',type=int,default=100);p.add_argument('--seed',type=int,default=42)
    p=sub.add_parser('report');p.add_argument('--output',required=True)
    p=sub.add_parser('run');p.add_argument('--dataset',default=str(ROOT/'data/game24/game24_test.jsonl'));p.add_argument('--config',default=str(ROOT/'configs/game24.json'));p.add_argument('--output',default=str(ROOT/'results/game24_test100_v2'));p.add_argument('--limit',type=int,default=100);p.add_argument('--seed',type=int);p.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    try:
        if args.command=='prepare':prepare(args.directory,args.size,args.seed)
        elif args.command=='report':report(args.output)
        else:run(args)
    except KeyboardInterrupt:print('Stopped; saved calls can be replayed by the same command.');return 130
    except Exception as e:print(f'ERROR: {e}');return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
