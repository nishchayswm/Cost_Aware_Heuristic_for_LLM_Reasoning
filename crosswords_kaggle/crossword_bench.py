"""Two-policy mini-crossword benchmark. Search never receives reference answers."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import re
import time
import urllib.request

METHODS = ('tot_dfs', 'heuristic_bestfirst')
DEFAULT = dict(max_tokens=120000, max_calls=160, max_expansions=60,
               max_depth=20, branch_factor=3, max_frontier=60,
               generation_tokens=350, evaluation_tokens=500,
               num_ctx=8192, temperature=0.6, prune_below=0.05, json_retries=2,
               weights=dict(quality=.35, success=.30, depth=.05, cost=.15, diversity=.15))

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8'); tmp.replace(path)

def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'crossword-research'}), timeout=60) as response:
        return response.read()

def prepare_dataset(root, selection_seed=2026):
    """Pin upstream revision; sample 50 unique grids before any model scoring."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'source_manifest.json'
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        if meta['selection_seed'] != selection_seed:
            raise ValueError('Selection seed changed: use a fresh dataset directory.')
        raw = (root/'mini0505.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == meta['source_sha256']
    else:
        repo = 'princeton-nlp/tree-of-thought-llm'
        commit = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/master'))['sha']
        if not re.fullmatch(r'[0-9a-f]{40}', commit): raise ValueError('Invalid upstream revision')
        url = f'https://raw.githubusercontent.com/{repo}/{commit}/src/tot/data/crosswords/mini0505.json'
        raw = fetch(url)
        license_bytes = fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/LICENSE')
        (root/'mini0505.json').write_bytes(raw)
        (root/'SOURCE_LICENSE.txt').write_bytes(license_bytes)
        meta = dict(repository=repo, commit=commit, url=url, source_sha256=hashlib.sha256(raw).hexdigest(), selection_seed=selection_seed)
    records = json.loads(raw)
    unique = {}; puzzles = []
    for index, (clues, letters) in enumerate(records):
        board = ''.join(letters).upper()
        if len(clues)!=10 or len(board)!=25 or not re.fullmatch('[A-Z]{25}', board):
            raise ValueError(f'Malformed puzzle {index}')
        unique.setdefault(board, dict(id=f'crossword-{index:04d}', source_index=index, clues=clues, solution=board))
    candidates = list(unique.values())
    if len(candidates)<50: raise ValueError('Need at least 50 distinct source puzzles')
    puzzles = random.Random(selection_seed).sample(candidates, 50)
    meta.update(source_count=len(records), unique_grids=len(candidates), count=50,
                selected_ids=[p['id'] for p in puzzles], dataset_sha256=digest(puzzles),
                selection='seeded random 50 unique grids; custom split, not the published ToT subset')
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != meta:
        raise ValueError('Dataset provenance changed')
    write_json(manifest_path, meta)
    write_json(root/'crosswords50.json', puzzles)
    # Search input artifact explicitly excludes answer keys.
    write_json(root/'questions50.json', [{k:v for k,v in p.items() if k!='solution'} for p in puzzles])
    return puzzles

def cells(slot):
    return list(range(slot*5, slot*5+5)) if slot<5 else list(range(slot-5,25,5))

def slot_name(slot):
    return ('h' if slot<5 else 'v')+str(slot%5+1)

def board_of(words):
    board = ['_']*25
    for slot, word in enumerate(words):
        if not word: continue
        for index, letter in zip(cells(slot), word):
            if board[index] not in ('_',letter): raise ValueError('Conflicting assignments')
            board[index] = letter
    return ''.join(board)

def place(words, slot, word):
    """Revision clears conflicting crossing assignments, retaining alternative branches."""
    if not isinstance(slot,int) or not 0<=slot<10 or not re.fullmatch('[A-Z]{5}',word):
        return None
    if words[slot] == word: return None
    result=list(words); result[slot]=''
    target=dict(zip(cells(slot),word))
    for other, old in enumerate(result):
        if old and any(index in target and target[index]!=letter for index,letter in zip(cells(other),old)):
            result[other]=''
    result[slot]=word
    return tuple(result)

@dataclass
class State:
    words: tuple = ('',)*10
    depth: int = 0
    q: float = 0.0
    p: float = 0.5
    confidence: float = 0.0

def bounded(value, default=0.0):
    try:
        x=float(value)
        return max(0.,min(1.,x)) if math.isfinite(x) else default
    except (ValueError,TypeError): return default

def render(clues, words):
    board=board_of(words)
    return '\n'.join(f'{slot_name(i)}: {clues[i]} | pattern={"".join(board[k] for k in cells(i))} | explicit={words[i] or "unfilled"}' for i in range(10))

def coverage(state):
    return sum(c!='_' for c in board_of(state.words))/25

def final_rank(state):
    # Identical answer selection for both methods; reference answers are unavailable.
    return (coverage(state)*state.q, coverage(state), state.p, -state.depth)

def features(state, frontier, cfg, estimated_call_tokens):
    other=[s for s in frontier if s is not state]
    diversity=min((sum(a!=b for a,b in zip(state.words,s.words))/10 for s in other), default=0.)
    remaining=sum(not w for w in state.words)
    # Proxy for future work, not sunk path cost or a calibrated completion forecast.
    cost=min(1., (remaining/10)*2*estimated_call_tokens/cfg['max_tokens'])
    return dict(quality=state.q, success=state.p,
                depth=min(1.,state.depth/cfg['max_depth'])*(1-state.p),
                cost=cost, diversity=diversity*math.sqrt(state.q*state.p))

def priority(state, frontier, cfg, estimated_call_tokens):
    f=features(state,frontier,cfg,estimated_call_tokens);w=cfg['weights']
    return w['quality']*f['quality']+w['success']*f['success']-w['depth']*f['depth']-w['cost']*f['cost']+w['diversity']*f['diversity']

class BudgetStop(Exception): pass
class InvalidResponse(ValueError): pass

def response_schema(kind, count=3):
    score=dict(type='number',minimum=0,maximum=1)
    if kind=='generate':
        key='moves'
        properties=dict(slot=dict(type='string',enum=[slot_name(i) for i in range(10)]),
                        word=dict(type='string',pattern='[A-Za-z]{5}'),confidence=score)
    elif kind=='evaluate':
        key='ratings'
        properties=dict(id=dict(type='integer',enum=list(range(count))),q=score,p=score)
    else:raise ValueError('Unknown response kind: '+kind)
    item=dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
    return dict(type='object',properties={key:dict(type='array',items=item,minItems=1,maxItems=count)},
                required=[key],additionalProperties=False)

def validate_response(value,kind,schema):
    key='moves' if kind=='generate' else 'ratings'
    if not isinstance(value,dict) or set(value)!={key}:raise InvalidResponse('Expected only '+key)
    items=value[key];spec=schema['properties'][key]
    if not isinstance(items,list) or not 1<=len(items)<=spec['maxItems']:
        raise InvalidResponse('Invalid '+key+' count')
    for item in items:
        props=spec['items']['properties']
        if not isinstance(item,dict) or set(item)!=set(props):raise InvalidResponse('Unexpected fields')
        for field,rule in props.items():
            v=item[field]
            if rule['type']=='number' and (type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1):
                raise InvalidResponse('Score must be finite and in [0,1]')
            if rule['type']=='integer' and type(v) is not int:raise InvalidResponse('ID must be integer')
            if 'enum' in rule and v not in rule['enum']:raise InvalidResponse('Invalid '+field)
            if field=='word' and (not isinstance(v,str) or not re.fullmatch('[A-Za-z]{5}',v)):
                raise InvalidResponse('Every word must contain exactly five English letters')
    if kind=='evaluate' and sorted(item['id'] for item in items)!=spec['items']['properties']['id']['enum']:
        raise InvalidResponse('Rate each candidate ID exactly once')
    return value

class HFClient:
    """Exact-token preflight and schema-constrained decoding for crossword calls."""
    def __init__(self,model,tokenizer,torch):
        self.model=model;self.tokenizer=tokenizer;self.torch=torch

    def encode(self,prompt):
        messages=[dict(role='system',content='Solve 5x5 crosswords. Return only the requested JSON object, without markdown.'),dict(role='user',content=prompt)]
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        return self.tokenizer(text,return_tensors='pt',add_special_tokens=False)

    def count(self,prompt):return int(self.encode(prompt)['input_ids'].shape[1])

    def generate(self,prompt,seed,limit,temperature,schema=None):
        torch=self.torch
        inputs=self.encode(prompt).to(self.model.device);n=int(inputs['input_ids'].shape[1])
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        kwargs=dict(max_new_tokens=limit,do_sample=temperature>0,pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id)
        if temperature>0:kwargs.update(temperature=temperature,top_k=0,top_p=1.)
        if schema is not None:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn)
            if not hasattr(self,'_constraint_tokenizer'):
                self._constraint_tokenizer=build_token_enforcer_tokenizer_data(self.tokenizer)
            kwargs['prefix_allowed_tokens_fn']=build_transformers_prefix_allowed_tokens_fn(
                self._constraint_tokenizer,JsonSchemaParser(schema))
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():output=self.model.generate(**inputs,**kwargs)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        generated=output[0,n:];ids=generated.tolist()
        eos=self.tokenizer.eos_token_id
        stopped=bool(ids) and ids[-1] in (eos if isinstance(eos,list) else [eos])
        return dict(text=self.tokenizer.decode(generated,skip_special_tokens=True),input_tokens=n,
                    output_tokens=len(ids),seconds=elapsed,truncated=len(ids)>=limit and not stopped)

class Session:
    def __init__(self,client,cfg,root,qid,seed):
        self.client=client;self.cfg=cfg;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.qid=qid;self.seed=seed
        self.calls=0;self.tokens=0;self.input_tokens=0;self.output_tokens=0;self.seconds=0.;self.parse_failures=0;self.truncated=0

    def ask(self,prompt,kind,limit,schema=None):
        schema=schema or response_schema(kind,self.cfg['branch_factor'])
        feedback=''
        for attempt in range(self.cfg.get('json_retries',2)+1):
            try:return self._ask_once(prompt+feedback,kind,limit,schema)
            except InvalidResponse as exc:
                if attempt==self.cfg.get('json_retries',2):raise
                feedback=('\nPrevious response failed validation: '+str(exc)+
                          '\nReturn one complete JSON object matching the requested fields. '
                          'Every word must contain exactly five letters. No markdown or conversation. '
                          f'Retry {attempt+1}.')

    def _ask_once(self,prompt,kind,limit,schema):
        n=self.client.count(prompt)
        # Reserve full completion allowance before inference; never overspend silently.
        if self.calls>=self.cfg['max_calls'] or self.tokens+n+limit>self.cfg['max_tokens']:
            raise BudgetStop('token_or_call_budget')
        if n+limit>self.cfg['num_ctx']:raise BudgetStop('context_limit')
        seed=int(digest([self.qid,self.seed,prompt,kind])[:8],16) % (2**31)
        request=dict(prompt=prompt,kind=kind,seed=seed,limit=limit,temperature=self.cfg['temperature'],schema=schema)
        path=self.root/f'{self.calls:03d}.json'
        if path.exists():
            record=json.loads(path.read_text())
            if record['request']!=request:raise ValueError('Replay mismatch; use a new output directory')
            response=record['response']
        else:
            pending=path.with_suffix('.pending')
            if pending.exists():
                raise RuntimeError('An earlier inference was interrupted or failed; its cost is unknown. Start a fresh output trial.')
            pending.write_text('Inference started; no saved response yet.',encoding='utf-8')
            response=self.client.generate(prompt,seed,limit,self.cfg['temperature'],schema=schema)
            write_json(path,dict(request=request,response=response))
            pending.unlink()
        if response['input_tokens']!=n or response['output_tokens']>limit:raise ValueError('Backend accounting mismatch')
        self.calls+=1;self.tokens+=n+response['output_tokens'];self.seconds+=response['seconds']
        self.input_tokens+=n;self.output_tokens+=response['output_tokens']
        self.truncated+=bool(response['truncated'])
        try:
            if response['truncated']:raise InvalidResponse('Truncated JSON response')
            value=json.loads(response['text'])
            return validate_response(value,kind,schema)
        except (ValueError,TypeError) as exc:
            self.parse_failures+=1
            raise InvalidResponse(str(exc)) from exc

def solve(clues,method,session,cfg):
    if len(clues)!=10 or method not in METHODS:raise ValueError('Invalid task or method')
    frontier=[State()];seen={frontier[0].words};incumbent=frontier[0];expanded=0;trace=[];reason='frontier_empty'
    while frontier and expanded<cfg['max_expansions']:
        estimate=session.tokens/max(1,session.calls) if session.calls else 1200
        if method=='tot_dfs':node=frontier[-1]
        else:
            node=max(frontier,key=lambda s:priority(s,frontier,cfg,estimate))
        selected_features=features(node,frontier,cfg,estimate)
        selected_priority=priority(node,frontier,cfg,estimate)
        frontier.remove(node)
        if node.depth>=cfg['max_depth']:continue
        expanded+=1
        prompt=(f'Propose up to {cfg["branch_factor"]} distinct promising next word placements for this grid. '
                'Each answer is exactly five English letters. You may revise an existing answer; '
                'conflicting crossing assignments will be cleared. Prefer matching known letters. '
                'Return only JSON with key moves: a list of objects with slot (h1-h5 or v1-v5), '
                'word (exactly five A-Z letters), and confidence (a number from 0 to 1). '
                'Answer the actual clues below.\n'+render(clues,node.words))
        try:
            proposed=session.ask(prompt,'generate',cfg['generation_tokens'])
            children=[]
            moves=proposed.get('moves',[])
            if not isinstance(moves,list):moves=[]
            for move in moves[:cfg['branch_factor']]:
                if not isinstance(move,dict):continue
                name=str(move.get('slot','')).lower();word=str(move.get('word','')).upper().strip()
                if not re.fullmatch('[hv][1-5]',name):continue
                slot=int(name[1])-1+(5 if name[0]=='v' else 0)
                words=place(node.words,slot,word)
                if words is None or words in seen:continue
                seen.add(words);children.append(State(words,node.depth+1,confidence=bounded(move.get('confidence'))))
            if not children:
                trace.append(dict(expansion=expanded,selected=asdict(node),children=0));continue
            prompt=('Evaluate each candidate grid against ALL ten clues, including implied crossing words. '
                    'Q is mean clue-answer plausibility for slots having all five letters. '
                    'P is estimated feasibility of completing/correctly satisfying the whole puzzle. '
                    'Use scores in [0,1]. An impossible remaining clue should make P very low. '
                    'Return {"ratings":[{"id":0,"q":0.7,"p":0.5}]}, one rating per candidate.\n'+
                    '\n\n'.join(f'Candidate {i}\n{render(clues,c.words)}' for i,c in enumerate(children)))
            evaluated=session.ask(prompt,'evaluate',cfg['evaluation_tokens'],response_schema('evaluate',len(children)))
        except BudgetStop as exc:
            reason=str(exc);break
        except InvalidResponse as exc:
            reason='invalid_model_response'
            trace.append(dict(expansion=expanded,error=str(exc),calls=session.calls,tokens=session.tokens))
            break
        ratings=evaluated.get('ratings',[])
        if not isinstance(ratings,list):ratings=[]
        by_id={}
        for r in ratings:
            if isinstance(r,dict) and type(r.get('id')) is int and 'q' in r and 'p' in r:
                by_id.setdefault(r['id'],r)
        accepted=[]
        for i,child in enumerate(children):
            if i not in by_id:continue  # An unevaluated node cannot earn an invented high score.
            child.q=bounded(by_id[i]['q']);child.p=bounded(by_id[i]['p'])
            if final_rank(child)>final_rank(incumbent):incumbent=child
            # Full boards remain candidates, never assumed correct by a reference oracle.
            if child.p>=cfg['prune_below']:accepted.append(child)
        trace.append(dict(expansion=expanded,selected=asdict(node),features=selected_features,heuristic_score=selected_priority,candidates=[asdict(c) for c in children],
                          accepted=len(accepted),calls=session.calls,tokens=session.tokens))
        if method=='tot_dfs':
            # Highest-confidence child explored first; DFS preserves other branches for backtracking.
            frontier.extend(sorted(accepted,key=lambda s:s.confidence))
            frontier=frontier[-cfg['max_frontier']:]
        else:
            frontier.extend(accepted)
            estimate=session.tokens/max(1,session.calls)
            scores={id(s):priority(s,frontier,cfg,estimate) for s in frontier}
            frontier=sorted(frontier,key=lambda s:scores[id(s)],reverse=True)[:cfg['max_frontier']]
    else:
        reason='expansion_budget' if expanded>=cfg['max_expansions'] else 'frontier_empty'
    return dict(prediction=board_of(incumbent.words),selected_state=asdict(incumbent),calls=session.calls,
                input_tokens=session.input_tokens,output_tokens=session.output_tokens,
                total_tokens=session.tokens,inference_seconds=session.seconds,parse_failures=session.parse_failures,
                truncated_calls=session.truncated,expanded_states=expanded,stop_reason=reason),trace

def score(board,solution):
    return dict(letter_accuracy=sum(a==b for a,b in zip(board,solution))/25,
                word_accuracy=sum(all(board[k]==solution[k] for k in cells(s)) for s in range(10))/10,
                solved=board==solution)

def report(root):
    root=Path(root);rows=[json.loads(p.read_text()) for p in sorted((root/'records').glob('*.json'))]
    if not rows:return []
    with (root/'results_long.csv').open('w',newline='',encoding='utf-8') as f:
        fields=[k for k in rows[0] if k!='selected_state'];writer=csv.DictWriter(f,fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    complete={r['question_id'] for r in rows if all(any(x['question_id']==r['question_id'] and x['method']==m for x in rows) for m in METHODS)}
    summaries=[]
    for method in METHODS:
        subset=[r for r in rows if r['method']==method and r['question_id'] in complete]
        row=dict(method=method,n=len(subset))
        for key in ('solved','word_accuracy','letter_accuracy','calls','total_tokens','inference_seconds','expanded_states','parse_failures'):
            row['mean_'+key]=sum(r[key] for r in subset)/len(subset) if subset else None
        summaries.append(row)
    with (root/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    paired=[]
    for qid in sorted(complete):
        pair={r['method']:r for r in rows if r['question_id']==qid}
        paired.append(dict(question_id=qid,**{f'{m}_{k}':pair[m][k] for m in METHODS for k in ('word_accuracy','letter_accuracy','solved','calls','total_tokens')}))
    if paired:
        with (root/'per_question.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,list(paired[0]));writer.writeheader();writer.writerows(paired)
    write_json(root/'summary.json',summaries)
    return summaries

def run(puzzles,client,cfg,runtime,output,seed=42):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if not 1<=len(puzzles)<=50:raise ValueError('Select between 1 and 50 puzzles from the fixed pool')
    if len({p['id'] for p in puzzles})!=len(puzzles):raise ValueError('Duplicate puzzle IDs')
    if set(cfg['weights'])!=set(DEFAULT['weights']) or not math.isclose(sum(cfg['weights'].values()),1):raise ValueError('Invalid weights')
    manifest=dict(config=cfg,runtime=runtime,seed=seed,dataset_sha256=digest(puzzles),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),methods=METHODS,
                  question_ids=[p['id'] for p in puzzles],puzzle_count=len(puzzles))
    manifest=json.loads(json.dumps(manifest))
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Experiment changed: use new output directory')
    write_json(path,manifest)
    lock=root/'run.lock'
    with lock.open('x') as handle:handle.write(str(seed))
    try:
        for index,puzzle in enumerate(puzzles):
            print(f'\nPUZZLE {index+1}/{len(puzzles)} {puzzle["id"]}',flush=True)
            order=METHODS if index%2==0 else METHODS[::-1]
            for method in order:
                path=root/'records'/f'{puzzle["id"]}_{method}.json'
                if path.exists():result=json.loads(path.read_text())
                else:
                    session=Session(client,cfg,root/'calls'/puzzle['id']/method,puzzle['id'],seed)
                    result,trace=solve(puzzle['clues'],method,session,cfg)
                    # Ground truth is introduced only AFTER search returns its chosen grid.
                    result.update(score(result['prediction'],puzzle['solution']))
                    result.update(question_id=puzzle['id'],method=method,seed=seed,solution=puzzle['solution'])
                    write_json(root/'traces'/f'{puzzle["id"]}_{method}.json',trace)
                    write_json(path,result)
                print(f'  {method:22s} words={result["word_accuracy"]:.0%} letters={result["letter_accuracy"]:.0%} solved={result["solved"]} calls={result["calls"]} tokens={result["total_tokens"]} stop={result["stop_reason"]} invalid_responses={result["parse_failures"]}',flush=True)
                report(root)
    finally:
        lock.unlink(missing_ok=True)
    return report(root)
