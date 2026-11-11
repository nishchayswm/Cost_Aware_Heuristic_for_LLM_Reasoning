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
DEFAULT = dict(max_tokens=80000, max_calls=120, max_expansions=60,
               max_depth=20, branch_factor=3, max_frontier=60,
               generation_tokens=350, evaluation_tokens=500,
               num_ctx=8192, temperature=0.6, prune_below=0.05, json_retries=2,
               repair_threshold=0.4, max_repairs=3, evaluation_batch=6,
               min_feasibility=0.15, min_final_word_quality=0.5,
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
    """Preserve crossing constraints; alternative choices require backtracking."""
    if not isinstance(slot,int) or not 0<=slot<10 or not re.fullmatch('[A-Z]{5}',word):
        return None
    if words[slot] == word: return None
    result=list(words); result[slot]=''
    target=dict(zip(cells(slot),word))
    for other, old in enumerate(result):
        if old and any(index in target and target[index]!=letter for index,letter in zip(cells(other),old)):
            return None
    result[slot]=word
    return tuple(result)

@dataclass
class State:
    words: tuple = ('',)*10
    depth: int = 0
    q: float = 0.0
    p: float = 0.5
    confidence: float = 0.0
    slot_scores: tuple = (-1.,)*10
    repairs: int = 0
    slot_feasibilities: tuple = (.5,)*10

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
    return (sum(max(0.,q) for q in state.slot_scores)/10, state.p, coverage(state), -state.depth)

def feasible(state,cfg):
    return min(state.slot_feasibilities)>=cfg['min_feasibility']

def final_eligible(state,cfg):
    return feasible(state,cfg) and all(q<0 or q>=cfg['min_final_word_quality'] for q in state.slot_scores)

PROPOSE_EXAMPLES = '''Examples (demonstrations only, not answers for the current puzzle):
Clue h2: Fruit often baked in pies, pattern A_P_E -> {"moves":[{"slot":"h2","word":"APPLE","confidence":0.9}]}
Positions: A P P L E; letters 1, 3, 5 match A, P, E.
Clue v4: Sounds made by a dog, pattern _A_K_ -> {"moves":[{"slot":"v4","word":"BARKS","confidence":0.9}]}
Clue h1: Fruit often baked in pies, pattern QZ___ -> {"moves":[]}
If h1 is uncertain but h2 is easy, propose h2 first. Do not invent strings to satisfy letters.
'''

VALUE_EXAMPLES = '''Calibration examples:
Fruit often baked in pies | APPLE: q=0.95, p=0.95 (fits the meaning).
Fruit often baked in pies | TABLE: q=0.01, p=0.01 (real word, wrong meaning).
Fruit often baked in pies | QZ___: q=0, p=0.01 (no plausible answer fits).
Sounds made by a dog | _A_K_: q=0, p=0.9 (BARKS fits).
Fruit often baked in pies | KTEHJ: q=0, p=0 (not a plausible clue answer).
Use these as scale examples only. Judge the actual pairs independently.
'''

def patterns(words):
    board=board_of(words)
    return [''.join(board[k] for k in cells(i)) for i in range(10)]

def target_slot(state):
    available=[i for i,p in enumerate(patterns(state.words)) if '_' in p]
    return max(available,key=lambda i:(5-patterns(state.words)[i].count('_'),-i)) if available else None

def repair_state(state,cfg):
    """Explicitly retract a suspect answer and one crossing support to open a cell."""
    if state.repairs>=cfg['max_repairs']:return None
    suspects=[i for i,q in enumerate(state.slot_scores) if q>=0]
    if not suspects:return None
    slot=min(suspects,key=lambda i:(state.slot_scores[i],i))
    if state.slot_scores[slot]>=cfg['repair_threshold']:return None
    words=list(state.words);removed=[]
    if words[slot]:words[slot]='';removed.append(slot)
    # If crossing assignments still imply all letters, retract their weakest support.
    if '_' not in patterns(tuple(words))[slot]:
        cross=[i for i,w in enumerate(words) if w and set(cells(i)) & set(cells(slot))]
        if cross:
            weakest=min(cross,key=lambda i:(state.slot_scores[i],i))
            words[weakest]='';removed.append(weakest)
    if not removed:return None
    repaired=State(tuple(words),state.depth+1,repairs=state.repairs+1)
    return repaired,removed

def completed_fraction(state):
    board=board_of(state.words)
    return sum(all(board[k]!='_' for k in cells(i)) for i in range(10))/10

def features(state, frontier, cfg, estimated_call_tokens):
    other=[s for s in frontier if s is not state]
    diversity=min((sum(a!=b for a,b in zip(state.words,s.words))/10 for s in other), default=0.)
    remaining=10*(1-completed_fraction(state))
    # Proxy for future work, not sunk path cost or a calibrated completion forecast.
    cost=min(1., remaining*2*estimated_call_tokens/cfg['max_tokens'])
    return dict(quality=state.q*math.sqrt(completed_fraction(state)), success=state.p,
                depth=min(1.,state.depth/cfg['max_depth'])*(1-state.p),
                cost=cost, diversity=diversity*math.sqrt(state.q*state.p))

def priority(state, frontier, cfg, estimated_call_tokens):
    f=features(state,frontier,cfg,estimated_call_tokens);w=cfg['weights']
    return w['quality']*f['quality']+w['success']*f['success']-w['depth']*f['depth']-w['cost']*f['cost']+w['diversity']*f['diversity']

class BudgetStop(Exception): pass
class InvalidResponse(ValueError): pass

def response_schema(kind, count=3, slot=None, pattern=None):
    score=dict(type='number',minimum=0,maximum=1)
    if kind=='generate':
        key='moves'
        properties=dict(slot=dict(type='string',enum=[slot_name(i) for i in range(10)]),
                        word=dict(type='string',pattern='[A-Za-z]{5}'),confidence=score)
        if slot is not None:properties['slot']['enum']=[slot_name(slot)]
        if pattern is not None:
            properties['word']['pattern']=''.join('[A-Za-z]' if c=='_' else '['+c+c.lower()+']' for c in pattern)
    elif kind=='evaluate':
        key='ratings'
        properties=dict(id=dict(type='integer',enum=list(range(count))),q=score,p=score)
    else:raise ValueError('Unknown response kind: '+kind)
    item=dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
    return dict(type='object',properties={key:dict(type='array',items=item,minItems=0 if kind=='generate' else count,maxItems=count)},
                required=[key],additionalProperties=False)

def validate_response(value,kind,schema):
    key='moves' if kind=='generate' else 'ratings'
    if not isinstance(value,dict) or set(value)!={key}:raise InvalidResponse('Expected only '+key)
    items=value[key];spec=schema['properties'][key]
    if not isinstance(items,list) or not spec['minItems']<=len(items)<=spec['maxItems']:
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
            if field=='word' and not re.fullmatch(rule['pattern'],v):
                raise InvalidResponse('Word violates target crossing pattern')
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

def evaluate_states(states,clues,session,cfg,cache):
    # A rating depends only on the clue and its letter pattern, enabling exact local reuse.
    missing=list(dict.fromkeys((i,p) for state in states for i,p in enumerate(patterns(state.words))
                              if p!='_____' and (i,p) not in cache))
    for begin in range(0,len(missing),cfg['evaluation_batch']):
        batch=missing[begin:begin+cfg['evaluation_batch']]
        prompt=('Rate each clue-pattern pair independently. For a complete word, q is how well it answers '
                'the clue, not whether it is merely an English word. For an incomplete pattern, p is '
                'whether a valid clue answer could fit it; q may be zero. For a complete word set p=q. '
                'Scores range from 0 (contradiction) to 1 (strong fit). Do not assume supplied words are correct. '
                'Return ratings with id, q, p, one for every ID.\n'+
                VALUE_EXAMPLES+'\n'+ '\n'.join(f'ID {j}: clue={clues[i]} | pattern={pat}' for j,(i,pat) in enumerate(batch)))
        result=session.ask(prompt,'evaluate',cfg['evaluation_tokens'],response_schema('evaluate',len(batch)))
        for rating in result['ratings']:
            cache[batch[rating['id']]]=(rating['q'],rating['p'])
    for state in states:
        scores=[];feasibilities=[]
        for i,pat in enumerate(patterns(state.words)):
            q,p=cache.get((i,pat),(0.,.5))
            scores.append(q if '_' not in pat else -1.)
            feasibilities.append(q if '_' not in pat else p)
        state.slot_scores=tuple(scores)
        state.slot_feasibilities=tuple(feasibilities)
        complete=[q for q in scores if q>=0]
        state.q=sum(complete)/len(complete) if complete else 0.
        state.p=math.exp(sum(math.log(max(.001,v)) for v in feasibilities)/10)


def solve(clues,method,session,cfg):
    if len(clues)!=10 or method not in METHODS:raise ValueError('Invalid task or method')
    frontier=[State()];seen={frontier[0].words};incumbent=frontier[0];expanded=0;trace=[];reason='frontier_empty'
    rejected_conflicts=0;duplicate_moves=0;empty_expansions=0;repair_count=0;cache={};selection_path=[];pruned_states=0
    while frontier and expanded<cfg['max_expansions']:
        estimate=session.tokens/max(1,session.calls) if session.calls else 1200
        node=frontier[-1] if method=='tot_dfs' else max(frontier,key=lambda s:priority(s,frontier,cfg,estimate))
        event=dict(expansion=expanded+1,selected=asdict(node),frontier_size=len(frontier),
                   features=features(node,frontier,cfg,estimate),heuristic_score=priority(node,frontier,cfg,estimate))
        frontier.remove(node)
        if node.depth>=cfg['max_depth']:continue
        selection_path.append(digest(node.words));expanded+=1
        children=[];available=[i for i,pat in enumerate(patterns(node.words)) if '_' in pat]
        try:
            if available and feasible(node,cfg):
                event.update(available_slots=[slot_name(i) for i in available])
                prompt=(f'Propose up to {cfg["branch_factor"]} distinct confident five-letter answers across ANY of the incomplete clues. '
                        'Choose the clues you can solve most confidently; do not always start at h1. '
                        'Known letters must match exactly. Return moves with slot, word, confidence in descending confidence. '
                        'Return an empty moves list if no plausible answer fits. Do not invent a word to fill the grid.\n'+
                        PROPOSE_EXAMPLES+'\nACTUAL PUZZLE:\n'+render(clues,node.words))
                schema=response_schema('generate',cfg['branch_factor'])
                schema['properties']['moves']['items']['properties']['slot']['enum']=[slot_name(i) for i in available]
                proposed=session.ask(prompt,'generate',cfg['generation_tokens'],
                                     schema)
                for move in proposed['moves']:
                    name=move['slot'];slot=int(name[1])-1+(5 if name[0]=='v' else 0)
                    words=place(node.words,slot,move['word'].upper())
                    if words is None:rejected_conflicts+=1;continue
                    if words in seen:duplicate_moves+=1;continue
                    seen.add(words)
                    children.append(State(words,node.depth+1,confidence=move['confidence'],repairs=node.repairs))
            evaluate_states(children,clues,session,cfg,cache)
            for child in children:
                if final_eligible(child,cfg) and final_rank(child)>final_rank(incumbent):incumbent=child
            # Repair the selected state as an alternative branch, including full grids.
            repaired=repair_state(node,cfg)
            if repaired is not None:
                child,removed=repaired
                if child.words not in seen:
                    evaluate_states([child],clues,session,cfg,cache)
                    seen.add(child.words);children.append(child);repair_count+=1
                    event['repair_removed']=[slot_name(i) for i in removed]
            if not children:empty_expansions+=1
        except BudgetStop as exc:
            reason=str(exc);event['error']=reason;trace.append(event);break
        except InvalidResponse as exc:
            reason='invalid_model_response';event['error']=str(exc);trace.append(event);break
        # Low-feasibility nodes with suspect words remain eligible for repair.
        # Infeasible states may only enter a repair step; never expand them by adding more words.
        accepted=[c for c in children if (feasible(c,cfg) and c.p>=cfg['prune_below']) or repair_state(c,cfg) is not None]
        pruned_states+=sum(not feasible(c,cfg) for c in children)
        event.update(candidates=[asdict(c) for c in children],accepted=len(accepted),calls=session.calls,tokens=session.tokens)
        trace.append(event)
        if method=='tot_dfs':
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
                truncated_calls=session.truncated,expanded_states=expanded,stop_reason=reason,
                rejected_conflicts=rejected_conflicts,duplicate_moves=duplicate_moves,
                empty_expansions=empty_expansions,filled_fraction=coverage(incumbent),
                repairs_created=repair_count,rated_clue_patterns=len(cache),pruned_states=pruned_states,selection_path=selection_path),trace


def score(board,solution):
    return dict(letter_accuracy=sum(a==b for a,b in zip(board,solution))/25,
                word_accuracy=sum(all(board[k]==solution[k] for k in cells(s)) for s in range(10))/10,
                solved=board==solution)

def report(root):
    root=Path(root);rows=[json.loads(p.read_text()) for p in sorted((root/'records').glob('*.json'))]
    if not rows:return []
    with (root/'results_long.csv').open('w',newline='',encoding='utf-8') as f:
        fields=[k for k in rows[0] if k not in ('selected_state','selection_path')];writer=csv.DictWriter(f,fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    complete={r['question_id'] for r in rows if all(any(x['question_id']==r['question_id'] and x['method']==m for x in rows) for m in METHODS)}
    summaries=[]
    for method in METHODS:
        subset=[r for r in rows if r['method']==method and r['question_id'] in complete]
        row=dict(method=method,n=len(subset))
        for key in ('solved','word_accuracy','letter_accuracy','calls','total_tokens','inference_seconds','expanded_states','parse_failures','rejected_conflicts','duplicate_moves','empty_expansions','filled_fraction','repairs_created','rated_clue_patterns'):
            row['mean_'+key]=sum(r[key] for r in subset)/len(subset) if subset else None
        summaries.append(row)
    with (root/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    paired=[]
    for qid in sorted(complete):
        pair={r['method']:r for r in rows if r['question_id']==qid}
        paired.append(dict(question_id=qid,
                           same_prediction=pair[METHODS[0]]['prediction']==pair[METHODS[1]]['prediction'],
                           same_search_path=pair[METHODS[0]]['selection_path']==pair[METHODS[1]]['selection_path'],
                           **{f'{m}_{k}':pair[m][k] for m in METHODS for k in ('prediction','word_accuracy','letter_accuracy','solved','calls','total_tokens')}))
    if paired:
        with (root/'per_question.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,list(paired[0]));writer.writeheader();writer.writerows(paired)
    write_json(root/'summary.json',summaries)
    return summaries

def run(puzzles,client,cfg,runtime,output,seed=42):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if not 1<=len(puzzles)<=50:raise ValueError('Use 1–50 puzzles')
    if len({p['id'] for p in puzzles})!=len(puzzles):raise ValueError('Duplicate puzzle IDs')
    if set(cfg['weights'])!=set(DEFAULT['weights']) or not math.isclose(sum(cfg['weights'].values()),1):raise ValueError('Invalid weights')
    manifest=dict(config=cfg,runtime=runtime,seed=seed,dataset_sha256=digest(puzzles),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),methods=METHODS)
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
                print(f'  {method:22s} words={result["word_accuracy"]:.0%} letters={result["letter_accuracy"]:.0%} solved={result["solved"]} calls={result["calls"]} tokens={result["total_tokens"]} stop={result["stop_reason"]} invalid_responses={result["parse_failures"]} conflicts={result["rejected_conflicts"]} duplicates={result["duplicate_moves"]} fill={result["filled_fraction"]:.0%}',flush=True)
                print('\n'.join('    '+result['prediction'][i:i+5] for i in range(0,25,5)),flush=True)
                report(root)
            pair=[json.loads((root/'records'/f'{puzzle["id"]}_{m}.json').read_text()) for m in METHODS]
            print(f'  Same grid: {pair[0]["prediction"]==pair[1]["prediction"]}; same selected-state sequence: {pair[0]["selection_path"]==pair[1]["selection_path"]}',flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return report(root)
