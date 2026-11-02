"""Shared, traceable benchmark implementation; also embedded in the Kaggle notebook.

No solver receives the reference answer until scoring. Unfinished branches never
become answers. Arithmetic checks verify equalities, not word-problem semantics.
"""
import ast
import heapq
import itertools
import json
import math
import operator
import re
import time
from dataclasses import dataclass

VERSION = '4'
METHODS = ('Standard CoT (1-Call)', 'Step-wise Greedy CoT',
           'Tree-of-Thoughts (LLM-Judge)', 'Cost-Aware Priority Search',
           'Ablation: Quality-Only (w3=0)', 'Thought of Search (Arithmetic Adaptation)')

STEP_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'step': {'type': 'string'}, 'equation': {'type': 'string'},
        'done': {'type': 'boolean'}, 'answer': {'type': ['number', 'null']},
        'target': {'type': 'string'},
        'expression': {'type': 'string'}, 'result': {'type': ['number', 'null']},
    }, 'required': ['step', 'equation', 'done', 'answer', 'target'],
}
REVIEW_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'valid': {'type': 'boolean'}, 'answer': {'type': 'number'},
                   'reason': {'type': 'string'}},
    'required': ['valid', 'answer', 'reason'],
}
JUDGE_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'score': {'type': 'number', 'minimum': 0, 'maximum': 1}},
    'required': ['score'],
}


@dataclass(frozen=True)
class Config:
    seed: int = 11
    max_depth: int = 10
    beam: int = 3
    candidates: int = 2
    max_expansions: int = 40
    max_calls: int = 240
    retries: int = 2
    step_tokens: int = 512
    final_tokens: int = 1536
    max_tokens: int = 4096
    w1: float = 0.5
    audit_tos: bool = False
    review_appeal: bool = False

    def __post_init__(self):
        for key in ('max_depth', 'beam', 'candidates', 'max_expansions', 'max_calls',
                    'step_tokens', 'final_tokens', 'max_tokens'):
            if getattr(self, key) < 1:
                raise ValueError(key + ' must be positive')
        if self.retries < 0 or not 0 <= self.w1 <= 1:
            raise ValueError('invalid retries or w1')


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def safe_arithmetic(expr):
    """Small arithmetic language; reject prose and bound expensive operations."""
    if not isinstance(expr, str) or len(expr) > 256:
        return None
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}

    def evaluate(node):
        if isinstance(node, ast.Constant) and numeric(node.value):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError('exponent limit')
            value = ops[type(node.op)](left, right)
        else:
            raise ValueError('unsupported arithmetic')
        if not numeric(value) or abs(value) > 1e15:
            raise ValueError('arithmetic limit')
        return value

    try:
        tree = ast.parse(expr.strip().replace('$', '').replace(',', ''), mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 80:
            return None
        return float(evaluate(tree.body))
    except (ValueError, SyntaxError, ArithmeticError, TypeError, RecursionError):
        return None


def equation_status(equation):
    if not equation or equation.lower() == 'none':
        return 'missing', None
    if equation.count('=') != 1:
        return 'unsupported', None
    lhs, rhs = map(safe_arithmetic, equation.split('='))
    if lhs is None or rhs is None:
        return 'unsupported', None
    return ('valid' if math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-6) else 'invalid'), rhs


def equation_key(equation):
    try:
        return '|'.join(ast.dump(ast.parse(s.strip().replace('$', '').replace(',', ''), mode='eval'))
                        for s in equation.split('='))
    except SyntaxError:
        return equation.lower().replace(' ', '')


def parse_object(raw):
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError('expected one JSON object')
    return obj


def validate_step(obj):
    if not set(STEP_SCHEMA['required']).issubset(obj) or not set(obj).issubset(STEP_SCHEMA['properties']):
        raise ValueError('missing or unexpected fields')
    if any(not isinstance(obj[k], str) for k in ('step', 'equation', 'target')):
        raise ValueError('text fields must be strings')
    if not isinstance(obj['done'], bool):
        raise ValueError('invalid done flag')
    expression = obj.get('expression', '').strip()
    result = obj.get('result')
    if expression or result is not None:
        computed = safe_arithmetic(expression)
        if computed is None or not numeric(result) or not math.isclose(
                computed, result, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('expression_result_mismatch')
        obj = dict(obj, equation=f'{expression} = {result}')
    if not obj['step'].strip():
        if equation_status(obj['equation'])[0] != 'valid' or not obj['target'].strip():
            raise ValueError('empty step without verified calculation')
        obj = dict(obj, step='Computed ' + obj['target'].strip())
    if obj['done']:
        if not numeric(obj['answer']) or not obj['target'].strip():
            raise ValueError('completed answer needs a number and target')
    elif obj['answer'] is not None:
        raise ValueError('unfinished step must have null answer')
    # A chain is acceptable only when EVERY expression has the same numeric value.
    # Keep the original response in the request trace; normalize the validated object.
    if obj['equation'].count('=') > 1:
        parts = obj['equation'].split('=')
        if len(parts) > 8 or len(obj['equation']) > 512:
            raise ValueError('equation_chain_limit')
        values = [safe_arithmetic(part) for part in parts]
        if any(value is None for value in values):
            raise ValueError('equation_unsupported')
        if not all(math.isclose(values[0], value, rel_tol=1e-9, abs_tol=1e-6) for value in values[1:]):
            raise ValueError('equation_invalid')
        obj = dict(obj, equation=parts[0].strip() + ' = ' + parts[-1].strip())
    status, _ = equation_status(obj['equation'])
    if status in ('unsupported', 'invalid'):
        raise ValueError('equation_' + status)
    if not obj['done'] and status != 'valid':
        raise ValueError('arithmetic step needs a valid equation')
    return obj


def extract_explicit_answer(text):
    """Only a complete final marker; never guess from intermediate numbers."""
    matches = re.findall(r'^\s*(?:\*\*)?FINAL ANSWER:\s*(?:\*\*)?\s*\$?\s*'
                         r'([+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*(?:\*\*)?\s*$',
                         text, re.MULTILINE | re.IGNORECASE)
    if len(matches) != 1:
        return None
    value = float(matches[0].replace(',', ''))
    return value if numeric(value) else None


class Solver:
    """backend(prompt, schema, temperature, seed, max_tokens) -> response dict.

    Response keys: text, prompt_tokens, completion_tokens, done_reason.
    All real requests including retries and terminal review count toward cost.
    """
    def __init__(self, backend, config=None):
        self.backend = backend
        self.cfg = config or Config()
        self.events = []
        self.calls = self.tokens = 0
        self.failure = None
        self.rejections = {}
        self.review_cache = {}

    def request(self, prompt, schema, temp, purpose, limit):
        feedback = ''
        for attempt in range(self.cfg.retries + 1):
            if self.calls >= self.cfg.max_calls:
                self.failure = 'call_limit'
                return None
            seed = (self.cfg.seed + self.calls) % (2**31)
            self.calls += 1
            event = {'kind': 'request', 'purpose': purpose, 'attempt': attempt,
                     'seed': seed, 'prompt': prompt + feedback, 'token_limit': limit,
                     'schema': schema, 'temperature': temp}
            self.events.append(event)
            try:
                response = self.backend(prompt + feedback, schema, temp, seed, limit)
                event.update(response)
                self.tokens += response.get('prompt_tokens', 0) + response.get('completion_tokens', 0)
                if response.get('done_reason') == 'length':
                    raise ValueError('truncated')
                raw = response['text']
                obj = parse_object(raw) if schema else extract_explicit_answer(raw)
                if schema == STEP_SCHEMA:
                    obj = validate_step(obj)
                elif schema == REVIEW_SCHEMA:
                    if (not isinstance(obj.get('valid'), bool) or not numeric(obj.get('answer'))
                            or not isinstance(obj.get('reason'), str)):
                        raise ValueError('invalid review')
                elif schema == JUDGE_SCHEMA:
                    if not numeric(obj.get('score')):
                        raise ValueError('invalid judge score')
                    score = obj['score']
                    if -1e-9 <= score <= 1 + 1e-9 and not 0 <= score <= 1:
                        obj['score'] = min(1.0, max(0.0, score))
                        event['score_clamped_from'] = score
                    elif not 0 <= score <= 1:
                        raise ValueError('invalid judge score')
                if obj is None:
                    raise ValueError('missing_final_answer')
                event['parsed'] = obj
                self.failure = None
                return obj
            except Exception as exc:
                event['error'] = str(exc)
                self.failure = str(exc)
                instruction = (
                    'Return the requested JSON search model with valid function definitions.'
                    if purpose == 'tos_model_synthesis' else
                    'Return calculations, result and target in the executable-equations JSON schema.'
                    if purpose == 'executable_equations' else
                    'Return valid (boolean), answer (number), and reason (string).'
                    if schema == REVIEW_SCHEMA else
                    'Return a score from zero to one in the JSON schema.'
                    if schema == JUDGE_SCHEMA else
                    'Return step, equation, done, answer and target. Include expression and numeric result '
                    'as separate fields; their arithmetic must agree. Use a numeric arithmetic equality; '
                    'done is a boolean and answer is null until the final quantity is solved.'
                    if schema == STEP_SCHEMA else
                    'Finish with FINAL ANSWER: <decimal number> on its own line.')
                feedback = ('\nPrevious response: ' + str(event.get('text', ''))[:2000]
                            + '\nYour previous response was rejected: ' + str(exc) + '. ' + instruction)
                if str(exc) == 'truncated':
                    limit = min(limit * 2, self.cfg.max_tokens)
        return None

    def terminal_review(self, question, steps, candidate):
        # No expected answer enters this prompt. This common check costs real tokens.
        key = (tuple(equation_key(step['equation']) for step in steps + [candidate]),
               candidate['answer'], candidate['target'].strip().lower())
        if key in self.review_cache:
            self.events.append({'kind': 'terminal_cache_hit', 'candidate': candidate})
            return self.review_cache[key]
        prompt = (f'Independently solve this problem: {question}\n'
                  f'Proposed reasoning: {json.dumps(steps + [candidate])}\n'
                  'Check the requested quantity, units, use of givens, and complete reasoning. '
                  'An intermediate subtotal, speed instead of distance, or number copied from '
                  'the question is insufficient. Return valid=true only if the proposed numeric '
                  'answer and target solve the whole problem. Supply your independently computed '
                  'answer and explain your decision. Do not merely check the last equation.')
        review = self.request(prompt, REVIEW_SCHEMA, 0.1, 'terminal_review', self.cfg.final_tokens)
        accepted = (review is not None and review['valid'] and
                    math.isclose(review['answer'], candidate['answer'], rel_tol=0, abs_tol=1e-4))
        if self.cfg.review_appeal and not accepted and self.calls < self.cfg.max_calls:
            # One independent appeal can recover a false negative. Both votes are logged.
            appeal = self.request(prompt + '\nRe-evaluate independently. Check every quantity '
                                  'and the final arithmetic before deciding.',
                                  REVIEW_SCHEMA, 0.1, 'terminal_review_appeal', self.cfg.final_tokens)
            accepted = (appeal is not None and appeal['valid'] and
                        math.isclose(appeal['answer'], candidate['answer'], rel_tol=0, abs_tol=1e-4))
        else:
            appeal = None
        self.events.append({'kind': 'terminal_decision', 'candidate': candidate,
                            'review': review, 'appeal': appeal, 'accepted': accepted})
        self.review_cache[key] = accepted
        return accepted

    def solve(self, question, method):
        if method == METHODS[5]:
            from tos_search import solve_thought_of_search
            return solve_thought_of_search(self, question)
        started = time.time()
        answer, status, best = None, 'exhausted', []
        if method == METHODS[0]:
            # Exactly one baseline request: truncation is diagnosed, never silently retried.
            from dataclasses import replace
            previous = self.cfg
            self.cfg = replace(previous, retries=0)
            answer = self.request(
                f'Problem: {question}\nSolve fully with clear reasoning. Finish on a new line '
                'with exactly FINAL ANSWER: <number>. Use a decimal number without units '
                'on that line.', None, 0.1, 'standard_cot', self.cfg.max_tokens)
            self.cfg = previous
            status = 'done' if answer is not None else (self.failure or 'missing_final_answer')
        else:
            greedy = method == METHODS[1]
            judge = method == METHODS[2]
            width = 1 if greedy else self.cfg.beam
            n = 1 if greedy else self.cfg.candidates
            w3 = 0.1 if method == METHODS[3] else 0.0
            counter = itertools.count()
            heap = [(-1., next(counter), [])]
            seen_paths = set()
            best_quality = -1.
            expansions = children = 0
            stalled = {}
            while heap and expansions < self.cfg.max_expansions and self.calls < self.cfg.max_calls:
                neg, _, steps = heapq.heappop(heap)
                if len(steps) >= self.cfg.max_depth:
                    continue
                expansions += 1
                self.events.append({'kind': 'expand', 'priority': -neg, 'steps': steps,
                                    'depth': len(steps), 'expansion': expansions})
                history = '\n'.join(s['target'] + ': ' + s['equation'] for s in steps)
                prompt = (f'Problem: {question}\nSteps completed: {history}\n'
                          'Give ONE new arithmetic step. Use the step schema. equation must '
                          'contain only numbers, parentheses, arithmetic operators and one =, '
                          'e.g. 3 * 12 = 36. No units, variables or prose in equation. '
                          'Use // and arithmetic to express rounding when needed. '
                          'Do not repeat a previous calculation. done is a boolean. '
                          'Until the entire requested quantity is solved, answer=null and done=false. '
                          'When solved, done=true, answer is the numeric final answer, target '
                          'describes the requested quantity and units; equation may be empty '
                          'if an earlier step already calculated that answer. Set done=true in the '
                          'same response that computes the requested final quantity. Explain the result in step. '
                          'Include expression as a numeric arithmetic expression and result as its numeric value '
                          'for every computed step.')
                branch_key = json.dumps(steps, sort_keys=True)
                if branch_key in stalled:
                    prompt += '\nThis branch stalled: ' + stalled[branch_key]['reason']
                    prompt += '. Produce a valid new calculation, or mark the computed final answer done=true.'
                if branch_key in self.rejections:
                    prompt += '\nEarlier terminal answers were rejected: ' + json.dumps(self.rejections[branch_key])
                    prompt += '. Recheck the requested quantity and continue with a new calculation.'
                candidates = []
                for _ in range(n):
                    c = self.request(prompt, STEP_SCHEMA, 0.1 if greedy else 0.7,
                                     'generate', self.cfg.step_tokens)
                    if c is None:
                        continue
                    eq_state, rhs = equation_status(c['equation'])
                    repeated = not c['done'] and any(
                        c['step'].strip().lower() == s['step'].strip().lower() or
                        equation_key(c['equation']) == equation_key(s['equation'])
                        for s in steps)
                    if repeated or c in candidates:
                        self.events.append({'kind': 'rejected', 'reason': 'repeated_step', 'candidate': c})
                        # One bounded replacement request with explicit feedback.
                        c = self.request(prompt + '\nThe previous candidate repeated a calculation. '
                                         'Advance to a new calculation or declare the final answer.',
                                         STEP_SCHEMA, 0.7, 'repeat_recovery', self.cfg.step_tokens)
                        if c is None or c in candidates or (not c['done'] and any(
                            c['step'].strip().lower() == s['step'].strip().lower() or
                            equation_key(c['equation']) == equation_key(s['equation'])
                            for s in steps)):
                            continue
                    candidates.append(c)
                if not candidates:
                    attempt = stalled.get(branch_key, {}).get('attempts', 0)
                    reason = self.failure or 'all candidates repeated earlier calculations'
                    self.events.append({'kind': 'branch_stalled', 'reason': reason, 'steps': steps})
                    if attempt < self.cfg.retries:
                        stalled[branch_key] = {'attempts': attempt + 1, 'reason': reason}
                        heapq.heappush(heap, (neg, next(counter), steps))
                children += len(candidates)
                texts = [c['step'] + ' ' + c['target'] for c in candidates]
                relevance = lexical_relevance(texts, question)
                completed = []
                for c, t in zip(candidates, relevance):
                    eq_state, rhs = equation_status(c['equation'])
                    r = 1.0 if eq_state == 'valid' else 0.25
                    observed_output = [e['completion_tokens'] for e in self.events
                                       if e.get('purpose') == 'generate'
                                       and isinstance(e.get('completion_tokens'), (int, float))]
                    expected_output = (sum(observed_output) / len(observed_output)
                                       if observed_output else 192)
                    next_history = prompt + '\n' + c['target'] + ': ' + c['equation']
                    next_prompt = len(next_history.encode('utf-8')) / 3
                    cost = next_prompt + expected_output
                    score = r * (self.cfg.w1 + (1-self.cfg.w1)*t) / (1+w3*cost/500)
                    if judge:
                        rating = self.request(
                            f'Problem: {question}\nPrior steps: {json.dumps(steps)}\n'
                            f'Candidate: {json.dumps(c)}\nRate progress and correctness from 0 to 1.',
                            JUDGE_SCHEMA, 0.1, 'judge', self.cfg.step_tokens)
                        if rating is None:
                            continue
                        score = rating['score']
                    self.events.append({'kind': 'candidate_score', 'candidate': c, 'depth': len(steps),
                                        'equation_status': eq_state, 'r': r, 't': t,
                                        'cost': cost, 'score': score})
                    if c['done']:
                        # Final numeric answer must be grounded in a verified calculation.
                        values = [equation_status(s['equation'])[1] for s in steps + [c]]
                        grounded = any(v is not None and math.isclose(v, c['answer'], abs_tol=1e-4,
                                                                    rel_tol=0) for v in values)
                        if grounded and self.terminal_review(question, steps, c):
                            completed.append((score, c, steps + [c]))
                        else:
                            self.events.append({'kind': 'rejected', 'reason': 'terminal_unverified',
                                                'candidate': c, 'grounded': grounded})
                            reasons = self.rejections.setdefault(branch_key, [])
                            reasons.append(c)
                            # Revisit this branch at most twice; global budgets still apply.
                            if len(reasons) <= self.cfg.retries:
                                heapq.heappush(heap, (-score, next(counter), steps))
                        continue
                    path = steps + [c]
                    if score > best_quality:
                        best_quality, best = score, path
                    signature = tuple(equation_key(s['equation']) for s in path)
                    if signature not in seen_paths:
                        seen_paths.add(signature)
                        heapq.heappush(heap, (-score, next(counter), path))
                if completed:
                    _, chosen, best = max(completed, key=lambda item: item[0])
                    answer, status = chosen['answer'], 'done'
                    break
                heap = heapq.nsmallest(width, heap)
                heapq.heapify(heap)
            if answer is None:
                status = ('call_limit' if self.calls >= self.cfg.max_calls else
                          'expansion_limit' if expansions >= self.cfg.max_expansions else
                          'depth_limit' if best and len(best) >= self.cfg.max_depth else 'exhausted')
        return {'method': method, 'ans': answer, 'status': status, 'steps': best,
                'calls': self.calls, 'tokens': self.tokens, 'sec': time.time()-started,
                'trace': self.events, 'last_error': self.failure, 'seed': self.cfg.seed,
                'config': vars(self.cfg), 'engine_version': VERSION}


def lexical_relevance(texts, question):
    if not texts:
        return []
    # Smooth IDF and L2 normalization, equivalent to the original vectorizer's
    # default weighting, using only stdlib so offline regressions need no model.
    from collections import Counter
    docs = [Counter(re.findall(r'\b\w\w+\b', text.lower())) for text in texts + [question]]
    frequency = Counter(word for doc in docs for word in doc)
    vectors = []
    for doc in docs:
        weighted = {word: count * (1 + math.log((1+len(docs))/(1+frequency[word])))
                    for word, count in doc.items()}
        norm = math.sqrt(sum(weight**2 for weight in weighted.values())) or 1
        vectors.append({word: weight/norm for word, weight in weighted.items()})
    goal = vectors[-1]
    scores = [sum(weight * goal.get(word, 0) for word, weight in vec.items()) for vec in vectors[:-1]]
    lo, hi = min(scores), max(scores)
    return [(s-lo)/(hi-lo) for s in scores] if hi-lo > 1e-9 else [0.5]*len(texts)


def evaluate(backend, question, expected, method, config=None, problem_id=None):
    if method not in METHODS:
        raise ValueError('unknown method')
    result = Solver(backend, config).solve(question, method)
    ans = result['ans']
    correct = ans is not None and math.isclose(ans, expected, rel_tol=0, abs_tol=1e-4)
    result.update(question=question, expected=expected, correct=correct, problem_id=problem_id,
                  outcome='missing' if ans is None else 'correct' if correct else 'wrong',
                  final_answer=ans, expected_answer=expected)
    return result
