"""Sources embedded as readable cells in Word_bank_approch.ipynb.

The notebook builder removes repository imports. No inference runs on import.
"""
import ast
import hashlib
import keyword
import json
import math
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from benchmark_engine import (
    Config, METHODS, Solver, STEP_SCHEMA, equation_key, equation_status,
    numeric, safe_arithmetic,
)
from benchmark_data import fixed_sample, fingerprint, read_snapshot


NEW_METHODS = (
    'Local Search: No Lexical Features',
    'Local Search: Question TF-IDF',
    'Local Search: TF-IDF + Word Bank',
    'Local Search: Word Bank + Compact State',
)
TREE_METHODS = (METHODS[0], METHODS[2], METHODS[3], METHODS[4], METHODS[5])
ALL_METHODS = TREE_METHODS + NEW_METHODS
EXPERIMENT_VERSION = 'word-bank-search-3'


def lexical_text(text):
    # Remove numeric answers before fitting/ranking. Keep words and phrases.
    text = re.sub(r'<<.*?>>', ' ', text)
    return re.sub(r'\d+(?:[.,]\d+)*', ' ', text.lower())


class ReferenceWordBank:
    """Question-conditioned phrase bank from gold TRAINING solutions only.

    This is positive-reference similarity, not a supervised correctness model.
    TF-IDF is fitted once on the training bank and frozen for all test queries.
    """
    def __init__(self, records):
        if len(records) < 2:
            raise ValueError('Need at least two bank examples')
        self.records = records
        self.questions = [row['question'] for row in records]
        self.solutions = [row['solution'] for row in records]
        self.steps, self.owners = [], []
        for index, solution in enumerate(self.solutions):
            parts = [line.strip() for line in solution.splitlines() if line.strip()]
            for part in parts:
                self.steps.append(part)
                self.owners.append(index)
        self.vectorizer = TfidfVectorizer(
            preprocessor=lexical_text, ngram_range=(1, 2),
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',
            max_features=12000, sublinear_tf=True,
        )
        self.vectorizer.fit(self.questions + self.steps)
        self.question_vectors = self.vectorizer.transform(self.questions)
        self.step_vectors = self.vectorizer.transform(self.steps)
        self.owners = np.asarray(self.owners)
        self.cache = {}

    def nearest(self, question, count=3):
        vector = self.vectorizer.transform([question])
        scores = (self.question_vectors @ vector.T).toarray().ravel()
        order = np.argsort(-scores, kind='stable')[:count]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def score(self, question, step, include_bank=True):
        if include_bank and question not in self.cache:
            nearest = self.nearest(question)
            owners = [index for index, _ in nearest]
            selected = np.flatnonzero(np.isin(self.owners, owners))
            self.cache[question] = selected
        selected = self.cache.get(question, []) if include_bank else []
        vector = self.vectorizer.transform([step])
        qvector = self.vectorizer.transform([question])
        relevance = float((vector @ qvector.T).toarray()[0, 0])
        preference = 0.0
        if len(selected):
            matches = (self.step_vectors[selected] @ vector.T).toarray().ravel()
            preference = float(matches.max())
        return relevance, preference

    def example(self, question, max_chars=1000):
        # Select a complete example; never truncate a reference equation midway.
        candidates = self.nearest(question, count=min(12, len(self.records)))
        eligible = []
        for index, similarity in candidates:
            row = self.records[index]
            text = 'Example question: ' + row['question'] + '\nSolution: ' + row['solution']
            if len(text) <= max_chars:
                eligible.append((similarity / (1 + len(text) / 500), index, text))
        if not eligible:
            return '', None
        _, index, text = max(eligible)
        return text, self.records[index]['source_id']

    def export(self, output):
        output = Path(output)
        with (output / 'bank_training_examples.jsonl').open('w', encoding='utf-8') as handle:
            for row in self.records:
                handle.write(json.dumps(row) + '\n')
        names = self.vectorizer.get_feature_names_out()
        weights = np.asarray(self.step_vectors.mean(axis=0)).ravel()
        table = pd.DataFrame({'phrase': names, 'mean_reference_tfidf': weights,
                              'idf': self.vectorizer.idf_})
        table.sort_values('mean_reference_tfidf', ascending=False).to_csv(
            output / 'word_bank_phrases.csv', index=False)
        # Save the exact fitted vocabulary/IDF, not an unsafe pickle.
        fitted = {'vocabulary': {k: int(v) for k, v in self.vectorizer.vocabulary_.items()},
                  'idf': self.vectorizer.idf_.tolist(),
                  'policy': 'Positive training-reference similarity; no negative preference labels.'}
        (output / 'word_bank_model.json').write_text(json.dumps(fitted, indent=2), encoding='utf-8')


def prepare_word_bank(cache_dir, subsets, size=200):
    catalog = {key: read_snapshot(key, cache_dir)[0]
               for key in ('gsm8k_train', 'gsm8k_test', 'svamp')}
    excluded = {row['question_sha256'] for name in ('development', 'evaluation')
                for row in subsets[name]}
    excluded.update(row['question_sha256'] for row in catalog['gsm8k_test'] + catalog['svamp'])
    selected = fixed_sample(catalog['gsm8k_train'], size, 809, excluded)
    raw = [json.loads(line) for line in
           (Path(cache_dir) / 'gsm8k_train.jsonl').read_text(encoding='utf-8').splitlines()]
    records = []
    for row in selected:
        index = int(row['source_id'].rsplit(':', 1)[1])
        solution = raw[index]['answer'].split('####', 1)[0].strip()
        records.append({'source_id': row['source_id'], 'question': row['question'],
                        'question_sha256': row['question_sha256'], 'solution': solution})
    assert not {row['question_sha256'] for row in records} & excluded
    return ReferenceWordBank(records)


class BudgetStop(BaseException):
    """Escape engine retries immediately when the shared budget is exhausted."""


class MeteredBackend:
    """Actual usage accounting plus a conservative preflight budget estimate.

    Ollama reports actual tokens after a call. The character estimate is not a
    tokenizer or a strict billable cap. Export any actual overshoot explicitly.
    """
    def __init__(self, backend, total_budget):
        self.backend, self.budget = backend, total_budget
        self.requests = []
        self.input_tokens = self.output_tokens = 0
        self.fallback = None

    @property
    def used(self):
        return self.input_tokens + self.output_tokens

    def __call__(self, prompt, schema, temperature, seed, max_tokens):
        estimated_input = math.ceil(len(prompt.encode('utf-8')) / 3) + 128
        remaining = self.budget - self.used - estimated_input
        if remaining < 64:
            raise BudgetStop('token_budget_preflight')
        limit = min(max_tokens, remaining)
        event = {'prompt': prompt, 'schema': schema, 'temperature': temperature,
                 'seed': seed, 'requested_output_limit': max_tokens,
                 'effective_output_limit': limit, 'estimated_input_tokens': estimated_input,
                 'usage_valid': False}
        self.requests.append(event)
        try:
            response = self.backend(prompt, schema, temperature, seed, limit)
        except Exception as exc:
            event['error'] = str(exc)
            raise
        event.update(response)
        if any(type(response.get(k)) is not int or response[k] < 0
               for k in ('prompt_tokens', 'completion_tokens')):
            event['error'] = 'missing_or_invalid_token_usage'
            raise RuntimeError('Backend must return actual nonnegative token usage')
        event['usage_valid'] = True
        self.input_tokens += response['prompt_tokens']
        self.output_tokens += response['completion_tokens']
        return response


def concise_answer(solver, question, example='', temperature=0.1):
    prompt = (example + '\nProblem: ' + question +
              '\nSolve using concise calculations. Preserve necessary reasoning. '
              'Finish on a separate line: FINAL ANSWER: <decimal number>. '
              'Do not put units or other text on that final line.')
    prior = solver.cfg
    solver.cfg = replace(prior, retries=0)
    try:
        return solver.request(prompt, None, temperature, 'concise_solution', 512)
    finally:
        solver.cfg = prior


def solution_checks(text, answer):
    # These checks are explicitly weak: no unsupported claim of semantic verification.
    equations = re.findall(r'(?m)^\s*([\d.,+*/()%\s-]+=[\d.,+*/()%\s-]+)\s*$', text)
    checks = [equation_status(equation.strip()) for equation in equations]
    grounded = any(state == 'valid' and math.isclose(value, answer, abs_tol=1e-4, rel_tol=0)
                   for state, value in checks) if answer is not None else False
    return bool(checks) and all(state == 'valid' for state, _ in checks) and grounded


EQUATIONS_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'calculations': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {'name': {'type': 'string'}, 'expression': {'type': 'string'}},
            'required': ['name', 'expression']}},
        'result': {'type': 'string'}, 'target': {'type': 'string'},
    }, 'required': ['calculations', 'result', 'target'],
}


def execute_equations(obj):
    calculations = obj.get('calculations')
    if not isinstance(calculations, list) or not 1 <= len(calculations) <= 24:
        raise ValueError('Need 1 to 24 calculations')
    if not isinstance(obj.get('target'), str) or not obj['target'].strip():
        raise ValueError('Need a target description')
    values = {}
    def expression_value(expr):
        if not isinstance(expr, str) or len(expr) > 256:
            raise ValueError('Expression must be a string of at most 256 characters')
        try:
            tree = ast.parse(expr.strip(), mode='eval')
        except SyntaxError as exc:
            raise ValueError('Expression is not valid arithmetic: ' + expr) from exc
        if sum(1 for _ in ast.walk(tree)) > 80:
            raise ValueError('Expression is too large')
        class Substitute(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id not in values:
                    raise ValueError('Undefined variable: ' + node.id)
                return ast.copy_location(ast.Constant(value=values[node.id]), node)
        expanded = ast.unparse(Substitute().visit(tree))
        value = safe_arithmetic(expanded)
        if value is None:
            raise ValueError('Unsupported or invalid arithmetic: ' + expr)
        return value
    for calculation in calculations:
        if not isinstance(calculation, dict):
            raise ValueError('Each calculation must have name and expression fields')
        name, expr = calculation.get('name'), calculation.get('expression')
        if (not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z][a-zA-Z_0-9]{0,63}', name)
                or keyword.iskeyword(name)):
            raise ValueError('Use variable names starting with a letter, with letters, digits or underscores')
        # Sequential reassignment is valid; the expression reads the preceding value.
        values[name] = expression_value(expr)
    result = obj.get('result')
    if not isinstance(result, str) or not result.strip():
        raise ValueError('result must name a computed variable or an expression using computed variables')
    result_value = expression_value(result)
    uses_variables = any(isinstance(node, ast.Name) for node in ast.walk(ast.parse(result.strip(), mode='eval')))
    if not uses_variables and not any(math.isclose(result_value, v, rel_tol=0, abs_tol=1e-6)
                                      for v in values.values()):
        raise ValueError('A literal result must match a computed variable; prefer its variable name')
    return result_value, values


def program_answer(solver, question):
    prompt = ('Problem: ' + question + '\nRepresent the solution as sequential named arithmetic '
              'expressions in the schema. Use given numbers and previously defined variables. '
              'Allowed operators: + - * / // % ** and parentheses. No imports or function calls. '
              'Variable names start with a letter and may contain letters, digits or underscores. '
              'result names the final computed variable. target describes the requested quantity. '
              'Example FORMAT for a DIFFERENT question: '
              '{"calculations":[{"name":"subtotal","expression":"2*3"},'
              '{"name":"remaining","expression":"subtotal-1"}],'
              '"result":"remaining","target":"items remaining"}. '
              'Use the actual problem numbers. Compute, do not guess.')
    for repair in range(solver.cfg.retries + 1):
        obj = solver.request(prompt, EQUATIONS_SCHEMA, 0.1, 'executable_equations', 768)
        if obj is None:
            return None
        try:
            answer, values = execute_equations(obj)
            solver.events.append({'kind': 'program_execution', 'values': values, 'model': obj})
            return answer
        except (ValueError, TypeError, AttributeError) as exc:
            solver.failure = 'program_execution_error: ' + str(exc)
            solver.events.append({'kind': 'program_rejected', 'error': str(exc)})
            prompt += '\nPrevious: ' + json.dumps(obj) + '\nRepair this error: ' + str(exc)
    return None


def search_prompt(question, steps, compact):
    if compact:
        # Deterministic compression: keep all equations and their labels; no summarizer call.
        history = '\n'.join(step['target'] + ': ' + step['equation'] for step in steps)
    else:
        history = json.dumps(steps)
    return ('Problem: ' + question + '\nPrior calculations: ' + history +
            '\nProduce ONE new arithmetic step using the schema. equation must contain only '
            'numbers, arithmetic operators, parentheses and one =. No variables or units in '
            'equation. target names the quantity calculated and its units for THIS step. '
            'Do not repeat calculations. Set done=false and answer=null until the whole '
            'question is solved. In the response computing the final quantity, set done=true '
            'and answer to that numeric result; do not leave the solved branch unfinished. '
            'Include expression (arithmetic only) and result (numeric value) as separate fields; '
            'they must agree when calculated locally. Give a nonempty step explanation. '
            'An empty final equation is allowed only if an earlier equation computed the answer.')


def local_search(solver, question, bank, lexical=False, word_bank=False, compact=False):
    import heapq
    import itertools
    counter = itertools.count()
    queue = [(-1.0, next(counter), [])]
    best, expansions = [], 0
    seen_paths = set()
    rejected_terminals = {}
    stalled_branches = {}
    terminal_cache = {}
    depth_cutoff = False
    while queue and expansions < solver.cfg.max_expansions and solver.calls < solver.cfg.max_calls:
        _, _, steps = heapq.heappop(queue)
        if len(steps) >= solver.cfg.max_depth:
            depth_cutoff = True
            continue
        expansions += 1
        solver.events.append({'kind': 'local_expand', 'steps': steps, 'expansion': expansions})
        prompt = search_prompt(question, steps, compact)
        branch_key = json.dumps(steps, sort_keys=True)
        if branch_key in stalled_branches:
            prompt += ('\nThe previous expansion stalled: ' + stalled_branches[branch_key]['reason']
                       + '. Produce a new valid equation. If prior calculations already solve the '
                       'question, declare done=true with that answer and its requested target.')
        if branch_key in rejected_terminals:
            prompt += ('\nEarlier final answers on this branch failed validation: '
                       + json.dumps(rejected_terminals[branch_key])
                       + '. Recheck the requested quantity and give a corrected next step.')
        candidates = []
        for _ in range(solver.cfg.candidates):
            candidate = solver.request(prompt, STEP_SCHEMA, 0.7, 'local_generate', 384)
            if candidate is not None and candidate not in candidates:
                candidates.append(candidate)
        scored = []
        repeated_count = 0
        for candidate in candidates:
            key = equation_key(candidate['equation'])
            repeated = any(key == equation_key(step['equation']) for step in steps)
            if repeated and not candidate['done']:
                repeated_count += 1
                solver.events.append({'kind': 'local_repeated_step', 'candidate': candidate})
                continue
            relevance, preference = (bank.score(
                question, candidate['step'] + ' ' + candidate['target'], include_bank=word_bank)
                if lexical or word_bank else (0.0, 0.0))
            # Novel equation is a weak progress PROXY, not proof a subgoal was solved.
            valid, _ = equation_status(candidate['equation'])
            progress = float(valid == 'valid' and not repeated)
            history_text = ' '.join(step['step'].lower() for step in steps)
            words = set(re.findall(r'\b[a-z]+\b', candidate['step'].lower()))
            repetition = len(words & set(history_text.split())) / max(1, len(words))
            path = steps + [candidate]
            next_input = len(search_prompt(question, path, compact).encode('utf-8')) / 3 + 128
            generated = [e.get('completion_tokens', 0) for e in solver.events
                         if e.get('purpose') == 'local_generate' and 'completion_tokens' in e]
            expected_output = float(np.mean(generated)) if generated else 192.0
            estimated_cost = max(1.0, next_input + expected_output)
            utility = (0.2 + 0.5 * progress - 0.1 * repetition
                       + (0.15 * relevance if lexical else 0)
                       + (0.15 * preference if word_bank else 0))
            score = utility / estimated_cost
            solver.events.append({'kind': 'local_score', 'candidate': candidate,
                                  'question_tfidf': relevance, 'word_bank': preference,
                                  'progress_proxy': progress, 'repetition': repetition,
                                  'estimated_next_tokens': estimated_cost, 'score': score})
            scored.append((score, path))
        if not scored:
            reason = ('all candidates repeated earlier equations' if repeated_count else
                      solver.failure or 'no valid candidates generated')
            solver.events.append({'kind': 'branch_stalled', 'reason': reason, 'steps': steps})
            attempts = stalled_branches.get(branch_key, {}).get('attempts', 0)
            if attempts < solver.cfg.retries:
                stalled_branches[branch_key] = {'attempts': attempts + 1, 'reason': reason}
                heapq.heappush(queue, (-1.0, next(counter), steps))
        rejection = None
        for score, path in sorted(scored, key=lambda item: item[0], reverse=True):
            candidate = path[-1]
            if candidate['done']:
                values = [equation_status(step['equation'])[1] for step in path]
                grounded = any(v is not None and math.isclose(
                    v, candidate['answer'], abs_tol=1e-4, rel_tol=0) for v in values)
                # All four local-search ablations use this same counted review.
                terminal_key = (tuple(equation_key(step['equation']) for step in path),
                                candidate['answer'], candidate['target'].strip().lower())
                if terminal_key not in terminal_cache:
                    terminal_cache[terminal_key] = (grounded and
                                                    solver.terminal_review(question, path[:-1], candidate))
                else:
                    solver.events.append({'kind': 'terminal_cache_hit', 'candidate': candidate})
                if terminal_cache[terminal_key]:
                    return candidate['answer'], path, 'done'
                rejection = {'candidate': candidate, 'grounded': grounded}
                solver.events.append({'kind': 'local_terminal_rejected', **rejection})
            else:
                signature = tuple(equation_key(step['equation']) for step in path)
                if signature not in seen_paths:
                    seen_paths.add(signature)
                    heapq.heappush(queue, (-score, next(counter), path))
                    if len(path) > len(best):
                        best = path
        # At most cfg.retries recovery expansions per branch. No reference-answer feedback.
        if rejection is not None:
            previous = rejected_terminals.setdefault(branch_key, [])
            if len(previous) < solver.cfg.retries:
                previous.append(rejection)
                heapq.heappush(queue, (-max(score for score, _ in scored), next(counter), steps))
        queue = heapq.nsmallest(solver.cfg.beam, queue)
        heapq.heapify(queue)
    status = ('call_limit' if solver.calls >= solver.cfg.max_calls else
              'expansion_limit' if expansions >= solver.cfg.max_expansions else
              'depth_limit' if depth_cutoff else 'exhausted')
    return None, best, status


def consistency_answer(solver, question, early, meter):
    votes = Counter()
    for index in range(5):
        answer = concise_answer(solver, question, temperature=0.7)
        if answer is not None:
            votes[round(answer, 4)] += 1
            winner, count = votes.most_common(1)[0]
            # Insertion order gives a deterministic first-seen tie breaker.
            meter.fallback = winner
            solver.events.append({'kind': 'vote', 'sample': index + 1,
                                  'votes': dict(votes), 'winner': winner})
            if early and index + 1 >= 3 and count >= 3 and count / sum(votes.values()) >= 0.75:
                return winner
    return votes.most_common(1)[0][0] if votes else None


def failure_diagnostics(events, status, last_error):
    counts = Counter()
    last_reason = last_error
    for event in events:
        reason = event.get('error')
        if event.get('kind') == 'branch_stalled':
            reason = event['reason']
        elif event.get('kind') == 'local_repeated_step':
            reason = 'repeated_equation'
        elif event.get('kind') in ('local_terminal_rejected', 'rejected'):
            reason = event.get('reason', 'terminal_review_or_grounding_rejected')
        elif event.get('kind') == 'terminal_decision' and not event.get('accepted'):
            review = event.get('review')
            reason = 'terminal_review: ' + str(review.get('reason') if review else 'no valid review')
        if reason:
            counts[str(reason)] += 1
            last_reason = str(reason)
    return {'failure_reason': last_reason or status, 'diagnostic_counts': dict(counts)}


def run_method(backend, question, method, bank, seed, token_budget=16000, config=None):
    if method not in ALL_METHODS:
        raise ValueError('Unknown method: ' + method)
    if type(token_budget) is not int or token_budget < 1:
        raise ValueError('Token budget must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Seed must be an integer from 0 to 2**31 - 1')
    started = time.perf_counter()
    cfg = replace(config or Config(max_calls=30, max_expansions=16, retries=1,
                                   audit_tos=True, review_appeal=True), seed=seed)
    meter = MeteredBackend(backend, token_budget)
    solver = Solver(meter, cfg)
    answer, steps, status = None, [], 'missing'
    extra = {}
    # Reset query-derived cache for comparable per-solver timing; fitted training bank stays frozen.
    bank.cache.clear()
    try:
        if method in METHODS:
            result = solver.solve(question, method)
            answer, steps, status = result['ans'], result['steps'], result['status']
            extra = {k: result[k] for k in ('baseline_version', 'search_model') if k in result}
        else:
            answer, steps, status = local_search(
                solver, question, bank, lexical=method != NEW_METHODS[0],
                word_bank=method in NEW_METHODS[2:4], compact=method == NEW_METHODS[3])
        if answer is not None and status in ('missing', 'exhausted'):
            status = 'done'
        elif answer is None and status == 'missing':
            status = solver.failure or 'missing_final_answer'
    except BudgetStop:
        answer = meter.fallback
        status = 'budget_fallback' if answer is not None else 'token_budget'
    except Exception as exc:
        status = 'runtime_error'
        solver.events.append({'kind': 'runtime_error', 'error': repr(exc)})
    diagnostics = failure_diagnostics(solver.events, status, solver.failure) if answer is None else {}
    return dict(method=method, ans=answer, status=status, steps=steps,
                calls=len(meter.requests), tokens=meter.used, input_tokens=meter.input_tokens,
                output_tokens=meter.output_tokens, sec=time.perf_counter() - started,
                trace=solver.events, requests=meter.requests, seed=seed, config=vars(cfg),
                token_budget=token_budget, budget_overshoot=max(0, meter.used - token_budget),
                usage_complete=all(r.get('usage_valid', False) for r in meter.requests),
                engine_version=EXPERIMENT_VERSION, last_error=solver.failure, **extra, **diagnostics)


def validate_prices(input_price, output_price):
    if (input_price is None) != (output_price is None):
        raise ValueError('Supply both input and output prices or neither')
    if input_price is not None and any(not numeric(v) or v < 0 for v in (input_price, output_price)):
        raise ValueError('Prices must be finite nonnegative numbers')


def experiment_report(rows, output, input_price=None, output_price=None):
    import matplotlib.pyplot as plt
    validate_prices(input_price, output_price)
    if not rows:
        raise ValueError('No completed solver records to report')
    output = Path(output)
    df = pd.DataFrame([{k: v for k, v in row.items() if k not in
                        ('trace', 'requests', 'steps', 'search_model', 'config')} for row in rows])
    summaries = []
    for (dataset, method), group in df.groupby(['dataset', 'method'], sort=False):
        correct = int(group['correct'].sum())
        complete = bool(group['usage_complete'].all())
        summaries.append(dict(
            dataset=dataset, method=method, runs=len(group), correct=correct,
            missing=int(group['ans'].isna().sum()),
            wrong=int(((~group['correct']) & group['ans'].notna()).sum()),
            accuracy_pct=100 * correct / len(group), mean_calls=group['calls'].mean(),
            mean_input_tokens=group['input_tokens'].mean() if complete else np.nan,
            mean_output_tokens=group['output_tokens'].mean() if complete else np.nan,
            mean_tokens=group['tokens'].mean() if complete else np.nan,
            mean_observed_tokens=group['tokens'].mean(),
            tokens_per_correct=group['tokens'].sum() / correct if correct and complete else np.nan,
            mean_seconds=group['sec'].mean(), budget_overshoot_runs=int((group['budget_overshoot'] > 0).sum()),
            usage_incomplete_runs=int((~group['usage_complete']).sum()),
        ))
    table = pd.DataFrame(summaries)
    for dataset in table['dataset'].unique():
        mask = table['dataset'] == dataset
        reference = table[mask & (table['method'] == METHODS[0])]
        if not reference.empty and reference.iloc[0]['mean_tokens'] > 0:
            table.loc[mask, 'token_savings_vs_cot_pct'] = (
                100 * (1 - table.loc[mask, 'mean_tokens'] / reference.iloc[0]['mean_tokens']))
    if input_price is not None:
        table['hypothetical_usd_per_problem'] = (
            table['mean_input_tokens'] * input_price + table['mean_output_tokens'] * output_price) / 1e6
    table.to_csv(output / 'comparison_summary.csv', index=False)
    df.to_csv(output / 'per_problem_metrics.csv', index=False)
    columns = ['dataset', 'source_id', 'method', 'seed', 'status', 'failure_reason',
               'diagnostic_counts', 'calls', 'input_tokens', 'output_tokens']
    df.reindex(columns=columns).loc[df['ans'].isna()].to_csv(output / 'missing_diagnostics.csv', index=False)
    df['complete_tokens'] = df['tokens'].where(df['usage_complete'])
    df.groupby(['dataset', 'method', 'seed']).agg(
        accuracy=('correct', 'mean'), tokens=('complete_tokens', lambda s: s.mean(skipna=False)),
        calls=('calls', 'mean')
    ).to_csv(output / 'per_seed_metrics.csv')
    df.groupby(['dataset', 'method', 'status']).size().to_csv(output / 'termination_counts.csv')
    for dataset in table['dataset'].unique():
        sub = table[table['dataset'] == dataset]
        fig, ax = plt.subplots(figsize=(12, 7))
        for _, row in sub.iterrows():
            if not math.isfinite(row['mean_tokens']):
                continue
            ax.scatter(row['mean_tokens'], row['accuracy_pct'])
            ax.annotate(row['method'], (row['mean_tokens'], row['accuracy_pct']),
                        fontsize=7, xytext=(3, 4), textcoords='offset points')
        ax.set(xlabel='Mean total input + output tokens per problem',
               ylabel='Accuracy (%) — missing answers count as failures', title=dataset, ylim=(-2, 105))
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / (dataset + '_accuracy_vs_tokens.png'), dpi=160)
        plt.close(fig)
    return table


def run_word_bank_suite(backend, cases, bank, output, seeds, methods, token_budget,
                       provenance, profile, config=None, input_price=None, output_price=None):
    validate_prices(input_price, output_price)
    if not cases or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Need cases and distinct seeds')
    if any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds):
        raise ValueError('Seeds must be integers from 0 to 2**31 - 1')
    if not methods or len(set(methods)) != len(methods) or any(m not in ALL_METHODS for m in methods):
        raise ValueError('Need distinct known methods')
    if type(token_budget) is not int or token_budget < 256:
        raise ValueError('Token budget must be an integer of at least 256')
    for case in cases:
        if (not isinstance(case.get('question'), str) or not case['question'].strip()
                or not numeric(case.get('expected')) or not isinstance(case.get('source_id'), str)
                or not isinstance(case.get('dataset'), str)
                or not re.fullmatch(r'[a-zA-Z0-9_-]+', case['dataset'])):
            raise ValueError('Each case needs a question, numeric reference, source ID and dataset name')
    splits = {case.get('evaluation_split', profile) for case in cases}
    if len(splits) != 1 or ('evaluation' in splits) != (profile == 'evaluation'):
        raise ValueError('Do not mix development and evaluation profiles')
    if len({case['source_id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate evaluation source IDs')
    bank_hashes = {row['question_sha256'] for row in bank.records}
    if any(fingerprint(case['question']) in bank_hashes for case in cases):
        raise ValueError('Training-bank leakage into benchmark')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bank.export(output)
    cfg = config or Config(max_calls=30, max_expansions=16, retries=1)
    manifest = dict(version=EXPERIMENT_VERSION, model='qwen2.5:7b', profile=profile,
                    methods=list(methods), seeds=list(seeds), token_budget=token_budget,
                    config=vars(cfg), provenance=provenance,
                    bank_source_ids=[row['source_id'] for row in bank.records],
                    bank_selection_seed=809, bank_policy='Positive gold TRAINING reference steps only',
                    baseline_policy='Existing methods with shared experiment limits; not old run settings',
                    prices_per_million=dict(input=input_price, output=output_price),
                    budget_policy='Estimated preflight; actual usage and overshoot recorded',
                    planned_runs=len(cases) * len(seeds) * len(methods))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with (output / 'selected_problems.jsonl').open('w', encoding='utf-8') as handle:
        for case in cases:
            handle.write(json.dumps(case) + '\n')
    rows = []
    with (output / 'full_results.jsonl').open('w', encoding='utf-8') as handle:
        for seed in seeds:
            for case in cases:
                # Rotate method order deterministically to reduce fixed-order timing bias.
                offset = int(hashlib.sha256(f'{seed}:{case["source_id"]}'.encode()).hexdigest(), 16)
                offset %= len(methods)
                order = list(methods[offset:]) + list(methods[:offset])
                for method in order:
                    result = run_method(backend, case['question'], method, bank, seed, token_budget, cfg)
                    # Reference answers are accessed ONLY after the solver returns.
                    result.update({key: case[key] for key in
                                   ('dataset', 'source_id', 'question', 'expected')})
                    answer = result['ans']
                    result['correct'] = answer is not None and math.isclose(
                        answer, case['expected'], abs_tol=1e-4, rel_tol=0)
                    result['outcome'] = ('missing' if answer is None else
                                         'correct' if result['correct'] else 'wrong')
                    result['experiment_id'] = output.name
                    result['profile'] = profile
                    handle.write(json.dumps(result) + '\n')
                    handle.flush()
                    rows.append(result)
                    print(seed, case['source_id'], method, result['outcome'],
                          result['tokens'], 'tokens', result['status'],
                          ('| ' + result.get('failure_reason', '')[:240]) if answer is None else '', flush=True)
    table = experiment_report(rows, output, input_price, output_price)
    return rows, table
