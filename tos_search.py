"""Thought of Search arithmetic adaptation: synthesize components, then local BFS.

Reference: Katz et al., https://arxiv.org/abs/2404.11833.
The original human-reviewed, reusable domain code is replaced here by bounded
per-problem synthesis of restricted Python functions. No paper-level soundness
or completeness guarantee is claimed for the inferred word-problem model.
"""
import ast
import copy
from collections import deque
import json
import math
import operator
import time

from benchmark_engine import REVIEW_SCHEMA, numeric

TOS_METHOD = 'Thought of Search (Arithmetic Adaptation)'
TOS_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'initial_state': {'type': 'object', 'additionalProperties': {'type': 'number'}},
        'successor_code': {'type': 'string'}, 'goal_code': {'type': 'string'},
        'answer_code': {'type': 'string'}, 'target': {'type': 'string'},
        'objective': {'type': 'string', 'enum': ['first', 'min', 'max']},
    },
    'required': ['initial_state', 'successor_code', 'goal_code', 'answer_code', 'target', 'objective'],
}


class RestrictedFunction:
    """Interpret bounded arithmetic Python; never exec/eval generated code."""
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Store,
               ast.Subscript, ast.Dict, ast.List, ast.Tuple, ast.BinOp, ast.UnaryOp,
               ast.BoolOp, ast.Compare, ast.IfExp, ast.ListComp, ast.comprehension,
               ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
               ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq,
               ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BitOr,
               ast.Assign, ast.AugAssign, ast.If, ast.For, ast.Return, ast.Expr,
               ast.Break, ast.Continue, ast.Pass, ast.Attribute, ast.keyword)
    functions = {'abs', 'min', 'max', 'sum', 'len', 'round', 'ceil', 'floor', 'range',
                 'dict', 'int', 'float'}
    methods = {'copy', 'get', 'update', 'append'}

    def __init__(self, code, name):
        if not isinstance(code, str) or len(code) > 12000:
            raise ValueError('component code must be a string of at most 12000 characters')
        tree = ast.parse(code)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError('component must define exactly one function')
        function = tree.body[0]
        args = function.args
        if (function.name != name or function.decorator_list or function.returns
                or len(args.args) != 1 or args.args[0].arg != 'state' or args.args[0].annotation
                or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults
                or args.kw_defaults):
            raise ValueError('use def ' + name + '(state): without decorators, defaults, or annotations')
        self.body = function.body
        nodes = [node for statement in self.body for node in ast.walk(statement)]
        if not any(isinstance(node, ast.Return) for node in nodes):
            raise ValueError('function requires a return statement')
        if len(nodes) > 1000:
            raise ValueError('component AST is too large')
        for node in nodes:
            if not isinstance(node, self.allowed):
                raise ValueError('unsupported syntax: ' + type(node).__name__)
            if isinstance(node, ast.Name) and node.id.startswith('_'):
                raise ValueError('private names are unsupported')
            if isinstance(node, ast.Attribute) and node.attr not in self.methods:
                raise ValueError('unsupported method or attribute: ' + node.attr)
            if isinstance(node, ast.Call):
                helper = isinstance(node.func, ast.Name) and node.func.id in self.functions
                method = isinstance(node.func, ast.Attribute) and node.func.attr in self.methods
                if not (helper or method) or any(isinstance(arg, ast.Starred) for arg in node.args):
                    raise ValueError('only arithmetic helpers and bounded collection methods are supported')
                if any(keyword.arg is None for keyword in node.keywords):
                    raise ValueError('keyword unpacking is unsupported')
                if node.keywords and not (helper and node.func.id == 'dict'):
                    raise ValueError('only dict accepts keyword arguments')
            if isinstance(node, ast.comprehension) and (
                    node.is_async or not isinstance(node.target, ast.Name)
                    or node.target.id == 'state' or node.target.id in self.functions):
                raise ValueError('comprehensions need a simple local variable')
        if not any(isinstance(node, ast.Name) and node.id == 'state' for node in nodes):
            raise ValueError('component must depend on state, not a constant answer')

    def __call__(self, state):
        self.remaining = 20000
        # Generated assignments cannot mutate states retained in the BFS queue or trace.
        outcome = self.execute(self.body, {'state': copy.deepcopy(state)})
        if outcome is None or outcome[0] != 'return':
            raise ValueError('function reached its end without returning')
        return outcome[1]

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError('expression operation limit')

    def assign(self, target, value, env):
        if isinstance(target, ast.Name) and target.id not in self.functions and not target.id.startswith('_'):
            env[target.id] = value
        elif isinstance(target, ast.Subscript):
            collection = self.visit(target.value, env)
            key = self.visit(target.slice, env)
            if not isinstance(collection, (dict, list)):
                raise ValueError('assignment needs a local collection')
            collection[key] = value
            if len(collection) > 256:
                raise ValueError('collection limit')
        else:
            raise ValueError('unsupported assignment target')

    def execute(self, statements, env):
        for statement in statements:
            self.tick()
            if isinstance(statement, ast.Return):
                return 'return', self.visit(statement.value, env) if statement.value else None
            if isinstance(statement, ast.Assign):
                value = self.visit(statement.value, env)
                for target in statement.targets:
                    self.assign(target, value, env)
            elif isinstance(statement, ast.AugAssign):
                expression = ast.BinOp(left=statement.target, op=statement.op, right=statement.value)
                self.assign(statement.target, self.visit(expression, env), env)
            elif isinstance(statement, ast.If):
                branch = statement.body if self.visit(statement.test, env) else statement.orelse
                outcome = self.execute(branch, env)
                if outcome:
                    return outcome
            elif isinstance(statement, ast.For):
                values = self.visit(statement.iter, env)
                if not isinstance(values, list) or len(values) > 256:
                    raise ValueError('for requires a bounded list or range')
                broke = False
                for value in list(values):
                    self.tick()
                    self.assign(statement.target, value, env)
                    outcome = self.execute(statement.body, env)
                    if outcome and outcome[0] == 'return':
                        return outcome
                    if outcome and outcome[0] == 'break':
                        broke = True
                        break
                if not broke:
                    outcome = self.execute(statement.orelse, env)
                    if outcome:
                        return outcome
            elif isinstance(statement, ast.Break):
                return 'break', None
            elif isinstance(statement, ast.Continue):
                return 'continue', None
            elif isinstance(statement, ast.Expr):
                self.visit(statement.value, env)
            elif not isinstance(statement, ast.Pass):
                raise ValueError('unsupported statement')

    def visit(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant):
            value = node.value
            if not isinstance(value, (str, bool)) and not numeric(value):
                raise ValueError('unsupported constant')
            if isinstance(value, str) and len(value) > 64:
                raise ValueError('string limit')
        elif isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError('undefined local variable: ' + node.id)
            value = env[node.id]
        elif isinstance(node, ast.Subscript):
            collection, key = self.visit(node.value, env), self.visit(node.slice, env)
            try:
                value = collection[key]
            except KeyError as exc:
                raise ValueError(f'Undefined state key {key!r}; dictionary literals read the OLD state. '
                                 'Compute dependent fields with sequential assignments or separate stages.') from exc
        elif isinstance(node, ast.Dict):
            if len(node.keys) > 32:
                raise ValueError('state size limit')
            value = {}
            for key, child in zip(node.keys, node.values):
                if key is None:
                    mapping = self.visit(child, env)
                    if not isinstance(mapping, dict):
                        raise ValueError('dictionary unpacking needs a dictionary')
                    value.update(mapping)
                else:
                    value[self.visit(key, env)] = self.visit(child, env)
        elif isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > 256:
                raise ValueError('collection limit')
            value = [self.visit(item, env) for item in node.elts]
        elif isinstance(node, ast.BinOp):
            a, b = self.visit(node.left, env), self.visit(node.right, env)
            if isinstance(node.op, ast.BitOr):
                if not isinstance(a, dict) or not isinstance(b, dict):
                    raise ValueError('union is supported only for dictionaries')
                value = dict(a, **b)
                if len(value) > 32:
                    raise ValueError('state size limit')
                return value
            if not numeric(a) or not numeric(b):
                raise ValueError('arithmetic requires numbers')
            if isinstance(node.op, ast.Pow) and abs(b) > 12:
                raise ValueError('exponent limit')
            operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                          ast.Mod: operator.mod, ast.Pow: operator.pow}
            value = operations[type(node.op)](a, b)
        elif isinstance(node, ast.UnaryOp):
            a = self.visit(node.operand, env)
            if isinstance(node.op, ast.Not):
                value = not a
            elif numeric(a):
                value = -a if isinstance(node.op, ast.USub) else a
            else:
                raise ValueError('unary arithmetic requires a number')
        elif isinstance(node, ast.BoolOp):
            value = self.visit(node.values[0], env)
            for child in node.values[1:]:
                if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value):
                    break
                value = self.visit(child, env)
        elif isinstance(node, ast.Compare):
            a = self.visit(node.left, env)
            operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                          ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
            value = True
            for op, child in zip(node.ops, node.comparators):
                b = self.visit(child, env)
                if not operations[type(op)](a, b):
                    value = False
                    break
                a = b
        elif isinstance(node, ast.IfExp):
            value = self.visit(node.body if self.visit(node.test, env) else node.orelse, env)
        elif isinstance(node, ast.ListComp):
            value = []
            def collect(index, scope):
                if index == len(node.generators):
                    value.append(self.visit(node.elt, scope))
                    if len(value) > 256:
                        raise ValueError('collection limit')
                    return
                gen = node.generators[index]
                iterable = self.visit(gen.iter, scope)
                if not isinstance(iterable, list) or len(iterable) > 256:
                    raise ValueError('bounded list required')
                for item in iterable:
                    self.remaining -= 1
                    if self.remaining < 0:
                        raise ValueError('expression operation limit')
                    local = dict(scope, **{gen.target.id: item})
                    if all(self.visit(condition, local) for condition in gen.ifs):
                        collect(index + 1, local)
            collect(0, env)
        elif isinstance(node, ast.Call):
            args = [self.visit(arg, env) for arg in node.args]
            if isinstance(node.func, ast.Attribute):
                receiver = self.visit(node.func.value, env)
                method = node.func.attr
                if isinstance(receiver, dict) and method == 'copy' and not args:
                    return dict(receiver)
                if isinstance(receiver, dict) and method == 'get' and 1 <= len(args) <= 2:
                    return receiver.get(*args)
                if isinstance(receiver, dict) and method == 'update' and len(args) == 1 and isinstance(args[0], dict):
                    receiver.update(args[0])
                elif isinstance(receiver, list) and method == 'append' and len(args) == 1:
                    receiver.append(args[0])
                else:
                    raise ValueError('unsupported collection method arguments')
                if len(receiver) > 256:
                    raise ValueError('collection limit')
                return None
            name = node.func.id
            if name == 'range':
                if not 1 <= len(args) <= 3 or any(not numeric(arg) or int(arg) != arg for arg in args):
                    raise ValueError('range needs integer bounds')
                sequence = range(*(int(arg) for arg in args))
                if len(sequence) > 256:
                    raise ValueError('range limit')
                value = list(sequence)
            elif name == 'dict':
                if len(args) > 1 or (args and not isinstance(args[0], dict)):
                    raise ValueError('dict needs at most one dictionary')
                value = dict(args[0]) if args else {}
                value.update({keyword.arg: self.visit(keyword.value, env) for keyword in node.keywords})
            else:
                funcs = {'abs': abs, 'min': min, 'max': max, 'sum': sum, 'len': len,
                         'round': round, 'ceil': math.ceil, 'floor': math.floor, 'int': int, 'float': float}
                value = funcs[name](*args)
        else:
            raise ValueError('unsupported expression')
        if isinstance(value, (dict, list, tuple)) and len(value) > 256:
            raise ValueError('collection limit')
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not numeric(value) or abs(value) > 1e15:
                raise ValueError('numeric limit')
        return value


def validate_state(state, keys=None):
    if (not isinstance(state, dict) or not 1 <= len(state) <= 32
            or any(not isinstance(key, str) or not key or len(key) > 64 for key in state)
            or any(not numeric(value) or abs(value) > 1e15 for value in state.values())):
        raise ValueError('state must have string keys and finite numeric values')
    return state


def validate_model(model):
    if set(model) != set(TOS_SCHEMA['required']):
        raise ValueError('missing or unexpected search-model fields')
    initial = validate_state(model['initial_state'])
    if model['objective'] not in ('first', 'min', 'max') or not isinstance(model['target'], str):
        raise ValueError('invalid objective or target')
    if not model['target'].strip():
        raise ValueError('target is required')
    successor = RestrictedFunction(model['successor_code'], 'successors')
    goal = RestrictedFunction(model['goal_code'], 'is_goal')
    answer = RestrictedFunction(model['answer_code'], 'answer')
    initial_goal = goal(initial)
    if not isinstance(initial_goal, bool):
        raise ValueError('goal test must return a boolean')
    children = [] if initial_goal else successor(initial)
    if not isinstance(children, list) or len(children) > 256:
        raise ValueError('initial successors must be a bounded list')
    for child in children:
        validate_state(child, set(initial))
    return initial, successor, goal, answer


def local_bfs(model, config, events):
    initial, successors, is_goal, answer = validate_model(model)
    queue = deque([(initial, [])])
    visited = {json.dumps(initial, sort_keys=True)}
    expansions = 0
    best = None
    depth_cutoff = False
    while queue:
        state, path = queue.popleft()
        goal = is_goal(state)
        if not isinstance(goal, bool):
            raise ValueError('goal test must return a boolean')
        events.append({'kind': 'tos_visit', 'state': state, 'depth': len(path), 'goal': goal})
        if goal:
            value = answer(state)
            if not numeric(value):
                raise ValueError('answer function must return a finite number')
            candidate = (value, path)
            if model['objective'] == 'first':
                return value, path, 'done', expansions
            if best is None or (value < best[0] if model['objective'] == 'min' else value > best[0]):
                best = candidate
            continue
        if len(path) >= config.max_depth:
            depth_cutoff = True
            continue
        if expansions >= config.max_expansions:
            return None, [], 'expansion_limit', expansions
        expansions += 1
        events.append({'kind': 'tos_expand', 'state': state, 'depth': len(path)})
        children = successors(state)
        if not isinstance(children, list) or len(children) > 256:
            raise ValueError('successor function must return a bounded list')
        for child in children:
            validate_state(child, set(initial))
            key = json.dumps(child, sort_keys=True)
            if key not in visited:
                visited.add(key)
                queue.append((child, path + [{'state': child}]))
        if len(visited) > config.max_expansions * 256 + 1:
            raise ValueError('state budget exceeded')
    if depth_cutoff:
        return None, [], 'depth_limit', expansions
    if best is not None:
        return best[0], best[1], 'done', expansions
    return None, [], 'exhausted', expansions


def solve_thought_of_search(solver, question):
    started = time.time()
    example = {
        'initial_state': {'bags': 2, 'each': 5, 'removed': 3, 'stage': 0, 'value': 0},
        'successor_code': (
            'def successors(state):\n'
            '    return [{"bags":state["bags"], "each":state["each"], '
            '"removed":state["removed"], "stage":state["stage"]+1, '
            '"value":state["bags"]*state["each"] if state["stage"]==0 '
            'else state["value"]-state["removed"]}] if state["stage"]<2 else []'),
        'goal_code': 'def is_goal(state):\n    return state["stage"] == 2',
        'answer_code': 'def answer(state):\n    return state["value"]',
        'target': 'beads remaining', 'objective': 'first',
    }
    prompt = (
        f'Problem: {question}\nImplement a symbolic arithmetic search model BEFORE searching. '
        'Return the JSON search-model schema. initial_state contains ONLY given numeric facts '
        'and numeric counters. States may add or omit keys, but preserve facts needed by later steps. '
        'target describes the requested quantity and units. successor_code defines def successors(state) '
        'returning a list of successor state dictionaries. goal_code defines def is_goal(state) '
        'returning a boolean. answer_code defines def answer(state) extracting the computed answer. '
        'Use ordinary local assignments, if/elif/else, bounded for loops, return, arithmetic, '
        'comparisons, conditional expressions and list comprehensions. Helpers: range, abs, min, '
        'max, sum, len, round, ceil, floor, int, float, dict. Methods: dict.copy/get/update and '
        'list.append. Dictionary union (state | updates) and {**state, ...} are supported. '
        'No imports, while loops, recursion, arbitrary attributes or external calls. '
        'range bounds must be integral, with at most 256 values. '
        'Do NOT put a guessed final answer in initial_state or hardcode a solution in the goal '
        'test. Compute through state transitions. A dictionary literal reads the OLD state: '
        'new fields in it cannot read one another. Use sequential local assignments or separate '
        'stages for dependent calculations. Advance stage only when its arithmetic is computed. '
        'Goal tests must not fire before all requested arithmetic is complete. '
        'Use explicit units and conversions from the question; document needed conversion factors '
        'as named numeric state fields, not unexplained result constants. '
        'For arithmetic chains, use an integer stage and compute the next result from state '
        'facts at each transition. For constraints, enumerate feasible choices and test them. '
        'For minimum/maximum questions use objective="min"/"max" and enumerate ALL feasible '
        'goal states, answer returning their objective value. Otherwise objective="first". '
        f'The local BFS has depth {solver.cfg.max_depth} and {solver.cfg.max_expansions} expansions. '
        'Example FORMAT for a different problem, 2 bags of 5 beads then remove 3: '
        + json.dumps(example) + '. '
        'Use the actual problem facts, not the example numbers.')
    result = None
    status = 'model_error'
    expansions = 0
    model = None
    # Repair structural/runtime failures and unsuccessful searches, never reference-answer feedback.
    for repair in range(solver.cfg.retries + 1):
        model = solver.request(prompt, TOS_SCHEMA, 0.1, 'tos_model_synthesis', solver.cfg.max_tokens)
        if model is None:
            break
        attempt_events = []
        try:
            result, path, status, expansions = local_bfs(model, solver.cfg, attempt_events)
            solver.events.extend(attempt_events)
            solver.events.append({'kind': 'tos_search_attempt', 'repair': repair,
                                  'status': status, 'expansions': expansions, 'answer': result})
            if result is not None:
                if solver.cfg.audit_tos:
                    audit_prompt = (
                        f'Problem: {question}\nGenerated search model: {json.dumps(model)}\n'
                        f'Visited solution path: {json.dumps(path)}\nComputed answer: {result}.\n'
                        'Independently verify that the initial facts, number of stages, state '
                        'transitions, requested target and units match the question. Compute '
                        'the answer independently. Return valid=false if the model skips or '
                        'duplicates a stage, uses an unsupported assumption, or returns the '
                        'wrong quantity. Give the independent numeric answer and reason.')
                    audit = solver.request(audit_prompt, REVIEW_SCHEMA, 0.1,
                                           'tos_semantic_audit', solver.cfg.final_tokens)
                    accepted = (audit is not None and audit['valid'] and
                                math.isclose(audit['answer'], result, rel_tol=0, abs_tol=1e-4))
                    solver.events.append({'kind': 'tos_audit', 'review': audit,
                                          'computed_answer': result, 'accepted': accepted})
                    if not accepted:
                        solver.failure = 'tos_semantic_audit_rejected'
                        result = None
                        status = 'semantic_unverified'
                        prompt += ('\nPrevious model: ' + json.dumps(model) +
                                   '\nIndependent audit rejected the model: ' +
                                   str(audit.get('reason') if audit else 'no valid audit') +
                                   '. Repair the transitions and target without using a '
                                   'guessed final answer.')
                        continue
                break
            solver.failure = 'search_' + status
            feedback = ('Search ended with ' + status + '. Check successor progress, goal reachability, '
                        'and whether unrelated counters consume the depth budget. Do not change the '
                        'problem to fit a guessed answer. Use aggregate arithmetic instead of '
                        'enumerating time steps if the stated depth budget is insufficient.')
            prompt += '\nPrevious model: ' + json.dumps(model) + '\nLocal failure: ' + feedback
            prompt += '\nLast visited states: ' + json.dumps(attempt_events[-3:])
        except (ValueError, TypeError, KeyError, SyntaxError, ArithmeticError, IndexError,
                RecursionError) as exc:
            solver.events.extend(attempt_events)
            solver.failure = str(exc)
            status = 'model_error'
            solver.events.append({'kind': 'tos_model_rejected', 'error': str(exc), 'repair': repair})
            prompt += '\nPrevious model: ' + json.dumps(model) + '\nLocal failure: ' + str(exc)
            prompt += '. Repair the model while preserving the problem semantics.'
    if solver.calls >= solver.cfg.max_calls and result is None:
        status = 'call_limit'
    return {'method': TOS_METHOD, 'ans': result, 'status': status,
            'steps': path if result is not None else [], 'calls': solver.calls,
            'tokens': solver.tokens, 'sec': time.time()-started, 'trace': solver.events,
            'last_error': solver.failure, 'seed': solver.cfg.seed, 'config': vars(solver.cfg),
            'engine_version': '4', 'baseline_version': 'tos-arithmetic-3' if solver.cfg.audit_tos
            else 'tos-arithmetic-2',
            'search_model': model,
            'local_expansions': sum(event['kind'] == 'tos_expand' for event in solver.events),
            'adaptation': 'Per-problem restricted Python components; local BFS; no human review '
                          'or LLM calls within BFS; bounded Python interpreter; structural/runtime '
                          'and unsuccessful-search repair calls are counted.'}
