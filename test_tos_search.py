"""Offline ToS tests: search, synthesis feedback, execution bounds and accounting."""
import json
import unittest
from dataclasses import replace

from benchmark_engine import Config, METHODS, Solver, evaluate
from test_benchmark_engine import Scripted
from tos_search import RestrictedFunction, TOS_METHOD, TOS_SCHEMA, local_bfs, validate_model


def arithmetic_model():
    return {
        'initial_state': {'stage': 0, 'value': 0, 'boxes': 3, 'each': 12, 'sold': 15},
        'successor_code': (
            'def successors(state):\n'
            '    return [{"stage":state["stage"]+1, "boxes":state["boxes"], '
            '"each":state["each"], "sold":state["sold"], '
            '"value":state["boxes"]*state["each"] if state["stage"]==0 '
            'else state["value"]-state["sold"]}] if state["stage"]<2 else []'),
        'goal_code': 'def is_goal(state):\n    return state["stage"] == 2',
        'answer_code': 'def answer(state):\n    return state["value"]',
        'target': 'apples left', 'objective': 'first',
    }


class ToSSearchTests(unittest.TestCase):
    def test_synthesis_then_local_search_without_model_calls(self):
        backend = Scripted([arithmetic_model()])
        row = evaluate(backend, 'Three boxes of twelve apples; sell fifteen. How many left?',
                       21, TOS_METHOD, Config(seed=23))
        self.assertTrue(row['correct'])
        self.assertEqual(row['calls'], 1)
        self.assertEqual(row['tokens'], 15)
        self.assertEqual(row['local_expansions'], 2)
        self.assertEqual(len(row['steps']), 2)
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(backend.requests[0][1], TOS_SCHEMA)
        self.assertEqual(row['baseline_version'], 'tos-arithmetic-2')

    def test_syntactic_failure_is_repaired_and_costed(self):
        invalid = dict(arithmetic_model(), successor_code='import os')
        backend = Scripted([invalid, arithmetic_model()])
        row = Solver(backend, Config(retries=1)).solve('Apples left?', TOS_METHOD)
        self.assertEqual(row['ans'], 21)
        self.assertEqual(row['calls'], 2)
        self.assertIn('Local failure:', backend.requests[1][0])
        self.assertTrue(any(e['kind'] == 'tos_model_rejected' for e in row['trace']))

    def test_bad_json_is_retried(self):
        backend = Scripted(['{broken', arithmetic_model()])
        row = Solver(backend, Config(retries=1)).solve('Apples?', TOS_METHOD)
        self.assertEqual(row['ans'], 21)
        self.assertEqual(row['calls'], 2)

    def test_model_semantics_are_scored_not_repaired_using_reference(self):
        backend = Scripted([arithmetic_model()])
        row = evaluate(backend, 'Apples left?', 987654321, TOS_METHOD)
        self.assertEqual(row['outcome'], 'wrong')
        self.assertEqual(row['calls'], 1)
        self.assertNotIn('987654321', backend.requests[0][0])

    def test_semantic_audit_rejects_wrong_transition_count(self):
        backend = Scripted([arithmetic_model(),
                            dict(valid=False, answer=22, reason='Skipped a stage')])
        row = Solver(backend, Config(retries=0, audit_tos=True)).solve(
            'Three boxes of twelve apples; sell fifteen. How many left?', TOS_METHOD)
        self.assertIsNone(row['ans'])
        self.assertEqual(row['status'], 'semantic_unverified')
        self.assertEqual(row['calls'], 2)
        self.assertTrue(any(e['kind'] == 'tos_audit' for e in row['trace']))

    def test_optimization_enumerates_all_feasible_goals(self):
        model = {
            'initial_state': {'stage': 0, 'cost': 0},
            'successor_code': 'def successors(state):\n    return '
                              '[{"stage":1,"cost":9*k+2*(20-6*k)} for k in range(4)] '
                              'if state["stage"]==0 else []',
            'goal_code': 'def is_goal(state):\n    return state["stage"]==1',
            'answer_code': 'def answer(state):\n    return state["cost"]',
            'target': 'minimum cost for exactly 20 pens', 'objective': 'min',
        }
        events = []
        value, path, status, _ = local_bfs(model, Config(), events)
        self.assertEqual(value, 31)
        self.assertEqual(status, 'done')
        self.assertEqual(sum(e.get('goal', False) for e in events), 4)

    def test_constraint_enumeration(self):
        model = {
            'initial_state': {'stage': 0, 'dimes': 0},
            'successor_code': 'def successors(state):\n    return '
                              '[{"stage":1,"dimes":d} for d in range(31)] '
                              'if state["stage"]==0 else []',
            'goal_code': 'def is_goal(state):\n    return state["stage"]==1 and '
                         '10*state["dimes"]+25*(30-state["dimes"])==510',
            'answer_code': 'def answer(state):\n    return state["dimes"]',
            'target': 'number of dimes', 'objective': 'first',
        }
        self.assertEqual(local_bfs(model, Config(), [])[0], 16)

    def test_rounding_and_short_circuit_helpers(self):
        function = RestrictedFunction(
            'def answer(state):\n    return ceil((state["signed"]-15)/12)', 'answer')
        self.assertEqual(function({'signed': 172}), 14)
        goal = RestrictedFunction(
            'def is_goal(state):\n    return state["x"] != 0 and 1/state["x"] > 0', 'is_goal')
        self.assertFalse(goal({'x': 0}))

    def test_exhaustion_and_cycle_detection(self):
        model = dict(arithmetic_model(), successor_code='def successors(state):\n    return [state]')
        answer, _, status, expansions = local_bfs(model, Config(), [])
        self.assertIsNone(answer)
        self.assertEqual(status, 'exhausted')
        self.assertEqual(expansions, 1)

    def test_depth_and_expansion_limits_do_not_guess(self):
        for config, expected in [(Config(max_depth=1), 'depth_limit'),
                                 (Config(max_expansions=1), 'expansion_limit')]:
            answer, _, status, _ = local_bfs(arithmetic_model(), config, [])
            self.assertIsNone(answer)
            self.assertEqual(status, expected)

    def test_optimization_cannot_return_partial_best_after_cutoff(self):
        model = {
            'initial_state': {'stage': 0, 'cost': 0},
            'successor_code': 'def successors(state):\n    return '
                              '[{"stage":1,"cost":10},{"stage":2,"cost":0}] '
                              'if state["stage"]==0 else [{"stage":1,"cost":1}]',
            'goal_code': 'def is_goal(state):\n    return state["stage"]==1',
            'answer_code': 'def answer(state):\n    return state["cost"]',
            'target': 'minimum cost', 'objective': 'min',
        }
        answer, _, status, _ = local_bfs(model, Config(max_expansions=1), [])
        self.assertIsNone(answer)
        self.assertEqual(status, 'expansion_limit')

    def test_invalid_initial_and_child_states_rejected(self):
        for initial in ({'x': True}, {'x': float('nan')}, {}, {'x': '21'}):
            with self.assertRaises(ValueError):
                validate_model(dict(arithmetic_model(), initial_state=initial))
        invalid = dict(arithmetic_model(), successor_code='def successors(state):\n    return [{"x":"bad"}]')
        with self.assertRaises(ValueError):
            validate_model(invalid)

    def test_constant_goal_and_answer_rejected(self):
        for field, code in [('goal_code', 'def is_goal(state):\n    return True'),
                            ('answer_code', 'def answer(state):\n    return 21')]:
            with self.assertRaises(ValueError):
                validate_model(dict(arithmetic_model(), **{field: code}))

    def test_unsupported_code_cannot_execute(self):
        for code in [
            'def answer(state):\n    return __import__("os").getcwd()',
            'def answer(state):\n    return state.__class__',
            'def answer(state):\n    while True: pass',
            'def answer(state):\n    return open("anything")',
            'def answer(state=print("side effect")):\n    return state["x"]',
        ]:
            with self.assertRaises((ValueError, SyntaxError)):
                RestrictedFunction(code, 'answer')

    def test_collection_and_numeric_limits(self):
        for expression in ['range(1000000000)', '9 ** state["x"]',
                           '[x for x in range(256) for y in range(256)]']:
            code = 'def answer(state):\n    return ' + expression + ' if state["x"] else 0'
            fn = RestrictedFunction(code, 'answer')
            with self.assertRaises((ValueError, OverflowError)):
                fn({'x': 100})

    def test_call_limit_prevents_unbounded_repairs(self):
        backend = Scripted(['bad'])
        row = Solver(backend, Config(max_calls=1)).solve('Apples?', TOS_METHOD)
        self.assertIsNone(row['ans'])
        self.assertEqual(row['calls'], 1)
        self.assertEqual(row['status'], 'call_limit')

    def test_tos_only_runner_selection(self):
        import tempfile
        from pathlib import Path
        from run_benchmark import run_suite
        backend = Scripted([arithmetic_model()])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            output = Path(directory)/'tos_only_synthetic'
            rows = run_suite(backend, output, seeds=[11], problems=[('Apples?', 21)], methods=[TOS_METHOD])
            self.assertEqual(len(rows), 1)
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['methods'], [TOS_METHOD])

    def test_sequential_assignments_and_state_mutation_are_isolated(self):
        fn = RestrictedFunction(
            'def successors(state):\n'
            '    state["dave"] = 3 * state["weight"]\n'
            '    state["craig"] = 0.2 * state["dave"]\n'
            '    state["mark"] = state["craig"] - 50\n'
            '    state["stage"] += 1\n'
            '    return [state]', 'successors')
        state = {'weight': 175, 'stage': 0}
        child = fn(state)[0]
        self.assertEqual(child['mark'], 55)
        self.assertEqual(child['stage'], 1)
        self.assertEqual(state, {'weight': 175, 'stage': 0})

    def test_conditionals_dict_copy_update_and_union(self):
        fn = RestrictedFunction(
            'def successors(state):\n'
            '    if state["stage"] == 0:\n'
            '        next_state = state.copy()\n'
            '        next_state.update({"value": state.get("value", 0) + 8})\n'
            '        return [next_state | {"stage": 1}]\n'
            '    elif state["stage"] == 1:\n'
            '        return [dict(state, stage=2, value=state["value"]+9)]\n'
            '    else:\n'
            '        return []', 'successors')
        child = fn({'stage': 0, 'value': 2})[0]
        self.assertEqual(fn(child)[0]['value'], 19)
        self.assertEqual(fn({'stage': 2, 'value': 19}), [])

    def test_new_and_omitted_state_keys_are_permitted(self):
        model = dict(arithmetic_model(),
                     successor_code='def successors(state):\n    return [{"stage":2, "value":'
                                    'state["boxes"]*state["each"]-state["sold"]}]')
        self.assertEqual(local_bfs(model, Config(), [])[0], 21)

    def test_for_loops_are_bounded_and_support_integral_float_bounds(self):
        fn = RestrictedFunction(
            'def successors(state):\n'
            '    children = []\n'
            '    for i in range(state["n"]):\n'
            '        if i == 1:\n'
            '            continue\n'
            '        children.append({**state, "value": int(i), "stage":1})\n'
            '    return children', 'successors')
        self.assertEqual([c['value'] for c in fn({'n': 3.0})], [0, 2])
        with self.assertRaises(ValueError):
            fn({'n': 1000000})

    def test_exhaustion_feedback_is_repaired_without_reference_answer(self):
        stalled = dict(arithmetic_model(), successor_code='def successors(state):\n    return [state]')
        backend = Scripted([stalled, arithmetic_model()])
        row = Solver(backend, Config(retries=1)).solve('Apples left?', TOS_METHOD)
        self.assertEqual(row['ans'], 21)
        self.assertEqual(row['calls'], 2)
        self.assertIn('Search ended with exhausted', backend.requests[1][0])
        self.assertEqual(row['local_expansions'], 3)

    def test_depth_limit_feedback_preserves_missing_if_repair_fails(self):
        backend = Scripted([arithmetic_model(), arithmetic_model()])
        row = Solver(backend, Config(retries=1, max_depth=1)).solve('Apples?', TOS_METHOD)
        self.assertIsNone(row['ans'])
        self.assertEqual(row['status'], 'depth_limit')
        self.assertEqual(row['calls'], 2)

    def test_syntax_support_does_not_correct_semantic_errors(self):
        model = dict(arithmetic_model(),
                     successor_code='def successors(state):\n    return [dict(state, stage=2)]')
        row = evaluate(Scripted([model]), 'Apples left?', 21, TOS_METHOD)
        self.assertEqual(row['ans'], 0)
        self.assertEqual(row['outcome'], 'wrong')

    def test_observed_rainfall_program_with_if_elif(self):
        model = {
            'initial_state': {'stage': 0, 'rain': 2, 'rate1': 4, 'rate2': 3, 'rate3': .5},
            'successor_code': (
                'def successors(state):\n'
                '    if state["stage"] == 0:\n'
                '        return [dict(state, rain=state["rain"]+state["rate1"]*2, stage=1)]\n'
                '    elif state["stage"] == 1:\n'
                '        return [dict(state, rain=state["rain"]+state["rate2"]*3, stage=2)]\n'
                '    elif state["stage"] == 2:\n'
                '        return [dict(state, rain=state["rain"]+state["rate3"]*2, stage=3)]\n'
                '    return []'),
            'goal_code': 'def is_goal(state):\n    return state["stage"] == 3',
            'answer_code': 'def answer(state):\n    return state["rain"]',
            'target': 'rainwater at 9pm', 'objective': 'first',
        }
        self.assertEqual(local_bfs(model, Config(), [])[0], 20)

    def test_missing_derived_key_gives_actionable_feedback(self):
        fn = RestrictedFunction(
            'def successors(state):\n    return [{"apples_today":state["apples"]+4, '
            '"oranges_today":2*state["apples_today"]}]', 'successors')
        with self.assertRaisesRegex(ValueError, 'dictionary literals read the OLD state'):
            fn({'apples': 3})


if __name__ == '__main__':
    unittest.main()
