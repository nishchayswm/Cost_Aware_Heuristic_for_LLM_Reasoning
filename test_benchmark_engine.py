"""Offline regressions for failure modes observed in results_2.jsonl."""
import ast
import json
import unittest
import tempfile
from dataclasses import replace
from pathlib import Path

from benchmark_engine import (Config, METHODS, Solver, STEP_SCHEMA, REVIEW_SCHEMA,
                              JUDGE_SCHEMA, equation_status, extract_explicit_answer,
                              safe_arithmetic, validate_step, lexical_relevance, evaluate)


def step(equation='3 * 12 = 36', done=False, answer=None, text='Compute total apples'):
    return dict(step=text, equation=equation, done=done, answer=answer,
                target='apples left' if done else '')


class Scripted:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def __call__(self, prompt, schema, temp, seed, limit):
        self.requests.append((prompt, schema, temp, seed, limit))
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, dict) and 'text' in value:
            return value
        return dict(text=json.dumps(value) if isinstance(value, dict) else value,
                    prompt_tokens=10, completion_tokens=5, done_reason='stop')


class ArithmeticTests(unittest.TestCase):
    def test_equations_and_format_errors_are_distinct(self):
        self.assertEqual(equation_status('3 * 12 = 36'), ('valid', 36.0))
        self.assertEqual(equation_status('3 * 12 = 35'), ('invalid', 35.0))
        self.assertEqual(equation_status('3 * 12 = 36 apples')[0], 'unsupported')
        self.assertEqual(equation_status('')[0], 'missing')

    def test_unsafe_and_expensive_arithmetic(self):
        for expression in ('__import__("os")', 'True', '1 / 0', '9 ** 999999', '1e300'):
            self.assertIsNone(safe_arithmetic(expression), expression)
        self.assertEqual(safe_arithmetic('(157 + 11) // 12'), 14)

    def test_explicit_answer_formats(self):
        for text, expected in [('FINAL ANSWER: $1,234.50', 1234.5),
                               ('FINAL ANSWER: -0.5', -.5),
                               ('**FINAL ANSWER: 21**', 21)]:
            self.assertEqual(extract_explicit_answer(text), expected)

    def test_no_intermediate_numeric_fallback(self):
        for text in ('15 pages for 6 days', '60 + 70 + 80 + 90 = 300',
                     'FINAL ANSWER: 21 apples', 'FINAL ANSWER: <number>',
                     'FINAL ANSWER: 1\nFINAL ANSWER: 2'):
            self.assertIsNone(extract_explicit_answer(text))

    def test_schema_types_and_validity(self):
        cases = [step(done=True, answer=True), step(done='false'), step(equation='none'),
                 step(equation='3 * 12 = 35'), step(answer=36), step(equation='3*12=36 apples')]
        for candidate in cases:
            with self.assertRaises(ValueError):
                validate_step(candidate)

    def test_separate_expression_and_result_are_checked_locally(self):
        candidate = dict(step('', True, 36, ''), target='total apples',
                         expression='3 * 12', result=36)
        parsed = validate_step(candidate)
        self.assertEqual(parsed['equation'], '3 * 12 = 36')
        self.assertEqual(parsed['step'], 'Computed total apples')
        with self.assertRaisesRegex(ValueError, 'expression_result_mismatch'):
            validate_step(dict(candidate, result=35))

    def test_symbolic_equation_requires_verified_expression_result(self):
        candidate = dict(step('3 * 12 = target', True, 36),
                         expression='3 * 12', result=36)
        self.assertEqual(validate_step(candidate)['equation'], '3 * 12 = 36')
        with self.assertRaisesRegex(ValueError, 'equation_unsupported'):
            validate_step(step('3 * 12 = target', True, 36))

    def test_relevance_is_bounded(self):
        scores = lexical_relevance(['apples left', 'train distance'], 'apples left')
        self.assertEqual(scores, [1., 0.])
        self.assertEqual(lexical_relevance(['same', 'same'], 'same'), [.5, .5])


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(candidates=1, beam=1, retries=1, max_expansions=6)

    def test_success_with_shared_terminal_review_and_cost(self):
        backend = Scripted([step(), step('36 - 15 = 21', True, 21, 'Compute apples remaining'),
                            dict(valid=True, answer=21, reason='The target is apples left')])
        row = evaluate(backend, 'Three boxes of twelve apples, sell fifteen. How many left?',
                       21, METHODS[3], self.cfg)
        self.assertTrue(row['correct'])
        self.assertEqual(row['calls'], 3)
        self.assertEqual(row['tokens'], 45)
        self.assertEqual(row['outcome'], 'correct')

    def test_tiny_judge_score_overshoot_is_logged_and_clamped(self):
        backend = Scripted([{'score': 1.0000000000000002}])
        solver = Solver(backend, self.cfg)
        self.assertEqual(solver.request('score', JUDGE_SCHEMA, .1, 'judge', 20)['score'], 1.0)
        self.assertGreater(solver.events[0]['score_clamped_from'], 1)

    def test_review_appeal_can_recover_false_negative(self):
        backend = Scripted([step('15*6=90', True, 90, 'Pages'),
                            dict(valid=False, answer=60, reason='Misread'),
                            dict(valid=True, answer=90, reason='Correct')])
        row = Solver(backend, replace(self.cfg, review_appeal=True)).solve('Pages?', METHODS[3])
        self.assertEqual(row['ans'], 90)
        self.assertEqual(row['calls'], 3)

    def test_premature_final_is_reviewed_and_recovered(self):
        backend = Scripted([
            step('4 * 6 = 24', True, 24, 'Notebook spending'),
            dict(valid=False, answer=26, reason='Requested money remaining'),
            step('50 - 24 = 26', True, 26, 'Money remaining'),
            dict(valid=True, answer=26, reason='Complete')])
        row = Solver(backend, self.cfg).solve('50 dollars, spend 24. Money left?', METHODS[1])
        self.assertEqual(row['ans'], 26)
        self.assertIn('rejected', backend.requests[2][0])

    def test_final_substring_in_description_does_not_stop(self):
        backend = Scripted([step(text='Prepare the final subtraction'),
                            step('36-15=21', True, 21, 'Remaining apples'),
                            dict(valid=True, answer=21, reason='Correct target')])
        self.assertEqual(Solver(backend, self.cfg).solve('Apples left?', METHODS[1])['ans'], 21)

    def test_parse_error_is_retried_and_logged(self):
        backend = Scripted(['not json', step('15*6=90', True, 90, 'Pages read'),
                            dict(valid=True, answer=90, reason='Pages after six days')])
        row = Solver(backend, self.cfg).solve('Pages read?', METHODS[1])
        self.assertEqual(row['ans'], 90)
        self.assertTrue(any('error' in event for event in row['trace']))
        self.assertEqual(row['calls'], 3)

    def test_failed_parse_is_missing_not_wrong_numeric_answer(self):
        backend = Scripted(['bad json', 'bad json'])
        row = evaluate(backend, 'Apples?', 21, METHODS[1], self.cfg)
        self.assertEqual(row['outcome'], 'missing')
        self.assertIsNone(row['ans'])

    def test_incomplete_steps_never_become_final_answers(self):
        backend = Scripted([step()])
        row = Solver(backend, replace(self.cfg, max_depth=1)).solve('Apples left?', METHODS[1])
        self.assertIsNone(row['ans'])
        self.assertEqual(row['status'], 'depth_limit')
        self.assertEqual(len(row['steps']), 1)

    def test_rephrased_equation_repetition_gets_replacement(self):
        backend = Scripted([step(), step('3*12=36', text='Another wording for total'),
                            step('36-15=21', True, 21, 'Remaining apples'),
                            dict(valid=True, answer=21, reason='Complete')])
        row = Solver(backend, self.cfg).solve('Apples left?', METHODS[1])
        self.assertEqual(row['ans'], 21)
        self.assertTrue(any(e.get('reason') == 'repeated_step' for e in row['trace']))

    def test_truncation_increases_budget(self):
        backend = Scripted([dict(text='{', done_reason='length', prompt_tokens=10, completion_tokens=512),
                            step('15*6=90', True, 90, 'Pages read'),
                            dict(valid=True, answer=90, reason='Complete')])
        row = Solver(backend, self.cfg).solve('Pages read?', METHODS[1])
        self.assertEqual(row['ans'], 90)
        self.assertEqual(backend.requests[1][4], 1024)
        self.assertEqual(row['tokens'], 552)

    def test_standard_cot_stays_one_call_and_reports_truncation(self):
        backend = Scripted([dict(text='FINAL ANSWER: 3', done_reason='length')])
        row = Solver(backend, self.cfg).solve('Pens cost?', METHODS[0])
        self.assertEqual(row['calls'], 1)
        self.assertIsNone(row['ans'])
        self.assertEqual(row['status'], 'truncated')

    def test_call_budget_includes_review(self):
        backend = Scripted([step('15*6=90', True, 90, 'Pages read')])
        row = Solver(backend, replace(self.cfg, max_calls=1)).solve('Pages?', METHODS[1])
        self.assertEqual(row['calls'], 1)
        self.assertIsNone(row['ans'])
        self.assertEqual(row['status'], 'call_limit')

    def test_no_grounding_no_terminal_acceptance(self):
        backend = Scripted([step('', True, 90, 'Pages read')])
        row = Solver(backend, replace(self.cfg, max_expansions=1)).solve('Pages?', METHODS[1])
        self.assertIsNone(row['ans'])

    def test_reviewer_disagreement_rejects_terminal(self):
        backend = Scripted([step('15*6=90', True, 90, 'Pages'),
                            dict(valid=True, answer=6, reason='Disagrees')])
        row = Solver(backend, replace(self.cfg, max_expansions=1)).solve('Pages?', METHODS[1])
        self.assertIsNone(row['ans'])

    def test_tot_charges_judge_and_review(self):
        backend = Scripted([step('15*6=90', True, 90, 'Pages'), {'score': .9},
                            dict(valid=True, answer=90, reason='Complete')])
        row = Solver(backend, self.cfg).solve('Pages?', METHODS[2])
        self.assertEqual(row['ans'], 90)
        self.assertEqual(row['calls'], 3)

    def test_seeded_requests_and_no_reference_leakage(self):
        def run():
            b = Scripted(['FINAL ANSWER: 21'])
            row = evaluate(b, 'Compute apples left', 987654321, METHODS[0], self.cfg)
            return b, row
        first, row = run()
        second, _ = run()
        self.assertEqual(first.requests, second.requests)
        self.assertNotIn('987654321', first.requests[0][0])
        self.assertEqual(row['outcome'], 'wrong')

    def test_terminal_selection_reviews_all_generated_candidates(self):
        backend = Scripted([step('8*6=48', True, 48, 'Subtotal'),
                            step('48-30=18', True, 18, 'Cupcakes left'),
                            dict(valid=False, answer=18, reason='Subtotal is not requested'),
                            dict(valid=True, answer=18, reason='Complete')])
        row = Solver(backend, replace(self.cfg, candidates=2)).solve('Cupcakes left?', METHODS[3])
        self.assertEqual(row['ans'], 18)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            Config(max_calls=0)


class NotebookTests(unittest.TestCase):
    def test_portable_notebook_embeds_current_engine(self):
        root = Path(__file__).resolve().parent
        notebook = json.loads((root/'kaggle/cost_aware_reasoning_benchmark.ipynb').read_text(encoding='utf-8'))
        from sync_notebook import readable_source
        modules = {cell['metadata']['source_module']: ''.join(cell['source'])
                   for cell in notebook['cells'] if 'source_module' in cell['metadata']}
        self.assertIn('benchmark_engine', modules)
        for name, code in modules.items():
            self.assertEqual(code, readable_source(name))
            ast.parse(code)
            self.assertLess(max(map(len, code.splitlines())), 200)
        # Execute the same plain cells Kaggle will execute, without invoking Ollama.
        namespace = {'__name__': '__main__'}
        exec("Model = 'qwen2.5:7b'", namespace)
        for code in modules.values():
            exec(compile(code, '<notebook-cell>', 'exec'), namespace)
        self.assertTrue(callable(namespace['run_suite']))
        self.assertFalse(any(cell.get('outputs') for cell in notebook['cells']))


class ReportingTests(unittest.TestCase):
    def test_complete_seeded_suite_writes_reports_and_traces(self):
        from run_benchmark import run_suite
        from tos_search import TOS_SCHEMA
        from test_tos_search import arithmetic_model
        def backend(prompt, schema, temp, seed, limit):
            if schema == TOS_SCHEMA:
                value = arithmetic_model()
            elif schema == JUDGE_SCHEMA:
                value = {'score': .9}
            elif schema == REVIEW_SCHEMA:
                value = dict(valid=True, answer=21, reason='Requested apples left')
            elif schema == STEP_SCHEMA:
                value = step('36-15=21', True, 21, 'Apples left')
            else:
                value = 'FINAL ANSWER: 21'
            return dict(text=json.dumps(value) if isinstance(value, dict) else value,
                        prompt_tokens=10, completion_tokens=5, done_reason='stop')
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            output = Path(directory)/'synthetic_regression_only'
            rows = run_suite(backend, output, seeds=[11, 23], problems=[('Apples left?', 21)])
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(r['correct'] for r in rows))
            for filename in ('full_results.jsonl', 'manifest.json', 'summary_table.csv',
                             'per_seed_summary.csv', 'termination_counts.csv',
                             'pareto_frontier.png', 'token_consumption.png'):
                self.assertTrue((output/filename).is_file(), filename)
            self.assertTrue(all(r['trace'] for r in rows))

    def test_reporting_keeps_missing_in_accuracy_denominator(self):
        from plot_results import generate_report_and_plots
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory)/'input.jsonl'
            rows = [dict(method='test', problem_id=i, ans=21 if i == 0 else None,
                         correct=True if i == 0 else None, calls=1, tokens=10, sec=1)
                    for i in range(2)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            summary = generate_report_and_plots(directory, [path])
            self.assertEqual(summary.iloc[0]['Accuracy'], 50)
            self.assertEqual(summary.iloc[0]['Missing'], 1)

    def test_reporting_rejects_mixed_versions_and_duplicate_rows(self):
        from plot_results import load_all_results
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory)/'input.jsonl'
            row = dict(method='test', problem_id=0, ans=1, correct=True)
            path.write_text(json.dumps(row)+'\n'+json.dumps(dict(row, engine_version='3'))+'\n',
                            encoding='utf-8')
            with self.assertRaises(ValueError):
                load_all_results([path])
            path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_all_results([path])


if __name__ == '__main__':
    unittest.main()
