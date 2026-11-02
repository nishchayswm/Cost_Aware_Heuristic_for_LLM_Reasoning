"""Synthetic regressions for known failure paths, not replays of user model responses."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'kaggle')]
import word_bank_experiment as wb
from benchmark_engine import Config, METHODS, Solver, STEP_SCHEMA, REVIEW_SCHEMA, validate_step
from test_benchmark_engine import Scripted
from test_word_bank_notebook import test_bank


def step(equation, answer=None, text='Computed quantity'):
    return dict(step=text, equation=equation, done=answer is not None, answer=answer, target='total')


class MissingRecoveryTests(unittest.TestCase):
    def test_valid_chain_normalizes_and_wrong_chain_rejects(self):
        candidate = validate_step(step('20*15*10 = 300*10 = 3000', 3000))
        self.assertEqual(candidate['equation'], '20*15*10 = 3000')
        self.assertEqual(candidate['answer'], 3000)
        for equation in ('20*15*10 = 30*10 = 3000', '20*15*10 = apples = 3000'):
            with self.assertRaises(ValueError):
                validate_step(step(equation, 3000))

    def test_chain_with_unsupported_long_expression_cannot_pass(self):
        with self.assertRaises(ValueError):
            validate_step(step('2=2=2=2=2=2=2=2=2', 2))

    def test_program_digit_names_and_sequential_reassignment(self):
        obj = dict(calculations=[dict(name='apples_1tree', expression='20*15'),
                                 dict(name='Total', expression='apples_1tree*10'),
                                 dict(name='Total', expression='Total+0')],
                   result='Total', target='apples from ten trees')
        self.assertEqual(wb.execute_equations(obj)[0], 3000)

    def test_rain_program_final_expression_supported(self):
        obj = dict(calculations=[dict(name='rain_2pm', expression='(4-2)*4'),
                                 dict(name='rain_4pm', expression='(7-4)*3'),
                                 dict(name='rain_7pm', expression='(9-7)*0.5')],
                   result='2 + rain_2pm + rain_4pm + rain_7pm', target='rain gauge inches')
        self.assertEqual(wb.execute_equations(obj)[0], 20)

    def test_program_scientific_literal_not_replaced_as_identifier(self):
        obj = dict(calculations=[dict(name='total', expression='1e-3*1000')], result='total', target='value')
        self.assertEqual(wb.execute_equations(obj)[0], 1)

    def test_program_literal_result_must_be_grounded(self):
        obj = dict(calculations=[dict(name='total', expression='20*15*10')], result='3000', target='apples')
        self.assertEqual(wb.execute_equations(obj)[0], 3000)
        obj['result'] = '9999'
        with self.assertRaisesRegex(ValueError, 'must match'):
            wb.execute_equations(obj)

    def test_program_unsafe_syntax_still_rejected(self):
        for expr in ('open("x")', '__import__("os")', '(1).__class__', '9**9999'):
            obj = dict(calculations=[dict(name='total', expression=expr)], result='total', target='items')
            with self.assertRaises(ValueError):
                wb.execute_equations(obj)

    def test_program_failure_reports_actual_execution_reason(self):
        self.skipTest('Executable-equations method removed from search-only notebook')
        obj = dict(calculations=[dict(name='total', expression='unknown*2')], result='total', target='items')
        row = wb.run_method(Scripted([obj, obj]), 'An arithmetic question', wb.NEW_METHODS[2],
                            test_bank(), 11)
        self.assertEqual(row['status'], 'program_error')
        self.assertIn('Undefined variable: unknown', row['failure_reason'])
        self.assertEqual(row['calls'], 2)

    def test_step_retry_includes_failed_response(self):
        backend = Scripted(['{"bad_field": "wrong_shape"}', step('2*3=6', 6)])
        solver = Solver(backend, Config(retries=1))
        self.assertIsNotNone(solver.request('Question', STEP_SCHEMA, 0.1, 'generate', 512))
        self.assertIn('wrong_shape', backend.requests[1][0])
        self.assertIn('step, equation, done, answer and target', backend.requests[1][0])
        self.assertIsNone(solver.failure)

    def test_final_declaration_can_repeat_prior_wording(self):
        for method in METHODS[1:5]:
            replies = [step('20*15*10=3000', text='Total apples')]
            if method == METHODS[2]:
                replies.append({'score': 0.9})
            replies.append(step('20*15*10=3000', 3000, 'Total apples'))
            if method == METHODS[2]:
                replies.append({'score': 0.9})
            replies.append(dict(valid=True, answer=3000, reason='All ten trees.'))
            row = Solver(Scripted(replies), Config(candidates=1, beam=1)).solve('Apples?', method)
            self.assertEqual(row['ans'], 3000, method)

    def test_all_search_variants_recover_from_invalid_initial_candidates(self):
        for method in [*METHODS[2:5], *wb.NEW_METHODS]:
            prompts = []
            def backend(prompt, schema, temp, seed, limit):
                prompts.append(prompt)
                if schema == REVIEW_SCHEMA:
                    value = dict(valid=True, answer=3000, reason='All ten trees.')
                elif 'stalled' in prompt:
                    value = step('20*15*10=300*10=3000', 3000)
                    if schema != STEP_SCHEMA:
                        value = {'score': 0.9}
                elif schema == STEP_SCHEMA:
                    value = {'step': 'No equation yet'}
                else:
                    value = {'score': 0.9}
                return dict(text=json.dumps(value), prompt_tokens=30, completion_tokens=15, done_reason='stop')
            row = wb.run_method(backend, 'Apples from ten trees?', method, test_bank(), 11,
                                config=Config(candidates=1, beam=1, retries=1))
            self.assertEqual(row['ans'], 3000, (method, row['status']))
            self.assertTrue(any('stalled' in prompt for prompt in prompts))
            self.assertEqual(row['tokens'], 45 * row['calls'])

    def test_local_repeated_unfinished_steps_get_completion_feedback(self):
        generated = 0
        def backend(prompt, schema, temp, seed, limit):
            nonlocal generated
            if schema == REVIEW_SCHEMA:
                value = dict(valid=True, answer=3000, reason='Complete requested quantity.')
            else:
                generated += 1
                value = step('20*15*10=3000', 3000 if 'stalled' in prompt else None)
            return dict(text=json.dumps(value), prompt_tokens=30, completion_tokens=15, done_reason='stop')
        row = wb.run_method(backend, 'Apples from ten trees?', wb.NEW_METHODS[2], test_bank(), 11,
                            config=Config(candidates=1, beam=1, retries=1))
        self.assertEqual(row['ans'], 3000)
        self.assertEqual(generated, 3)
        self.assertEqual(row['calls'], 4)

    def test_persistent_invalid_candidates_stop_and_report_cause(self):
        for method in [METHODS[3], wb.NEW_METHODS[2]]:
            def backend(*args):
                return dict(text='not JSON', prompt_tokens=20, completion_tokens=5, done_reason='stop')
            row = wb.run_method(backend, 'Question', method, test_bank(), 11,
                                config=Config(candidates=1, beam=1, retries=1))
            self.assertIsNone(row['ans'])
            self.assertEqual(row['calls'], 4)
            self.assertIn('Expecting value', row['failure_reason'])


if __name__ == '__main__':
    unittest.main()
