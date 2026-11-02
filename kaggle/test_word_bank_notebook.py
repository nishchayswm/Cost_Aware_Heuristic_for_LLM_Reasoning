"""Offline tests: no model download, server, or inference required."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'kaggle'))
import word_bank_experiment as wb
from benchmark_data import fingerprint, prepare_benchmarks
from benchmark_engine import Config, JUDGE_SCHEMA, REVIEW_SCHEMA, STEP_SCHEMA
from tos_search import TOS_SCHEMA


def test_bank():
    examples = [
        ('Each bag has five beads. How many in two bags?',
         'Total beads in the bags = 2 * 5 = 10.\nThere are ten beads altogether.'),
        ('How far does a train travel in three hours at ten miles per hour?',
         'Distance traveled = speed multiplied by time.\n10 * 3 = 30 miles.'),
        ('How much change from twenty dollars after buying a five dollar notebook?',
         'Remaining money = initial money minus total cost.\n20 - 5 = 15 dollars.'),
    ]
    return wb.ReferenceWordBank([
        dict(source_id='gsm8k:train:' + str(i), question=q, solution=s,
             question_sha256=fingerprint(q)) for i, (q, s) in enumerate(examples)
    ])


def scripted_backend(prompt, schema, temperature, seed, max_tokens):
    if schema == STEP_SCHEMA:
        value = dict(step='The three bags contain twelve beads.', equation='3 * 4 = 12',
                     done=True, answer=12, target='beads in all bags')
    elif schema == REVIEW_SCHEMA:
        value = dict(valid=True, answer=12, reason='Three bags times four beads.')
    elif schema == JUDGE_SCHEMA:
        value = dict(score=0.9)
    elif schema == TOS_SCHEMA:
        value = dict(
            initial_state={'bags': 3, 'each': 4, 'stage': 0, 'value': 0},
            successor_code='def successors(state):\n'
                           '    return [{**state, "stage": 1, "value": state["bags"] * state["each"]}]',
            goal_code='def is_goal(state):\n    return state["stage"] == 1',
            answer_code='def answer(state):\n    return state["value"]',
            target='beads', objective='first',
        )
    elif schema == wb.EQUATIONS_SCHEMA:
        value = dict(calculations=[{'name': 'total', 'expression': '3*4'}],
                     result='total', target='beads')
    else:
        return dict(text='3 * 4 = 12\nFINAL ANSWER: 12', prompt_tokens=40,
                    completion_tokens=20, done_reason='stop')
    return dict(text=json.dumps(value), prompt_tokens=40, completion_tokens=20, done_reason='stop')


class WordBankTests(unittest.TestCase):
    def setUp(self):
        self.bank = test_bank()
        self.question = 'There are three bags with four beads each. How many beads altogether?'

    def test_all_9_dispatch_and_metering(self):
        self.assertEqual(len(wb.ALL_METHODS), 9)
        for method in wb.ALL_METHODS:
            with self.subTest(method=method):
                result = wb.run_method(scripted_backend, self.question, method, self.bank, 11)
                self.assertEqual(result['ans'], 12, result)
                self.assertEqual(result['tokens'], 60 * result['calls'])
                self.assertEqual(result['input_tokens'], 40 * result['calls'])
                self.assertTrue(result['usage_complete'])

    def test_early_stop_saves_two_samples(self):
        self.skipTest('Self-consistency removed from search-only notebook')
        fixed = wb.run_method(scripted_backend, self.question, wb.NEW_METHODS[4], self.bank, 11)
        early = wb.run_method(scripted_backend, self.question, wb.NEW_METHODS[5], self.bank, 11)
        self.assertEqual(fixed['calls'], 5)
        self.assertEqual(early['calls'], 3)
        self.assertEqual(fixed['ans'], early['ans'])

    def test_word_bank_context_score(self):
        _, relevant = self.bank.score(self.question, 'Total beads in the bags')
        _, irrelevant = self.bank.score(self.question, 'Volcanoes erupt molten lava')
        self.assertGreater(relevant, irrelevant)
        self.assertNotIn('12', self.bank.vectorizer.vocabulary_)

    def test_one_candidate_similarity_is_not_flat_minmax(self):
        good, _ = self.bank.score(self.question, 'beads bags')
        bad, _ = self.bank.score(self.question, 'volcano lava')
        self.assertGreater(good, bad)

    def test_wordbank_ablation_is_off_in_control(self):
        control = wb.run_method(scripted_backend, self.question, wb.NEW_METHODS[1], self.bank, 11)
        treatment = wb.run_method(scripted_backend, self.question, wb.NEW_METHODS[2], self.bank, 11)
        a = next(event for event in control['trace'] if event['kind'] == 'local_score')
        b = next(event for event in treatment['trace'] if event['kind'] == 'local_score')
        self.assertGreater(b['score'], a['score'])
        self.assertEqual(control['calls'], treatment['calls'])

    def test_invalid_equation_repair_cost_counted(self):
        calls = []
        def backend(*args):
            calls.append(args)
            result = scripted_backend(*args)
            if len(calls) == 1:
                value = json.loads(result['text'])
                value['equation'] = '3 * 4 = 99'
                result['text'] = json.dumps(value)
            return result
        result = wb.run_method(backend, self.question, wb.NEW_METHODS[2], self.bank, 11)
        self.assertEqual(result['ans'], 12)
        self.assertEqual(result['tokens'], 60 * len(calls))
        self.assertTrue(any('error' in event for event in result['trace']))

    def test_budget_stop_does_not_make_api_request(self):
        meter = wb.MeteredBackend(scripted_backend, 200)
        with self.assertRaises(wb.BudgetStop):
            meter('x' * 600, None, 0.1, 11, 512)
        self.assertEqual(len(meter.requests), 0)
        result = wb.run_method(scripted_backend, self.question, wb.NEW_METHODS[2], self.bank,
                               11, token_budget=200)
        self.assertEqual(result['status'], 'token_budget')
        self.assertIsNone(result['ans'])

    def test_budget_preserves_consistency_vote(self):
        self.skipTest('Self-consistency removed from search-only notebook')
        def expensive(*args):
            result = scripted_backend(*args)
            result.update(prompt_tokens=150, completion_tokens=200)
            return result
        result = wb.run_method(expensive, self.question, wb.NEW_METHODS[4], self.bank,
                               11, token_budget=600)
        self.assertEqual(result['ans'], 12)
        self.assertEqual(result['status'], 'budget_fallback')
        self.assertEqual(result['calls'], 1)

    def test_program_sequential_dependencies_and_unsafe_calls(self):
        self.skipTest('Executable-equations method removed from search-only notebook')
        program = dict(calculations=[{'name': 'subtotal', 'expression': '3*4'},
                                     {'name': 'remaining', 'expression': 'subtotal-2'}],
                       result='remaining', target='remaining beads')
        self.assertEqual(wb.execute_equations(program)[0], 10)
        program['calculations'][0]['expression'] = "__import__('os').system('x')"
        with self.assertRaises(ValueError):
            wb.execute_equations(program)

    def test_compact_state_keeps_equations_labels_question(self):
        steps = [dict(step='A very long explanatory sentence. ' * 10,
                      equation='3*4=12', target='total beads', answer=None, done=False)]
        compact = wb.search_prompt(self.question, steps, True)
        full = wb.search_prompt(self.question, steps, False)
        self.assertIn(self.question, compact)
        self.assertIn('total beads: 3*4=12', compact)
        self.assertLess(len(compact), len(full))

    def test_leakage_rejected_before_output(self):
        row = self.bank.records[0]
        case = dict(question=row['question'], expected=10, dataset='gsm8k', source_id=row['source_id'])
        with tempfile.TemporaryDirectory(dir=ROOT / 'kaggle') as temp:
            output = Path(temp) / 'results'
            with self.assertRaisesRegex(ValueError, 'leakage'):
                wb.run_word_bank_suite(scripted_backend, [case], self.bank, output, [11],
                                       [wb.METHODS[0]], 16000, {}, 'pilot')
            self.assertFalse(output.exists())

    def test_actual_training_split_disjoint(self):
        cache = ROOT / 'kaggle/data/official'
        subsets, _ = prepare_benchmarks(cache, download=False)
        bank = wb.prepare_word_bank(cache, subsets, 200)
        hashes = {row['question_sha256'] for row in bank.records}
        self.assertEqual(len(hashes), 200)
        self.assertFalse(hashes & {row['question_sha256'] for rows in subsets.values() for row in rows})

    def test_report_and_reference_not_in_prompts(self):
        case = dict(question=self.question, expected=918273.5, dataset='gsm8k', source_id='heldout:1')
        with tempfile.TemporaryDirectory(dir=ROOT / 'kaggle') as temp:
            output = Path(temp) / 'results'
            rows, table = wb.run_word_bank_suite(
                scripted_backend, [case], self.bank, output, [11], list(wb.ALL_METHODS),
                16000, {}, 'pilot')
            self.assertEqual(len(rows), 9)
            self.assertTrue((table['wrong'] == 1).all())
            self.assertTrue((table['accuracy_pct'] == 0).all())
            for row in rows:
                for request in row['requests']:
                    self.assertNotIn('918273', request['prompt'])
            for filename in ('word_bank_phrases.csv', 'word_bank_model.json', 'manifest.json',
                             'comparison_summary.csv', 'gsm8k_accuracy_vs_tokens.png'):
                self.assertTrue((output / filename).exists())


class NotebookTests(unittest.TestCase):
    def test_portable_cells_and_formatting(self):
        path = ROOT / 'kaggle/Word_bank_approch.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        namespace = {'__name__': 'notebook_offline_test'}
        for entry in notebook['cells']:
            if entry['cell_type'] != 'code':
                continue
            source = ''.join(entry['source'])
            ast.parse(source)
            self.assertLessEqual(max(map(len, source.splitlines()), default=0), 160)
            self.assertFalse(entry['outputs'])
            if 'source_module' in entry['metadata']:
                self.assertNotIn('from benchmark_', source)
                self.assertNotIn('from tos_search', source)
                exec(compile(source, str(path), 'exec'), namespace)
        bank = namespace['ReferenceWordBank'](test_bank().records)
        for method in namespace['ALL_METHODS']:
            result = namespace['run_method'](scripted_backend, 'Three bags of four beads?', method, bank, 11)
            self.assertEqual(result['ans'], 12, method)


if __name__ == '__main__':
    unittest.main()
