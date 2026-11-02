"""Failure-path and full-notebook audit with network/GPU/model operations mocked."""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'kaggle')]
import word_bank_experiment as wb
import build_word_bank_notebook as builder
from test_word_bank_notebook import test_bank, scripted_backend
from benchmark_engine import Config, Solver, STEP_SCHEMA, REVIEW_SCHEMA


def response(value, reason='stop'):
    return dict(text=value if isinstance(value, str) else json.dumps(value),
                prompt_tokens=40, completion_tokens=20, done_reason=reason)


class FailurePathTests(unittest.TestCase):
    def setUp(self):
        self.bank = test_bank()
        self.question = 'Three bags of four beads; remove two. How many remain?'

    def test_invalid_usage_never_marked_complete(self):
        for invalid in (None, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                def backend(*args):
                    result = scripted_backend(*args)
                    result['prompt_tokens'] = invalid
                    return result
                row = wb.run_method(backend, self.question, wb.METHODS[0], self.bank, 11)
                self.assertFalse(row['usage_complete'])
                self.assertIsNone(row['ans'])
                self.assertEqual(row['tokens'], 0)

    def test_cost_savings_undefined_when_transport_usage_missing(self):
        complete = wb.run_method(scripted_backend, self.question, wb.METHODS[0], self.bank, 11)
        def offline(*args):
            raise ConnectionError('Mock server disconnected')
        missing = wb.run_method(offline, self.question, wb.NEW_METHODS[0], self.bank, 11)
        rows = [complete, missing]
        for row in rows:
            row.update(dataset='gsm8k', expected=10, correct=False, source_id='test:1')
        with tempfile.TemporaryDirectory(dir=ROOT / 'kaggle') as temp:
            table = wb.experiment_report(rows, temp, 1.0, 2.0).set_index('method')
            self.assertTrue(wb.np.isnan(table.loc[wb.NEW_METHODS[0], 'mean_tokens']))
            self.assertTrue(wb.np.isnan(table.loc[wb.NEW_METHODS[0], 'token_savings_vs_cot_pct']))
            self.assertTrue(wb.np.isnan(table.loc[wb.NEW_METHODS[0], 'hypothetical_usd_per_problem']))
            self.assertEqual(table.loc[wb.NEW_METHODS[0], 'usage_incomplete_runs'], 1)

    def test_invalid_settings_fail_before_inference_or_output(self):
        case = dict(question=self.question, expected=10, dataset='gsm8k', source_id='test:1')
        changes = [dict(input_price=1), dict(input_price=-1, output_price=2),
                   dict(input_price=float('nan'), output_price=2), dict(token_budget=500.5),
                   dict(seeds=['eleven']), dict(seeds=[True]),
                   dict(cases=[dict(case, expected=float('nan'))]),
                   dict(cases=[dict(case, evaluation_split='evaluation')])]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=ROOT / 'kaggle') as temp:
                options = dict(backend=scripted_backend, cases=[case], bank=self.bank,
                               output=Path(temp) / 'results', seeds=[11], methods=[wb.METHODS[0]],
                               token_budget=16000, provenance={}, profile='pilot')
                options.update(change)
                with self.assertRaises(ValueError):
                    wb.run_word_bank_suite(**options)
                self.assertFalse(options['output'].exists())

    def test_truncated_answer_is_not_accepted(self):
        self.skipTest('Concise CoT removed from search-only notebook')
        def truncated(*args):
            return response('3*4=12\nFINAL ANSWER: 12', 'length')
        row = wb.run_method(truncated, self.question, wb.NEW_METHODS[0], self.bank, 11)
        self.assertIsNone(row['ans'])
        self.assertEqual(row['status'], 'truncated')
        self.assertEqual(row['tokens'], 60)

    def test_no_lexical_control_does_not_call_bank(self):
        with patch.object(self.bank, 'score', side_effect=AssertionError('Bank must be disabled')):
            row = wb.run_method(scripted_backend, self.question, wb.NEW_METHODS[0], self.bank, 11)
        self.assertEqual(row['ans'], 12)

    def test_tfidf_control_does_not_retrieve_reference_steps(self):
        with patch.object(self.bank, 'nearest', side_effect=AssertionError('Reference bank disabled')):
            row = wb.run_method(scripted_backend, self.question, wb.NEW_METHODS[1], self.bank, 11)
        self.assertEqual(row['ans'], 12)

    def test_rejected_terminal_can_recover_with_bounded_feedback(self):
        prompts = []
        def backend(prompt, schema, *args):
            prompts.append(prompt)
            if schema == REVIEW_SCHEMA:
                if '"answer": 10' in prompt:
                    return response(dict(valid=True, answer=10, reason='Correct remaining beads.'))
                return response(dict(valid=False, answer=10, reason='Subtotal is not remaining beads.'))
            if 'Earlier final answers' in prompt:
                return response(dict(step='Remove two beads.', equation='12-2=10',
                                     target='beads remaining', answer=10, done=True))
            return response(dict(step='Total beads.', equation='3*4=12',
                                 target='total beads', answer=12, done=True))
        row = wb.run_method(backend, self.question, wb.NEW_METHODS[2], self.bank, 11,
                            config=Config(candidates=1, retries=1))
        self.assertEqual(row['ans'], 10)
        self.assertEqual(row['calls'], 4)
        self.assertTrue(any('Earlier final answers' in prompt for prompt in prompts))

    def test_repeated_terminal_rejections_are_bounded(self):
        def backend(prompt, schema, *args):
            if schema == REVIEW_SCHEMA:
                return response(dict(valid=False, answer=10, reason='Wrong requested quantity.'))
            return response(dict(step='Total beads.', equation='3*4=12',
                                 target='total beads', answer=12, done=True))
        row = wb.run_method(backend, self.question, wb.NEW_METHODS[2], self.bank, 11,
                            config=Config(candidates=1, retries=1))
        self.assertIsNone(row['ans'])
        self.assertEqual(row['calls'], 3)
        self.assertTrue(any(event['kind'] == 'terminal_cache_hit' for event in row['trace']))
        self.assertEqual(row['status'], 'exhausted')

    def test_local_depth_limit_recorded(self):
        def backend(*args):
            return response(dict(step='Subtotal beads.', equation='3*4=12',
                                 target='subtotal beads', answer=None, done=False))
        row = wb.run_method(backend, self.question, wb.NEW_METHODS[1], self.bank, 11,
                            config=Config(max_depth=1, candidates=1))
        self.assertIsNone(row['ans'])
        self.assertEqual(row['status'], 'depth_limit')
        self.assertEqual(row['calls'], 1)

    def test_program_repair_respects_retry_setting(self):
        self.skipTest('Executable-equations method removed from search-only notebook')
        calls = []
        def backend(*args):
            calls.append(args)
            return response(dict(calculations=[dict(name='total', expression='undefined*4')],
                                 result='total', target='beads'))
        row = wb.run_method(backend, self.question, wb.NEW_METHODS[2], self.bank, 11,
                            config=Config(retries=0))
        self.assertIsNone(row['ans'])
        self.assertEqual(len(calls), 1)


class FullNotebookTests(unittest.TestCase):
    def test_sources_in_notebook_match_builder(self):
        notebook = json.loads((ROOT / 'kaggle/Word_bank_approch.ipynb').read_text(encoding='utf-8'))
        sources = {}
        for cell in notebook['cells']:
            module = cell['metadata'].get('source_module')
            if module:
                sources[module] = sources.get(module, '') + ''.join(cell['source'])
        for name, source in sources.items():
            expected = (builder.experiment_source() if name == 'word_bank_experiment'
                        else builder.readable_source(name))
            self.assertEqual(source, expected, name)

    def test_all_cells_through_zip_export_without_real_network_or_model(self):
        notebook = json.loads((ROOT / 'kaggle/Word_bank_approch.ipynb').read_text(encoding='utf-8'))
        shown, calls, client_options = [], [], []
        class FakeClient:
            def __init__(self, **kwargs):
                client_options.append(kwargs)
            def chat(self, **kwargs):
                calls.append(kwargs)
                opts = kwargs['options']
                result = scripted_backend(kwargs['messages'][0]['content'], kwargs.get('format'),
                                          opts['temperature'], opts['seed'], opts['num_predict'])
                return dict(message={'content': result['text']}, prompt_eval_count=result['prompt_tokens'],
                            eval_count=result['completion_tokens'], done_reason=result['done_reason'])
        fake_ollama = types.SimpleNamespace(
            Client=FakeClient,
            list=lambda: types.SimpleNamespace(model_dump=lambda **kwargs: {'models': [{'digest': 'mock'}]}),
        )
        fake_display = types.ModuleType('IPython.display')
        class DisplayArtifact:
            def __init__(self, path=None, filename=None):
                self.path = path or filename
        fake_display.FileLink = DisplayArtifact
        fake_display.Image = DisplayArtifact
        fake_display.display = lambda item: shown.append(item)
        fake_ipython = types.ModuleType('IPython')
        fake_ipython.display = fake_display
        fake_ipython.get_ipython = lambda: None
        fake_ipython.version_info = (9, 0)
        before = Path.cwd()
        with tempfile.TemporaryDirectory(dir=ROOT / 'kaggle') as temp:
            work = Path(temp)
            shutil.copytree(ROOT / 'kaggle/data/official', work / 'word_bank_data_cache')
            namespace = {'__name__': 'notebook_full_audit'}
            try:
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch('subprocess.run'))
                    stack.enter_context(patch('shutil.which', return_value=None))
                    stack.enter_context(patch('urllib.request.urlretrieve'))
                    stack.enter_context(patch('urllib.request.urlopen', return_value=io.BytesIO(b'{}')))
                    stack.enter_context(patch.dict(sys.modules, {'ollama': fake_ollama}))
                    stack.enter_context(patch.dict(sys.modules, {
                        'IPython': fake_ipython, 'IPython.display': fake_display}))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    for cell in notebook['cells']:
                        if cell['cell_type'] != 'code':
                            continue
                        source = ''.join(cell['source'])
                        if cell['metadata'].get('role') == 'setup':
                            source = source.replace("Path('/kaggle/working')", 'Path(' + repr(str(work)) + ')')
                        exec(compile(source, cell['id'], 'exec'), namespace)
                self.assertEqual(len(namespace['all_results']), 45)
                self.assertEqual(len(namespace['summary']), 9)
                self.assertEqual(len(namespace['difference_table']), 5)
                self.assertTrue(calls)
                self.assertTrue(all(call['model'] == 'qwen2.5:7b' for call in calls))
                self.assertTrue(all(call['options']['num_predict'] > 0 for call in calls))
                self.assertTrue(client_options)
                archive = Path(namespace['archive'])
                self.assertTrue(archive.exists())
                with zipfile.ZipFile(archive) as handle:
                    self.assertIn('ablation_differences.csv', handle.namelist())
                    self.assertIn('full_results.jsonl', handle.namelist())
                    self.assertEqual(len(handle.read('full_results.jsonl').splitlines()), 45)
                self.assertEqual(shown[-1].path, archive.name)
                # Exercise analysis with one method: it must still produce a readable empty CSV.
                namespace['summary'] = namespace['summary'].iloc[:1]
                with patch.dict(sys.modules, {'IPython': fake_ipython,
                                              'IPython.display': fake_display}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    exec(builder.ANALYSIS, namespace)
                columns = wb.pd.read_csv(namespace['OUTPUT'] / 'ablation_differences.csv').columns
                self.assertIn('mean_token_delta', columns)
            finally:
                os.chdir(before)


if __name__ == '__main__':
    unittest.main()
