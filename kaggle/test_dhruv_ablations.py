"""Offline checks for the six-variant Kaggle notebook and ranking ablations."""
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import build_dhruv_ablation_notebook as builder


class AblationNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(builder.OUTPUT.read_text(encoding='utf-8'))

    def test_bundle_matches_branch_and_runner(self):
        revision = builder.original.git('rev-parse', builder.original.REF).decode().strip()
        self.assertEqual(self.notebook['metadata']['source_commit'], revision)
        roles = [c['metadata'].get('role') for c in self.notebook['cells']]
        self.assertEqual(roles.count('ablation_runner'), 1)
        self.assertEqual(roles.count('run'), 1)
        for item in self.notebook['cells']:
            role = item['metadata'].get('role')
            if role == 'source_file':
                relative = item['metadata']['file']
                original = builder.original.git('show', f'{revision}:final_proj/{relative}')
                self.assertEqual(item['metadata']['sha256'], hashlib.sha256(original).hexdigest())
            elif role == 'ablation_runner':
                original = builder.RUNNER.read_bytes()
                self.assertEqual(item['metadata']['sha256'], hashlib.sha256(original).hexdigest())

    def test_feature_removal_and_reports(self):
        with tempfile.TemporaryDirectory(dir=builder.ROOT / 'kaggle') as folder:
            root = Path(folder)
            before = Path.cwd()
            sys.path.insert(0, str(root))
            try:
                os.chdir(root)
                for item in self.notebook['cells']:
                    role = item['metadata'].get('role')
                    source = ''.join(item['source'])
                    if role in ('source_file', 'ablation_runner'):
                        path = root / item['metadata']['file']
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(source.split('\n', 1)[1], encoding='utf-8')
                    elif role == 'bundled_data':
                        exec(compile(source, item['id'], 'exec'), {})
                    elif role in ('settings', 'offline_tests', 'ollama_setup', 'run', 'download'):
                        compile(source, item['id'], 'exec')
                from costbench.__main__ import config as load_config
                from costbench.backend import Session, append_json
                from costbench.graph import Candidate
                from costbench.methods import Node
                ablation = importlib.import_module('ablation_runner')
                base = load_config(root / 'configs/default.json')
                self.assertEqual(len(ablation.VARIANTS), 6)
                original_weights = dict(base['weights'])
                for excluded in (None, *ablation.FEATURES):
                    cfg = ablation.variant_config(base, excluded)
                    self.assertAlmostEqual(sum(cfg['weights'].values()), 1)
                    if excluded:
                        self.assertEqual(cfg['weights'][excluded], 0)
                    session = Session(cfg, 'synthetic', 'heuristic_bestfirst', 42,
                                      root / f'{excluded}.jsonl')
                    engine = ablation.RankingAblationEngine(session, 'How many apples?', excluded)
                    node = Node(1, Candidate(steps=[{'claim': 'apples', 'expression': '2+3'}]),
                                quality=.8, success=.6)
                    engine.features([node])
                    features = node.features
                    self.assertTrue(math.isfinite(features['heuristic']))
                    if excluded:
                        self.assertEqual(features[excluded], 0)
                    if excluded == 'quality':
                        self.assertAlmostEqual(features['diversity'], math.sqrt(.6))
                    if excluded == 'success':
                        self.assertAlmostEqual(features['depth'], 1 / cfg['max_depth'])
                        self.assertAlmostEqual(features['diversity'], math.sqrt(.8))
                self.assertEqual(base['weights'], original_weights)
                output = root / 'results' / 'synthetic'
                for method in ablation.VARIANTS:
                    append_json(output / 'results.jsonl', {
                        'question_id': 'synthetic', 'seed': 42, 'method': method,
                        'question': '2+3?', 'gold': '5', 'gold_reviewed': '5',
                        'prediction': '5', 'correct': True, 'correct_reviewed': True,
                        'calls': 1, 'input_tokens': 10, 'output_tokens': 5,
                        'total_tokens': 15, 'inference_seconds': 1,
                        'unknown_cost': False, 'stop_reason': 'done',
                    })
                summary = ablation.report(output)
                self.assertEqual(len(summary), 6)
                self.assertTrue(all(row['questions'] == 1 for row in summary))
                self.assertEqual(len((output / 'paired_comparisons.csv').read_text().splitlines()), 6)
                first = json.loads((root / 'data/gsm8k_dev_100.jsonl').read_text(
                    encoding='utf-8').splitlines()[0])
                args = SimpleNamespace(config=root / 'configs/default.json',
                                       dataset=root / 'data/gsm8k_dev_100.jsonl',
                                       label_corrections=root / 'data/label_corrections.json',
                                       output=root / 'results/mock_run', limit=1,
                                       seed=42, model='qwen2.5:7b', retry_failed=True)
                mocked_result = {'prediction': str(first['answer']), 'events': [],
                                 'candidates': [], 'stop_reason': 'mock_done'}
                with patch.object(ablation, 'doctor', return_value={
                        'model': {'digest': 'mock-model'}}), \
                        patch.object(ablation, 'request', return_value={'message': {'content': 'OK'}}), \
                        patch.object(ablation.RankingAblationEngine, 'run',
                                     return_value=mocked_result) as engine_run:
                    ablation.run(args)
                    self.assertEqual(engine_run.call_count, 6)
                    ablation.run(args)
                    self.assertEqual(engine_run.call_count, 6)
                saved = (root / 'results/mock_run/results.jsonl').read_text(
                    encoding='utf-8').splitlines()
                self.assertEqual(len(saved), 6)
                self.assertEqual({json.loads(line)['method'] for line in saved},
                                 set(ablation.VARIANTS))
                result = subprocess.run(
                    [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'],
                    cwd=root, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for item in self.notebook['cells']:
                    if item['metadata'].get('role') == 'bundled_data':
                        path = root / item['metadata']['file']
                        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                         item['metadata']['sha256'])
                run_code = next(''.join(c['source']) for c in self.notebook['cells']
                                if c['metadata'].get('role') == 'run')
                namespace = {'SPLIT': 'dev', 'LIMIT': 1, 'MODEL': 'qwen2.5:7b',
                             'SEED': 42, 'RETRY_FAILED': True, 'PROJECT_ROOT': root}
                with patch('subprocess.run') as called:
                    exec(compile(run_code, '<notebook-run>', 'exec'), namespace)
                args = called.call_args.args[0]
                self.assertIn('ablation_runner.py', ' '.join(map(str, args)))
                self.assertIn('--retry-failed', args)
                self.assertNotIn('costbench.game24', args)
            finally:
                os.chdir(before)
                sys.path.remove(str(root))
                for module in list(sys.modules):
                    if module == 'ablation_runner' or module == 'costbench' or module.startswith('costbench.'):
                        sys.modules.pop(module, None)


if __name__ == '__main__':
    unittest.main()
