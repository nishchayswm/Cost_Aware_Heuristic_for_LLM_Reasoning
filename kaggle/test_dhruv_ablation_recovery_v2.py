"""Offline transport-failure and replay checks for the recovery notebook."""
import hashlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_dhruv_ablation_v2 as builder


class RecoveryTests(unittest.TestCase):
    def test_bounded_token_repeat_recovery_and_exact_replay(self):
        notebook = json.loads(builder.OUTPUT.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory(dir=builder.ROOT / 'kaggle') as folder:
            root = Path(folder)
            before = Path.cwd()
            sys.path.insert(0, str(root))
            try:
                os.chdir(root)
                for item in notebook['cells']:
                    role = item['metadata'].get('role')
                    source = ''.join(item['source'])
                    if role in ('source_file', 'ablation_runner'):
                        path = root / item['metadata']['file']
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(source.split('\n', 1)[1].encode('utf-8'))
                    elif role == 'bundled_data':
                        exec(compile(source, item['id'], 'exec'), {})
                    elif role in ('settings', 'offline_tests', 'ollama_setup', 'run', 'download'):
                        compile(source, item['id'], 'exec')
                from costbench.__main__ import config as load_config
                from costbench.backend import BackendError
                ablation = importlib.import_module('ablation_runner')
                self.assertEqual(ablation.MAX_ATTEMPTS_PER_CALL, 4)
                cfg = load_config(root / 'configs/default.json')
                good = {'message': {'content': '{"steps":[],"complete":false}'},
                        'prompt_eval_count': 7, 'eval_count': 3, 'done_reason': 'stop'}
                log = root / 'calls' / 'one.jsonl'
                session = ablation.RecoverableSession(cfg, 'q1', 'heuristic_bestfirst',
                                                      42, log, retry_failed=True)
                with patch.object(ablation, 'request', side_effect=[
                        BackendError('HTTP 500: token repeat limit reached'), good]) as call:
                    parsed, _, _ = session.ask('A small question', 'generate', 260)
                self.assertEqual(call.call_count, 2)
                self.assertFalse(parsed['complete'])
                attempts = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(len(attempts), 2)
                self.assertIn('error', attempts[0])
                self.assertEqual(attempts[1]['attempt'], 1)
                self.assertNotEqual(attempts[0]['request']['options']['seed'],
                                    attempts[1]['request']['options']['seed'])
                self.assertEqual(attempts[1]['request']['options']['num_predict'], 192)
                self.assertTrue(session.metrics()['unknown_cost'])
                replay = ablation.RecoverableSession(cfg, 'q1', 'heuristic_bestfirst',
                                                     42, log, retry_failed=True)
                with patch.object(ablation, 'request', side_effect=AssertionError('replayed call')):
                    self.assertEqual(replay.ask('A small question', 'generate', 260)[0], parsed)
                self.assertTrue(replay.metrics()['unknown_cost'])
                repeated_log = root / 'calls' / 'always_fails.jsonl'
                with patch.object(ablation, 'request',
                                  side_effect=BackendError('token repeat limit reached')) as calls:
                    for _ in range(2):
                        retry = ablation.RecoverableSession(cfg, 'q2', 'heuristic_bestfirst',
                                                            42, repeated_log, retry_failed=True)
                        with self.assertRaises(BackendError):
                            retry.ask('Another question', 'generate', 260)
                    self.assertEqual(calls.call_count, 4)
                    retry = ablation.RecoverableSession(cfg, 'q2', 'heuristic_bestfirst',
                                                        42, repeated_log, retry_failed=True)
                    with self.assertRaisesRegex(BackendError, 'recovery exhausted'):
                        retry.ask('Another question', 'generate', 260)
                    self.assertEqual(calls.call_count, 4)
                self.assertTrue(all('error' in json.loads(line)
                                    for line in repeated_log.read_text().splitlines()))
                old_archive = builder.ROOT / 'kaggle' / (
                    'gsm8k_dev_n50_seed42_qwen2_5_7b_five_factor_ablations.zip')
                if old_archive.exists():
                    cfg['model'] = 'qwen2.5:7b'
                    cfg['base_url'] = 'http://127.0.0.1:11434'
                    config_path = root / 'configs/migration_qwen.json'
                    config_path.write_text(json.dumps(cfg), encoding='utf-8')
                    migrated = root / 'results/migrated'
                    source = ablation.migrate_saved_results(
                        old_archive, migrated, root / 'data/gsm8k_dev_100.jsonl',
                        config_path, root / 'data/label_corrections.json', 42, 50)
                    self.assertEqual(source['imported_method_rows'], 216)
                    self.assertEqual(len((migrated / 'results.jsonl').read_text().splitlines()), 216)
                    self.assertEqual(ablation.migrate_saved_results(
                        old_archive, migrated, root / 'data/gsm8k_dev_100.jsonl',
                        config_path, root / 'data/label_corrections.json', 42, 50), source)
                    question = next(json.loads(line) for line in
                                    (root / 'data/gsm8k_dev_100.jsonl').read_text().splitlines()
                                    if json.loads(line)['id'] == 'gsm8k-train-6630')
                    log = migrated / 'calls/gsm8k-train-6630_42_heuristic_bestfirst.jsonl'
                    replay_session = ablation.RecoverableSession(
                        cfg, question['id'], 'heuristic_bestfirst', 42, log,
                        retry_failed=True)
                    engine = ablation.RankingAblationEngine(
                        replay_session, question['question'])
                    with patch.object(ablation, 'request',
                                      side_effect=BackendError('stop after replay')) as call:
                        with self.assertRaisesRegex(BackendError, 'stop after replay'):
                            engine.run('heuristic_bestfirst')
                    self.assertEqual(call.call_count, 1)
                    self.assertEqual(replay_session.call_count, 6)
                    old_seed = json.loads(log.read_text().splitlines()[-2])[
                        'request']['options']['seed']
                    self.assertNotEqual(call.call_args.args[2]['options']['seed'], old_seed)
                source = next(''.join(c['source']).split('\n', 1)[1] for c in notebook['cells']
                              if c['metadata'].get('role') == 'ablation_runner')
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                                 hashlib.sha256((builder.ROOT / 'kaggle' /
                                                 'dhruv_ablation_runner_v2.py').read_bytes()).hexdigest())
            finally:
                os.chdir(before)
                sys.path.remove(str(root))
                for module in list(sys.modules):
                    if module == 'ablation_runner' or module == 'costbench' or module.startswith('costbench.'):
                        sys.modules.pop(module, None)


if __name__ == '__main__':
    unittest.main()
