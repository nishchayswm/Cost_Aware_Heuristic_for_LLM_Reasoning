"""Offline integrity and execution checks for the bundled teammate notebook."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_dhruv_branch_notebook as builder


class DhruvNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(builder.OUTPUT.read_text(encoding='utf-8'))

    def test_bundle_matches_fetched_commit(self):
        revision = builder.git('rev-parse', builder.REF).decode().strip()
        self.assertEqual(self.notebook['metadata']['source_commit'], revision)
        files = [c for c in self.notebook['cells']
                 if c['metadata'].get('role') in ('source_file', 'bundled_data')]
        self.assertEqual(len(files), len(builder.bundled_paths()))
        for item in files:
            relative = item['metadata']['file']
            original = builder.git('show', f'{revision}:final_proj/{relative}')
            self.assertEqual(item['metadata']['sha256'], hashlib.sha256(original).hexdigest())
            if item['metadata']['role'] == 'source_file':
                source = ''.join(item['source'])
                self.assertEqual(source, f'%%writefile {relative}\n' + original.decode())

    def test_restored_project_passes_teammate_offline_suite(self):
        with tempfile.TemporaryDirectory(dir=builder.ROOT / 'kaggle') as folder:
            root = Path(folder)
            before = Path.cwd()
            try:
                os.chdir(root)
                for item in self.notebook['cells']:
                    role = item['metadata'].get('role')
                    source = ''.join(item['source'])
                    if role == 'source_file':
                        relative = item['metadata']['file']
                        path = root / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(source.split('\n', 1)[1], encoding='utf-8')
                    elif role == 'bundled_data':
                        exec(compile(source, item['id'], 'exec'), {})
                    elif role in ('settings', 'offline_tests', 'ollama_setup', 'run', 'download'):
                        compile(source, item['id'], 'exec')
                result = subprocess.run(
                    [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'],
                    cwd=root, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for item in self.notebook['cells']:
                    if item['metadata'].get('role') == 'bundled_data':
                        path = root / item['metadata']['file']
                        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                         item['metadata']['sha256'])
                for task in ('gsm8k', 'game24'):
                    namespace = {'TASK': task, 'SPLIT': 'dev', 'LIMIT': 1,
                                 'MODEL': 'qwen2.5:7b', 'SEED': 42,
                                 'RETRY_FAILED': True,
                                 'PROJECT_ROOT': root}
                    code = next(''.join(c['source']) for c in self.notebook['cells']
                                if c['metadata'].get('role') == 'run')
                    with patch('subprocess.run') as called:
                        exec(compile(code, '<run>', 'exec'), namespace)
                    args = called.call_args.args[0]
                    self.assertIn('costbench.game24' if task == 'game24' else 'costbench', args)
                    self.assertIn('--limit', args)
                    self.assertIn('--retry-failed', args)
                    configured = json.loads((root / f'configs/kaggle_{task}.json').read_text())
                    self.assertEqual(configured['model'], 'qwen2.5:7b')
            finally:
                os.chdir(before)


if __name__ == '__main__':
    unittest.main()
