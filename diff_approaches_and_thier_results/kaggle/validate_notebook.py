"""Offline notebook validation; no network, model download or inference."""
import ast
import contextlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent

class NotebookTests(unittest.TestCase):
    def test_cells_and_dataset(self):
        nb = json.loads((ROOT/'kaggle/Game24_Original_ToT_Kaggle.ipynb').read_text(encoding='utf-8'))
        cells = [c['source'] for c in nb['cells'] if c['cell_type']=='code']
        for source in cells:
            compile(source, '<notebook>', 'exec')
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory(dir=ROOT/'tests/_tmp') as tmp:
                project = Path(tmp)
                ns = {'sys':__import__('sys')}
                extract = next(s for s in cells if 'ARCHIVE_B64 =' in s)
                extract = extract.replace("Path('/kaggle/working/cost_game24')", repr(str(project)))
                extract = extract.replace('PROJECT = '+repr(str(project)), 'PROJECT = Path('+repr(str(project))+')')
                exec(extract, ns)
                os.chdir(previous)
                self.assertTrue((project/'costbench/game24.py').exists())
                rows=list(itertools.combinations_with_replacement(range(1,14),4))[:1000]
                csv=('Puzzles\n'+'\n'.join(' '.join(map(str,r)) for r in rows)).encode()
                def fake_bytes(url):
                    if '/commits/' in url:return json.dumps({'sha':'a'*40}).encode()
                    if url.endswith('/LICENSE'):return b'test license'
                    return csv
                source = next(s for s in cells if 'REQUESTED_TOT_COMMIT =' in s)
                tree=ast.parse(source)
                tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef)]
                ns['get_bytes']=fake_bytes
                exec(compile(tree,'<dataset>','exec'),ns)
                records=[json.loads(s) for s in ns['ORIGINAL_DATASET'].read_text().splitlines()]
                self.assertEqual(len(records),100)
                self.assertEqual(records[0]['numbers'],list(rows[900]))
                exec(compile(tree,'<cached-dataset>','exec'),ns)
        finally:
            os.chdir(previous)

    def test_adapter(self):
        spec=importlib.util.spec_from_file_location('adapter',ROOT/'kaggle/transformers_backend.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class Tensor:
            shape=(1,3)
        class Inputs(dict):
            def to(self,device):return self
        class Tokenizer:
            pad_token_id=0;eos_token_id=2
            def apply_chat_template(self,messages,**kw):return messages[-1]['content']
            def __call__(self,*a,**kw):return Inputs(input_ids=Tensor())
            def decode(self,*a,**kw):return '{"expression":"6*4"}'
        class Output:
            def __getitem__(self,key):return [8,9,2]
        class Model:
            device='cuda:0'
            def generate(self,**kw):return Output()
        class Torch:
            def manual_seed(self,seed):pass
            def manual_seed_all(self,seed):pass
            def synchronize(self):pass
            def inference_mode(self):return contextlib.nullcontext()
        torch=Torch();torch.cuda=torch
        adapter=module.TransformersBackend(Model(),Tokenizer(),torch,'test','revision',{})
        payload={'messages':[{'role':'user','content':'test'}],'format':'json',
                 'options':{'seed':42,'num_ctx':100,'num_predict':10}}
        result=adapter('', '/api/chat',payload)
        self.assertEqual(result['prompt_eval_count'],3)
        self.assertEqual(result['eval_count'],3)
        self.assertEqual(payload['messages'][0]['content'],'test')
        self.assertEqual(adapter('','/api/tags')['models'][0]['name'],'test')
        payload['options']['num_ctx']=5
        with self.assertRaises(ValueError):adapter('','/api/chat',payload)

if __name__=='__main__':
    (ROOT/'tests/_tmp').mkdir(exist_ok=True)
    unittest.main()
