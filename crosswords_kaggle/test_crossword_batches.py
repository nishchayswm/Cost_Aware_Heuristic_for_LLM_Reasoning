"""Offline checks for disjoint batches, accounting, resume and safe merging."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import crossword_bench as bench
import crossword_batches as batches
from test_crossword_bench import FakeClient


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.puzzles=[dict(id=f'q{i}',clues=['clue']*10,solution='Z'*25) for i in range(10)]
        self.plan=dict(version='test',dataset_sha256=bench.digest(self.puzzles),
                       question_ids=[p['id'] for p in self.puzzles],
                       model=dict(id='mock',revision='fixed'),
                       batches=[dict(batch_id=i+1,indices=[2*i,2*i+1]) for i in range(5)])
        self.cfg={**copy.deepcopy(bench.DEFAULT),'max_tokens':30000,'max_calls':60,'max_expansions':1}
        self.runtime=dict(model='mock',revision='fixed',backend='mock',packages={},gpu='mock')

    def test_partition_covers_each_puzzle_once_and_verifies_input(self):
        selected=[p['id'] for i in range(1,6) for p in batches.batch_puzzles(self.puzzles,self.plan,i)]
        self.assertEqual(selected,[p['id'] for p in self.puzzles])
        with self.assertRaises(ValueError):batches.batch_puzzles(self.puzzles[::-1],self.plan,1)
        with self.assertRaises(ValueError):batches.batch_puzzles(self.puzzles,self.plan,6)

    def test_run_resume_merge_and_reject_incomplete_or_duplicate_batches(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp,patch('builtins.print'):
            root=Path(temp);files=[]
            for i in range(1,6):
                client=FakeClient()
                out=batches.run_batch(self.puzzles,client,self.cfg,self.runtime,root/f'batch{i}',self.plan,i)
                self.assertEqual(len(list((out/'records').glob('*.json'))),4)
                before=len(client.requests)
                batches.run_batch(self.puzzles,client,self.cfg,self.runtime,root/f'batch{i}',self.plan,i)
                self.assertEqual(len(client.requests),before)
                files.append(out/(out.name+'_results.jsonl'))
            rows=batches.merge_batches(files,root/'merged')
            self.assertEqual(len(rows),20)
            self.assertEqual(len({(r['question_id'],r['method']) for r in rows}),20)
            summary=json.loads((root/'merged/summary.json').read_text())
            self.assertTrue(all(r['n']==10 for r in summary))
            with self.assertRaisesRegex(ValueError,'all five'):batches.merge_batches(files[:-1],root/'incomplete')
            with self.assertRaisesRegex(ValueError,'Duplicate batch'):batches.merge_batches(files+files[:1],root/'duplicate')
            row=json.loads(files[-1].read_text().splitlines()[0])
            files[-1].write_text(json.dumps(row)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Incomplete run'):batches.merge_batches(files,root/'missing_record')

    def test_mismatch_and_partial_export(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp,patch('builtins.print'):
            root=Path(temp)
            out=batches.run_batch(self.puzzles,FakeClient(),self.cfg,self.runtime,root,self.plan,1)
            record=next((out/'records').glob('*.json'))
            record.unlink()
            result=batches.export_batch(out)
            self.assertEqual(len(result.read_text().splitlines()),3)
            metadata=result.with_name(out.name+'_manifest.json')
            manifest=json.loads(metadata.read_text())
            manifest['experiment']['config']['max_tokens']=999
            metadata.write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'fingerprint'):batches.merge_batches([result],root/'bad')
            with self.assertRaisesRegex(ValueError,'Model differs'):
                batches.identity(self.plan,self.cfg,{**self.runtime,'revision':'changed'},42)


if __name__ == '__main__':unittest.main()
