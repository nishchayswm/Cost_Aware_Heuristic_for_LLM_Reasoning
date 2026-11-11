import csv
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run as runner

ROOT=Path(__file__).resolve().parent

def source_csv():
    stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['Rank','Puzzles'])
    numbers=list(itertools.combinations_with_replacement(range(1,14),4))[:1000]
    for i,ns in enumerate(numbers):writer.writerow([i+1,' '.join(map(str,ns))])
    return stream.getvalue().encode(),numbers

def fake_fetch(url):
    if '/commits/' in url:return json.dumps({'sha':'a'*40}).encode()
    if url.endswith('/LICENSE'):return b'test license'
    return source_csv()[0]

class Tests(unittest.TestCase):
    def test_official_indices_and_cached_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp,patch.object(runner,'fetch',side_effect=fake_fetch) as fetch:
            dataset,puzzles,meta=runner.prepare(tmp)
            self.assertEqual(len(puzzles),100)
            self.assertEqual(puzzles[0]['id'],'tot24-0900')
            self.assertEqual(puzzles[0]['numbers'],list(source_csv()[1][900]))
            self.assertEqual(len(dataset.read_text().splitlines()),100)
            count=fetch.call_count
            self.assertEqual(runner.prepare(tmp)[1],puzzles)
            self.assertEqual(fetch.call_count,count)
            with self.assertRaises(ValueError):runner.prepare(tmp,'b'*40)
            (Path(tmp)/'24.csv').write_bytes(b'changed')
            with self.assertRaises(ValueError):runner.prepare(tmp)

    def test_local_dispatch_without_inference(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp,patch.object(runner,'fetch',side_effect=fake_fetch),patch.object(runner.game24,'run') as engine:
            output=Path(tmp)/'output';data=Path(tmp)/'data'
            args=['run','--directory',str(data),'--output',str(output),'--limit','15','--seed','43']
            self.assertEqual(runner.main(args),0)
            self.assertEqual(engine.call_args.args[0].limit,15)
            self.assertEqual(engine.call_args.args[0].seed,43)
            cfg=json.loads((output/'run_config.json').read_text())
            self.assertEqual(cfg['model'],'qwen2.5:7b-instruct')
            self.assertIn('local_runner_sha256',cfg)
            self.assertEqual(runner.main(args),0)
            self.assertEqual(runner.main(args+['--model','other']),1)
            self.assertEqual(engine.call_count,2)

    def test_empty_disjoint_report_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp)
            rows=[dict(question_id='q',method=m,correct=False,total_tokens=5,calls=1) for m in runner.game24.METHODS]
            (root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (root/'overlap_audit.csv').write_text('id,overlap_generated_train,overlap_generated_dev,overlap_generated_test\nq,True,False,False\n')
            with patch.object(runner.game24,'report'):runner.report(root)
            with (root/'disjoint_subset_summary.csv').open() as f:summary=list(csv.DictReader(f))
            self.assertEqual(len(summary),8)
            self.assertTrue(all(r['n']=='0' and r['accuracy']=='' for r in summary))

if __name__=='__main__':unittest.main()
