import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import crossword_bench as b

class FakeClient:
    def __init__(self):self.requests=[]
    def count(self,prompt):return 20
    def generate(self,prompt,seed,limit,temperature,schema=None):
        self.requests.append(prompt)
        if prompt.startswith('Propose'):
            value={'moves':[{'slot':'h1','word':'APPLE','confidence':.9}, {'slot':'h2','word':'PEARS','confidence':.8}]}
        else:value={'ratings':[{'id':0,'q':.8,'p':.7},{'id':1,'q':.7,'p':.8}]}
        return dict(text=json.dumps(value),input_tokens=20,output_tokens=10,seconds=.1,truncated=False)

class Tests(unittest.TestCase):
    def test_revision_and_patterns(self):
        words=b.place(('',)*10,0,'APPLE')
        self.assertEqual(b.board_of(words)[:5],'APPLE')
        words=b.place(words,5,'BREAD')
        self.assertEqual(words[0],'')
        self.assertEqual(''.join(b.board_of(words)[i] for i in b.cells(5)),'BREAD')
        self.assertIsNone(b.place(words,0,'LONGER'))
        self.assertIsNone(b.place(words,5,'BREAD'))

    def test_score(self):
        truth='ABCDEFGHIJKLMNOPQRSTUVWXY'
        self.assertTrue(b.score(truth,truth)['solved'])
        self.assertEqual(b.score('_'+truth[1:],truth)['word_accuracy'],.8)
        self.assertEqual(b.score('_'*25,truth)['letter_accuracy'],0)

    def test_budget_replay_and_search(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            cfg={**b.DEFAULT,'max_expansions':3}
            for method in b.METHODS:
                client=FakeClient();root=Path(tmp)/method
                session=b.Session(client,cfg,root,'q',42)
                result,_=b.solve(['clue']*10,method,session,cfg)
                self.assertGreater(result['calls'],0)
                self.assertLessEqual(result['total_tokens'],cfg['max_tokens'])
                replay=b.Session(client,cfg,root,'q',42);before=len(client.requests)
                result2,_=b.solve(['clue']*10,method,replay,cfg)
                self.assertEqual(result,result2)
                self.assertEqual(len(client.requests),before)
            cfg={**cfg,'max_tokens':25}
            client=FakeClient();session=b.Session(client,cfg,Path(tmp)/'budget','q',42)
            with self.assertRaises(b.BudgetStop):session.ask('hello','generate',10)
            self.assertEqual(client.requests,[])

    def test_malformed_json(self):
        class Bad(FakeClient):
            def generate(self,*a,**kwargs):return dict(text='not json',input_tokens=20,output_tokens=5,seconds=.1,truncated=False)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            session=b.Session(Bad(),b.DEFAULT,tmp,'q',42)
            with self.assertRaises(b.InvalidResponse):session.ask('hello','generate',10)
            self.assertEqual(session.parse_failures,3)
            self.assertEqual(session.calls,3)
            self.assertEqual(session.tokens,75)

    def test_dataset_pinning_and_no_answer_input(self):
        records=[]
        for i in range(60):
            board=list('A'*25);board[0]=chr(65+i//26);board[1]=chr(65+i%26)
            records.append([['clue']*10,board])
        def fetch(url):
            if '/commits/' in url:return json.dumps({'sha':'a'*40}).encode()
            if url.endswith('LICENSE'):return b'license'
            return json.dumps(records).encode()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp,patch.object(b,'fetch',side_effect=fetch):
            puzzles=b.prepare_dataset(tmp)
            self.assertEqual(len(puzzles),50)
            self.assertEqual(len({p['id'] for p in puzzles}),50)
            self.assertEqual(puzzles,b.prepare_dataset(tmp))
            self.assertNotIn('solution',json.loads((Path(tmp)/'questions50.json').read_text())[0])
            with self.assertRaises(ValueError):b.prepare_dataset(tmp,99)

    def test_runner_resume_and_paired_reports(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            puzzles=[dict(id=f'q{i}',clues=['clue']*10,solution='Z'*25) for i in range(10)]
            cfg={**b.DEFAULT,'max_expansions':1}
            client=FakeClient()
            # No real inference occurs.
            with patch('builtins.print'):
                summary=b.run(puzzles,client,cfg,{},tmp)
                count=len(client.requests)
                b.run(puzzles,client,cfg,{},tmp)
            self.assertEqual(count,len(client.requests))
            self.assertTrue(all(s['n']==10 for s in summary))
            self.assertTrue(all('Z'*25 not in p for p in client.requests))
            self.assertEqual(len(list((Path(tmp)/'records').glob('*.json'))),20)
            manifest=json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertEqual(manifest['puzzle_count'],10)
            self.assertEqual(manifest['question_ids'],[p['id'] for p in puzzles])
            self.assertTrue((Path(tmp)/'per_question.csv').exists())
            with self.assertRaises(ValueError):b.run(puzzles,client,{**cfg,'max_tokens':999},{},tmp)
            with self.assertRaises(ValueError):b.run(puzzles[:5],client,cfg,{},tmp)

    def test_heuristic_features(self):
        s=b.State(words=('APPLE',)+('',)*9,depth=1,q=.8,p=.7)
        f=b.features(s,[s,b.State()],b.DEFAULT,1000)
        self.assertEqual(set(f),set(b.DEFAULT['weights']))
        self.assertTrue(all(0<=v<=1 for v in f.values()))
        self.assertTrue(b.priority(s,[s],b.DEFAULT,1000)>b.priority(b.State(),[s],b.DEFAULT,1000))

    def test_interrupted_call_cannot_hide_cost(self):
        class Broken(FakeClient):
            def generate(self,*a,**kwargs):raise RuntimeError('backend failure')
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            session=b.Session(Broken(),b.DEFAULT,tmp,'q',42)
            with self.assertRaisesRegex(RuntimeError,'backend failure'):session.ask('hello','generate',10)
            session=b.Session(FakeClient(),b.DEFAULT,tmp,'q',42)
            with self.assertRaisesRegex(RuntimeError,'cost is unknown'):session.ask('hello','generate',10)

    def test_short_words_and_trailing_markdown_are_retried_and_charged(self):
        short={'moves':[{'slot':'h1','word':'HOLD','confidence':.8}]}
        good={'moves':[{'slot':'h1','word':'APPLE','confidence':.8}]}
        class Recovering(FakeClient):
            def generate(self,prompt,seed,limit,temperature,schema=None):
                self.requests.append((prompt,seed,schema))
                text=[json.dumps(good)+'```',json.dumps(short),json.dumps(good)][len(self.requests)-1]
                return dict(text=text,input_tokens=20,output_tokens=10,seconds=.1,truncated=False)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            client=Recovering();s=b.Session(client,b.DEFAULT,tmp,'q',42)
            self.assertEqual(s.ask('propose','generate',350),good)
            self.assertEqual((s.calls,s.tokens,s.parse_failures),(3,90,2))
            self.assertEqual(len({r[1] for r in client.requests}),3)
            before=len(client.requests)
            replay=b.Session(client,b.DEFAULT,tmp,'q',42)
            self.assertEqual(replay.ask('propose','generate',350),good)
            self.assertEqual(replay.tokens,90)
            self.assertEqual(len(client.requests),before)

    def test_retry_respects_budget_and_reports_failure(self):
        class Bad(FakeClient):
            def generate(self,*args,**kwargs):
                return dict(text='{}',input_tokens=20,output_tokens=5,seconds=.1,truncated=False)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            cfg={**b.DEFAULT,'max_calls':1}
            s=b.Session(Bad(),cfg,tmp,'q',42)
            result,_=b.solve(['clue']*10,'tot_dfs',s,cfg)
            self.assertEqual(result['stop_reason'],'token_or_call_budget')
            self.assertEqual((result['calls'],result['total_tokens']),(1,25))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            s=b.Session(Bad(),b.DEFAULT,tmp,'q',42)
            result,trace=b.solve(['clue']*10,'tot_dfs',s,b.DEFAULT)
            self.assertEqual(result['stop_reason'],'invalid_model_response')
            self.assertEqual(result['calls'],3)
            self.assertIn('error',trace[-1])

    def test_decoding_constraint_rejects_short_words_and_markdown(self):
        from lmformatenforcer import JsonSchemaParser
        def accepts(text):
            parser=JsonSchemaParser(b.response_schema('generate'))
            for ch in text:
                if ch not in parser.get_allowed_characters():return False
                parser=parser.add_character(ch)
            return parser.can_end()
        good='{"moves":[{"slot":"h1","word":"APPLE","confidence":0.8}]}'
        self.assertTrue(accepts(good))
        for invalid in (good+'```',good+'\nuser\nCorrect JSON',good.replace('APPLE','HOLD'),
                        good.replace('APPLE','BAH'),good.replace('h1','h6')):
            self.assertFalse(accepts(invalid),invalid)

if __name__=='__main__':unittest.main()

