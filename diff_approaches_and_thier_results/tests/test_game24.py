"""Offline synthetic Game of 24 tests; no model/server calls."""
import contextlib, io, json, tempfile, unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
from costbench.__main__ import config, ROOT
SCRATCH=ROOT/"tests/_tmp"
SCRATCH.mkdir(exist_ok=True)
from costbench.backend import Session
from costbench.game24 import Term, State, GameEngine, inspect_expression, combine, prepare, report, run, METHODS, legal_children
import argparse

class GameTests(unittest.TestCase):
    def test_previously_discarded_global_solutions_are_accepted(self):
        cases=[([3,3,9,10],'(10*3)-(9-3)'),([3,5,10,13],'(13-10)*(5+3)'),([6,6,9,12],'(12/(9-6))*6')]
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            for nums,expr in cases:
                e=GameEngine(Session(config(ROOT/'configs/game24.json'),'q','heuristic_bestfirst',42,Path(tmp)/'calls'),nums)
                wrong_branch=State(1,combine(e.root.terms,0,1,'+'),1)
                valid,goal,branch=e.accept(expr,wrong_branch)
                self.assertTrue(valid and goal);self.assertFalse(branch)
                self.assertEqual(e.answer,expr)

    def test_legal_moves_are_unique_exact_and_fractional(self):
        terms=tuple(Term(Fraction(n),str(n)) for n in (3,3,8,8))
        choices=legal_children(terms)
        keys=[tuple(sorted(t.value for t in child)) for _,child in choices]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertTrue(any(Fraction(8,3) in key for key in keys))
        for _,child in choices:
            inspect_expression('+'.join(t.expression for t in child),[3,3,8,8])

    def test_endgame_uses_no_model_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp,patch('costbench.backend.request',side_effect=AssertionError('No call allowed')):
            e=GameEngine(Session(config(ROOT/'configs/game24.json'),'q','heuristic_bestfirst',42,Path(tmp)/'calls'),[3,3,8,8])
            terms=combine(e.root.terms,2,0,'/')
            terms=combine(terms,0,2,'-')
            self.assertTrue(e.endgame(State(2,terms,2)))
            self.assertTrue(inspect_expression(e.answer,[3,3,8,8])[1])

    def test_ratings_vary_and_malformed_scores_do_not_inflate_confidence(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            e=GameEngine(Session(config(ROOT/'configs/game24.json'),'q','heuristic_bestfirst',42,Path(tmp)/'calls'),[3,3,8,8])
            children=[State(i+1,t,1,quality=0,success=0) for i,(_,t) in enumerate(legal_children(e.root.terms)[:3])]
            out={'ratings':[dict(id=1,quality=.8,success=.9),dict(id=2,quality=.2,success=.1),dict(id=3,quality=float('nan'),success=1),dict(id=999,quality=1,success=1)]}
            with patch.object(e.s,'ask',return_value=(out,'mock',80)):e.rate(e.root,children)
            self.assertEqual([c.success for c in children],[.9,.1,0])
            self.assertEqual(e.rating_failures,2)

    def test_move_rollout_never_exceeds_auxiliary_call_cap(self):
        cfg=config(ROOT/'configs/game24.json');cfg['max_rollout_calls']=1
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            e=GameEngine(Session(cfg,'q','heuristic_bestfirst',42,Path(tmp)/'calls'),[3,3,8,8])
            with patch.object(e.s,'ask',return_value=({'choice':0},'mock',20)) as ask:
                e.move_rollout(e.root);e.move_rollout(e.root)
                self.assertEqual(ask.call_count,1)

    def test_deferred_states_restore_after_frontier_empties(self):
        cfg=config(ROOT/'configs/game24.json');cfg['max_frontier']=1;cfg['max_rollout_calls']=0
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            e=GameEngine(Session(cfg,'q','heuristic_bestfirst',42,Path(tmp)/'calls'),[1,1,3,8])
            bad=State(1,(Term(Fraction(2),'(1+1)'),Term(Fraction(11),'(3+8)')),2)
            good=State(2,(Term(Fraction(1),'(1*1)'),Term(Fraction(24),'(3*8)')),2)
            original=e.expand
            def expand(state):
                if state.id==0:
                    e.expansions+=1;e.expanded.append(state);e.nodes.extend([bad,good]);return [bad,good]
                return original(state)
            with patch.object(e,'expand',side_effect=expand):result=e.run('heuristic_bestfirst')
            self.assertTrue(result['correct'])
            self.assertTrue(any(x['event']=='restore_deferred' for x in result['events']))

    def test_exact_fraction_solution_and_duplicate_leaves(self):
        self.assertEqual(inspect_expression('8/(3-8/3)',[3,3,8,8]),(Fraction(24),True))
        self.assertEqual(inspect_expression('(1+2+3)*4',[1,2,3,4]),(Fraction(24),True))
        for expr in ('24','8*3','3+3+8+8+2','8/(3-8/3)+0','__import__("os")','3**3-3','-3+3+8+8'):
            with self.assertRaises((ValueError,SyntaxError,ZeroDivisionError)):inspect_expression(expr,[3,3,8,8])

    def test_moves_preserve_expression_forest_and_allow_fractions(self):
        terms=tuple(Term(Fraction(n),str(n)) for n in (3,3,8,8))
        child=combine(terms,2,0,'/')
        self.assertEqual(child[-1].value,Fraction(8,3))
        self.assertEqual(len(child),3)
        for i,j,op in ((0,0,'+'),(0,9,'+'),(True,1,'+'),(0,1,'**')):
            with self.assertRaises(ValueError):combine(terms,i,j,op)

    def test_preparation_disjoint_multisets_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            prepare(tmp,2)
            sets=[]
            for split in ('train','dev','test'):
                rows=[json.loads(l) for l in (Path(tmp)/f'game24_{split}.jsonl').read_text().splitlines()]
                sets.append({tuple(sorted(r['numbers'])) for r in rows})
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            with self.assertRaises(ValueError):prepare(tmp,2)

    @staticmethod
    def response(*args,**kwargs):
        prompt=args[2]['messages'][1]['content']
        result=dict(reasoning='synthetic',expression='8/(3-8/3)')
        if 'Candidates: ' in prompt:
            candidates=json.loads(prompt.split('Candidates: ')[1])
            preferred=[c for c in candidates if '8/3' in c['values'] or '1/3' in c['values']]
            result={'ratings':[dict(id=c['id'],quality=.95,success=.9,reason='Synthetic ranking') for c in preferred]}
        return dict(message=dict(content=json.dumps(result)),eval_count=40,prompt_eval_count=60)

    def test_all_six_methods_without_runtime_oracle(self):
        cfg=config(ROOT/'configs/game24.json')
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp,patch('costbench.backend.request',side_effect=self.response),patch('costbench.game24.solvable',side_effect=AssertionError('Oracle must never reach policy')):
            for method in METHODS:
                s=Session(cfg,'synthetic',method,42,Path(tmp)/method)
                result=GameEngine(s,[3,3,8,8]).run(method)
                self.assertTrue(result['correct'],method)
                self.assertLessEqual(s.call_count,cfg['max_calls'])
                if method in ('cot','self_consistency','verified_sampling','symbolic_search'):
                    self.assertEqual(s.call_count,6 if method=='self_consistency' else 0 if method=='symbolic_search' else 1)
                else:
                    self.assertTrue(any(r['kind']=='game24_rank' for r in s.records))
                    self.assertFalse(any(r['kind']=='game24_expression' for r in s.records))

    def test_rollout_must_preserve_fixed_subtree(self):
        cfg=config(ROOT/'configs/game24.json')
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp,patch('costbench.backend.request',side_effect=self.response):
            e=GameEngine(Session(cfg,'synthetic','heuristic_bestfirst',42,Path(tmp)/'calls'),[3,3,8,8])
            child=State(1,combine(e.root.terms,0,1,'+'),1)
            self.assertEqual(e.completion(child),(False,False))
            self.assertEqual(e.answer,"8/(3-8/3)")
            self.assertEqual(e.off_branch_solutions,1)

    def test_invalid_generations_and_rollouts_respect_budget(self):
        cfg=config(ROOT/'configs/game24.json');cfg['max_calls']=3
        bad=dict(message=dict(content='{}'),eval_count=40,prompt_eval_count=60)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp,patch('costbench.backend.request',return_value=bad):
            s=Session(cfg,'synthetic','heuristic_bestfirst',42,Path(tmp)/'calls')
            result=GameEngine(s,[3,3,8,8]).run('heuristic_bestfirst')
            self.assertLessEqual(s.call_count,3)
            if result['correct']:self.assertTrue(inspect_expression(result['prediction'],[3,3,8,8])[1])

    def test_question_major_runner_and_resume(self):
        cfg=ROOT/'configs/game24.json'
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            root=Path(tmp);data=root/'puzzles.jsonl';data.write_text(json.dumps(dict(id='synthetic',numbers=[3,3,8,8]))+'\n')
            args=argparse.Namespace(config=str(cfg),dataset=str(data),limit=1,seed=42,output=str(root/'out'),retry_failed=False)
            with patch('costbench.game24.doctor',return_value={'model':{'digest':'mock'}}),patch('costbench.backend.request',side_effect=self.response) as network,contextlib.redirect_stdout(io.StringIO()):
                run(args);n=network.call_count;run(args);self.assertEqual(network.call_count,n)
            rows=[json.loads(l) for l in (root/'out/results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),8);self.assertTrue(all(r['correct'] for r in rows))
            for name in ('summary.csv','per_question.csv','report.html','report.md'):self.assertTrue((root/'out'/name).exists())
            self.assertFalse((root/'out/run.lock').exists())

if __name__=='__main__':unittest.main()
