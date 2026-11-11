"""Synthetic fixtures only. Never reads GSM8K or contacts a model/server."""
import argparse
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from costbench.__main__ import config, run
from costbench.backend import Session, BudgetExceeded, BackendError, append_json
from costbench.graph import Candidate, arithmetic, evaluate, number
from costbench.methods import Engine, METHODS, Node, diversity_scores
from costbench.report import build_report
from costbench.labels import score
from costbench.schema import response_schema
import re

ROOT=Path(__file__).resolve().parent.parent
SCRATCH=ROOT/"tests/_tmp"


def temporary():
    SCRATCH.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=SCRATCH)


def fixture(depth=2):
    steps=[{"id":"s1","claim":"initial items","expression":"3*4","depends_on":[],"source":"three groups of four","value":"12"},
           {"id":"s2","claim":"total items","expression":"s1+2","depends_on":["s1"],"source":"two more","value":"14"}]
    return {"steps":steps[:depth],"complete":depth==2,"final_step":"s2" if depth==2 else "","answer":"14" if depth==2 else ""}


def fake_response(base,path,payload=None,timeout=15):
    text=str(payload)
    if "Evaluate the following candidate" in text:
        result={"evaluations":[{"quality":.8,"success":.7,**dict.fromkeys(("coverage","interpretation","units","completion"),"pass")} for _ in range(4)]}
    elif "Python will label it s2" in text:
        result=fixture();result["steps"]=result["steps"][1:]
    elif "exactly ONE new step" in text:
        result=fixture(1)
    else:result=fixture()
    for step in result.get("steps",[]):
        step["expression"]=step["expression"].replace("s1+2","12+2")
    return {"message":{"content":json.dumps(result)},"eval_count":80,"prompt_eval_count":120,"done_reason":"stop"}


class OfflineTests(unittest.TestCase):
    def setUp(self):self.cfg=config(ROOT/"configs/default.json")

    def test_bounded_integer_powers(self):
        self.assertEqual(str(arithmetic("16*(1.25)**2",{})),"25")
        self.assertEqual(str(arithmetic("2**(-3)",{})),"1/8")
        for expression in ("2**33","2**0.5","0**(-1)"):
            with self.assertRaises((ValueError,ZeroDivisionError)):arithmetic(expression,{})

    def test_numeric_schema_excludes_variable_references(self):
        schema=response_schema("generate","NUMERIC EXPRESSIONS ONLY.")
        pattern=schema["properties"]["steps"]["items"]["properties"]["expression"]["pattern"]
        self.assertIsNotNone(re.fullmatch(pattern,"20-3*2"))
        self.assertIsNone(re.fullmatch(pattern,"s1-6"))

    def test_search_initial_solution_is_recorded_and_charged(self):
        with temporary() as tmp,patch("costbench.backend.request",side_effect=fake_response):
            session=Session(self.cfg,"q","heuristic_bestfirst",42,Path(tmp)/"calls")
            result=Engine(session,"Synthetic problem").run("heuristic_bestfirst")
            self.assertTrue(any(e["event"]=="initial_solution" for e in result["events"]))
            self.assertEqual(session.records[0]["kind"],"generate")
            self.assertEqual(session.records[1]["kind"],"judge")
            self.assertEqual(session.output_tokens,80*session.call_count)

    def test_judge_uses_json_mode_but_generation_keeps_schema(self):
        with temporary() as tmp,patch("costbench.backend.request",side_effect=fake_response) as network:
            session=Session(self.cfg,"q","heuristic_bestfirst",42,Path(tmp)/"calls")
            session.ask("synthetic","judge",100)
            self.assertEqual(network.call_args.args[2]["format"],"json")
            session.ask("synthetic","generate",100)
            self.assertIsInstance(network.call_args.args[2]["format"],dict)

    def test_progress_aware_depth_and_judge_cost(self):
        with temporary() as tmp:
            engine=Engine(Session(self.cfg,"q","heuristic_bestfirst",42,Path(tmp)/"calls"),"synthetic")
            n=Node(0,evaluate(fixture(1)),quality=.8,success=.9)
            engine.features([n])
            self.assertAlmostEqual(n.features["depth"],.1/self.cfg["max_depth"])
            self.assertGreater(n.features["estimated_judge_share"],0)

    def test_sound_premature_completion_reopens_for_all_search_methods(self):
        response={"message":{"content":json.dumps({"evaluations":[dict(quality=.9,success=.8,coverage="fail",completion="fail",interpretation="pass",units="pass") for _ in range(2)]})},"eval_count":30,"prompt_eval_count":40}
        with temporary() as tmp,patch("costbench.backend.request",return_value=response):
            e=Engine(Session(self.cfg,"q","heuristic_bestfirst",42,Path(tmp)/"calls"),"synthetic")
            c=fixture(1);c["complete"]=True
            child=Node(0,evaluate(c));e.completed=[child]
            with patch.object(e,"generate",side_effect=[child,None]):
                frontier=e.expand(Node(-1,Candidate()))
            self.assertEqual(frontier,[child])
            self.assertEqual(e.completed,[])
            self.assertIsNone(child.candidate.answer)
            self.assertAlmostEqual(child.quality,.45)
            self.assertTrue(any(x["event"]=="reopen_incomplete_answer" for x in e.events))

    def test_redundant_assignment_normalization_is_narrow(self):
        for expression in ("s1 = 7*3","s1 := 7*3","  s1=7*3  "):
            c=evaluate({"steps":[{"claim":"pencils","expression":expression}],"complete":True})
            self.assertEqual(c.answer,"21")
            self.assertEqual(c.steps[0]["original_expression"],expression)
            self.assertTrue(c.metadata_repairs)
        for expression in ("s2=7*3","a1=7*3","s1=unknown","s1=7;3","s1=s1+3","s1=1/0","s1=__import__('os')"):
            c=evaluate({"steps":[{"claim":"pencils","expression":expression}],"complete":True})
            self.assertIsNone(c.answer)
            self.assertTrue(c.issues)

    def test_expression_schema_limits_root_and_prior_references(self):
        root=response_schema("extend","Python will label it s1.")["properties"]["steps"]["items"]["properties"]["expression"]["pattern"]
        self.assertIsNotNone(re.fullmatch(root,"7*3"))
        for expression in ("s1+3","s1=7*3","total/3","7;3"):
            self.assertIsNone(re.fullmatch(root,expression))
        child=response_schema("extend","Python will label it s2.")["properties"]["steps"]["items"]["properties"]["expression"]["pattern"]
        self.assertIsNotNone(re.fullmatch(child,"s1/3"))
        self.assertIsNone(re.fullmatch(child,"s2/3"))

    def test_gated_diversity_does_not_reward_zero_quality(self):
        with temporary() as tmp:
            e=Engine(Session(self.cfg,"q","heuristic_bestfirst",42,Path(tmp)/"calls"),"synthetic")
            node=Node(0,evaluate(fixture(1)),quality=0,success=.9)
            e.features([node])
            self.assertEqual(node.features["raw_diversity"],1)
            self.assertEqual(node.features["diversity"],0)
            node.quality=.4;node.success=.9
            e.features([node]);self.assertAlmostEqual(node.features["diversity"],.6)
            e.cfg=copy.deepcopy(e.cfg);e.cfg["quality_gated_diversity"]=False
            e.features([node]);self.assertEqual(node.features["diversity"],1)

    def test_repeated_model_id_and_wrong_dependencies_are_metadata_only(self):
        payload=fixture()
        payload["steps"][1].update(id="s1",depends_on=["s8"])
        result=evaluate(payload)
        self.assertEqual(result.answer,"14")
        self.assertEqual(result.steps[1]["id"],"s2")
        self.assertEqual(result.steps[1]["depends_on"],["s1"])
        self.assertEqual(len(result.metadata_repairs),2)

    def test_new_minimal_protocol_and_unsafe_references(self):
        payload={"steps":[{"claim":"items","expression":"3*4"},
                          {"claim":"total items","expression":"s1+2"}],"complete":True}
        self.assertEqual(evaluate(payload).answer,"14")
        for expression in ("a1+2","s2+2","s99+2","s1/0","s1**999"):
            payload["steps"][1]["expression"]=expression
            self.assertIsNone(evaluate(payload).answer)
            self.assertTrue(evaluate(payload).issues)

    def test_invalid_generation_retries_are_charged_for_every_policy(self):
        for method in METHODS:
            attempts=[0]
            def response(*args,**kwargs):
                attempts[0]+=1
                if attempts[0]==1:
                    return {"message":{"content":"{malformed"},"eval_count":25,"prompt_eval_count":30}
                return fake_response(*args,**kwargs)
            with temporary() as tmp,patch("costbench.backend.request",side_effect=response):
                session=Session(self.cfg,"synthetic",method,42,Path(tmp)/"calls")
                result=Engine(session,"Three groups of four and two extra items.").run(method)
                self.assertEqual(result["prediction"],"14",method)
                self.assertEqual(result["generation_retries"],1,method)
                self.assertEqual(session.output_tokens,25+80*(session.call_count-1))
                self.assertLessEqual(session.call_count,self.cfg["max_calls"])

    def test_wrong_but_structurally_valid_answer_does_not_trigger_retry(self):
        with temporary() as tmp,patch("costbench.backend.request",side_effect=fake_response):
            session=Session(self.cfg,"synthetic","cot",42,Path(tmp)/"calls")
            result=Engine(session,"There is one item. How many items?").run("cot")
            self.assertEqual(result["prediction"],"14")
            self.assertEqual(result["generation_retries"],0)
            self.assertEqual(session.call_count,1)

    def test_retry_respects_one_call_budget(self):
        cfg=copy.deepcopy(self.cfg);cfg["max_calls"]=1
        response={"message":{"content":"{}"},"eval_count":25,"prompt_eval_count":30}
        with temporary() as tmp,patch("costbench.backend.request",return_value=response) as network:
            result=Engine(Session(cfg,"synthetic","cot",42,Path(tmp)/"calls"),"synthetic").run("cot")
            self.assertIsNone(result["prediction"])
            self.assertEqual(network.call_count,1)
            self.assertEqual(result["stop_reason"],"common_budget_exhausted")

    def test_reviewed_labels_and_reports_preserve_original(self):
        question={"id":"synthetic","question":"Three groups of four and two extra.","answer":"15"}
        corrections={"synthetic":dict(id="synthetic",question=question["question"],original="15",reviewed="14",reason="Synthetic arithmetic audit")}
        result=score(question,"14",corrections)
        self.assertFalse(result["correct_original"])
        self.assertTrue(result["correct_reviewed"])
        with self.assertRaises(ValueError):score({**question,"answer":"16"},"14",corrections)
        with self.assertRaises(ValueError):score({**question,"question":"different"},"14",corrections)
        with temporary() as tmp:
            root=Path(tmp)
            for method in METHODS:
                append_json(root/"results.jsonl",dict(question_id="synthetic",question=question["question"],seed=42,method=method,prediction="14",status="ok",total_tokens=10,**result))
            summary=build_report(root)
            self.assertTrue(all(r["accuracy"]==0 and r["accuracy_reviewed"]==1 for r in summary))
            self.assertIn("Reviewed accuracy",(root/"report.html").read_text())
            self.assertIn("reviewed_accuracy_difference",(root/"paired_comparisons.csv").read_text())

    def test_arithmetic_rejects_executable_and_explosive_input(self):
        for expression in ["__import__('os').system('bad')","2**99999999","[1][0]","x.attr"]:
            with self.assertRaises(ValueError):arithmetic(expression,{})

    def test_numeric_normalization(self):
        self.assertEqual(number("1,000.0"),"1000")
        self.assertEqual(number(".5"),number("1/2"))
        self.assertIsNone(number("answer might be 12 or 14"))

    def test_arithmetic_correction_and_dependencies(self):
        p=fixture();p["steps"][1]["value"]="99"
        self.assertEqual(evaluate(p).answer,"14")
        p["steps"][1]["depends_on"]=[]
        self.assertFalse(evaluate(p).issues)
        self.assertEqual(evaluate(p).answer,"14")
        self.assertTrue(evaluate(p).metadata_repairs)

    def test_diversity_changes_with_explored_history(self):
        a=Node(0,evaluate(fixture(1)))
        self.assertEqual(diversity_scores([a],[]),[1.0])
        self.assertAlmostEqual(diversity_scores([a],[a])[0],0)

    def test_quality_and_full_heuristic_can_choose_different_states(self):
        with temporary() as tmp:
            s=Session(self.cfg,"fixture","heuristic_bestfirst",42,Path(tmp)/"calls")
            e=Engine(s,"Synthetic question")
            a=Node(0,evaluate(fixture(1)),quality=.9,success=.1)
            b=Node(1,evaluate(fixture(1)),quality=.8,success=.95)
            self.assertEqual(e.rank([a,b],"quality_bestfirst")[0].id,0)
            self.assertEqual(e.rank([a,b],"heuristic_bestfirst")[0].id,1)
            self.assertEqual(set(a.features)&set(self.cfg["weights"]),set(self.cfg["weights"]))

    def test_all_policies_synthetic_and_budgeted(self):
        with temporary() as tmp,patch("costbench.backend.request",side_effect=fake_response):
            for method in METHODS:
                s=Session(self.cfg,"fixture",method,42,Path(tmp)/f"{method}.jsonl")
                result=Engine(s,"There are three groups of four items and two more. How many items?").run(method)
                self.assertEqual(result["prediction"],"14",method)
                self.assertLessEqual(s.call_count,self.cfg["max_calls"])
                self.assertLessEqual(s.output_tokens,self.cfg["max_output_tokens"])
                if method in ("quality_bestfirst","heuristic_bestfirst"):
                    self.assertTrue(any(x["event"]=="expand" for x in result["events"]))

    def test_invalid_single_step_is_not_inserted(self):
        with temporary() as tmp,patch("costbench.backend.request",return_value={"message":{"content":json.dumps(fixture())},"eval_count":80,"prompt_eval_count":20}):
            e=Engine(Session(self.cfg,"q","cot",42,Path(tmp)/"calls"),"synthetic")
            self.assertIsNone(e.generate(one_step=True))
            self.assertEqual(e.invalid,2)
            self.assertEqual(e.generation_retries,1)

    def test_missing_judge_scores_are_flagged(self):
        with temporary() as tmp,patch("costbench.backend.request",return_value={"message":{"content":'{"evaluations":null}'},"eval_count":20,"prompt_eval_count":20}):
            e=Engine(Session(self.cfg,"q","quality_bestfirst",42,Path(tmp)/"calls"),"synthetic")
            node=Node(0,evaluate(fixture(1)))
            e.evaluate_nodes([node])
            self.assertFalse(node.evaluated)
            self.assertEqual(e.evaluation_failures,1)

    def test_replay_preserves_cost_and_avoids_extra_calls(self):
        with temporary() as tmp,patch("costbench.backend.request",side_effect=fake_response) as network:
            path=Path(tmp)/"calls"
            a=Session(self.cfg,"q","cot",42,path);a.ask("synthetic",max_tokens=100)
            b=Session(self.cfg,"q","cot",42,path);b.ask("synthetic",max_tokens=100)
            self.assertEqual(network.call_count,1)
            self.assertEqual(a.metrics(),b.metrics())

    def test_common_call_limit_is_enforced_before_network(self):
        cfg=copy.deepcopy(self.cfg);cfg["max_calls"]=1
        with temporary() as tmp,patch("costbench.backend.request",side_effect=fake_response) as network:
            s=Session(cfg,"q","cot",42,Path(tmp)/"calls")
            s.ask("test",max_tokens=100)
            with self.assertRaises(BudgetExceeded):s.ask("test2",max_tokens=100)
            self.assertEqual(network.call_count,1)

    def test_failed_request_retry_is_explicit_and_cost_flagged(self):
        with temporary() as tmp:
            path=Path(tmp)/"calls"
            with patch("costbench.backend.request",side_effect=RuntimeError("backend failed")):
                with self.assertRaises(BackendError):Session(self.cfg,"q","cot",42,path).ask("test",max_tokens=100)
            with patch("costbench.backend.request",side_effect=fake_response) as network:
                with self.assertRaises(BackendError):Session(self.cfg,"q","cot",42,path).ask("test",max_tokens=100)
                self.assertEqual(network.call_count,0)
                s=Session(self.cfg,"q","cot",42,path,retry_failed=True);s.ask("test",max_tokens=100)
                self.assertTrue(s.metrics()["unknown_cost"])
                again=Session(self.cfg,"q","cot",42,path);again.ask("test",max_tokens=100)
                self.assertEqual(network.call_count,1)

    def test_config_rejects_missing_or_invalid_weights(self):
        with temporary() as tmp:
            p=Path(tmp)/"config.json";cfg=copy.deepcopy(self.cfg);cfg["weights"]["depth"]=-1
            p.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):config(p)

    def test_question_major_runner_gold_isolation_and_resume(self):
        prompts=[]
        def wrapped(*args,**kwargs):
            prompts.append(str(args)+str(kwargs));return fake_response(*args,**kwargs)
        with temporary() as tmp:
            root=Path(tmp);dataset=root/"synthetic.jsonl"
            dataset.write_text("\n".join(json.dumps({"id":f"synthetic-{i}","question":f"Three groups of four and two extra. Case {i}.","answer":"987654321"}) for i in range(2)))
            args=argparse.Namespace(config=str(ROOT/"configs/default.json"),model=None,seed=None,dataset=str(dataset),limit=2,output=str(root/"out"),retry_failed=False)
            correction_file=root/"corrections.json"
            corrections=[dict(id="synthetic-0",question="Three groups of four and two extra. Case 0.",original="987654321",reviewed="123456789",reason="Synthetic isolation sentinel")]
            correction_file.write_text(json.dumps(corrections))
            args.label_corrections=str(correction_file)
            output=io.StringIO()
            with patch("costbench.__main__.doctor",return_value={"model":{"digest":"SYNTHETIC"}}),patch("costbench.__main__.request",side_effect=wrapped),patch("costbench.backend.request",side_effect=wrapped),contextlib.redirect_stdout(output):
                run(args)
                n=len(prompts);run(args)
                self.assertEqual(len(prompts),n)
                corrections[0]["reviewed"]="123456788"
                correction_file.write_text(json.dumps(corrections))
                with self.assertRaisesRegex(ValueError,"NEW output directory"):run(args)
                self.assertEqual(len(prompts),n)
            self.assertFalse(any("987654321" in p for p in prompts))
            self.assertFalse(any("123456789" in p for p in prompts))
            rows=[json.loads(l) for l in (root/"out/results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows),12)
            self.assertEqual([r["question_id"] for r in rows],["synthetic-0"]*6+["synthetic-1"]*6)
            self.assertIn("quality_bestfirst",output.getvalue())
            self.assertIn("Ground truth",output.getvalue())
            self.assertTrue((root/"out/report.html").exists())
            self.assertTrue((root/"out/per_question.csv").exists())
            self.assertFalse((root/"out/run.lock").exists())

    def test_reports_use_equal_question_coverage(self):
        with temporary() as tmp:
            root=Path(tmp)
            for method in METHODS:
                append_json(root/"results.jsonl",{"question_id":"a","seed":42,"method":method,"question":"synthetic","gold":"14","prediction":"14","correct":True,"status":"ok","calls":1,"total_tokens":10})
            append_json(root/"results.jsonl",{"question_id":"b","seed":42,"method":"cot","question":"synthetic","gold":"14","prediction":"0","correct":False,"status":"ok"})
            summary=build_report(root)
            self.assertTrue(all(r["n"]==1 and r["accuracy"]==1 for r in summary))


if __name__=="__main__":unittest.main()
