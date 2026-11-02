"""Recreate the fixed GSM8K splits. Run from any working directory."""
import hashlib
import json
import random
import urllib.request
from pathlib import Path

root=Path(__file__).resolve().parent.parent/"data"
root.mkdir(exist_ok=True)
manifest={}
for split,n in [("test",100),("train",100)]:
    url=f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
    raw=urllib.request.urlopen(url,timeout=60).read()
    rows=[json.loads(line) for line in raw.splitlines()]
    indices=random.Random(42).sample(range(len(rows)),n)
    selected=[{"id":f"gsm8k-{split}-{i:04d}","source_index":i,"split":split,
               "question":rows[i]["question"],"answer":rows[i]["answer"].split("####")[-1].strip()} for i in indices]
    name="gsm8k_test_100.jsonl" if split=="test" else "gsm8k_dev_100.jsonl"
    (root/name).write_text("".join(json.dumps(x)+"\n" for x in selected),encoding="utf-8")
    manifest[split]={"url":url,"sha256":hashlib.sha256(raw).hexdigest(),"selected_indices":indices,
                     "seed":42,"source_count":len(rows)}
(root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print("Saved 100 test and 100 separate training/development questions.")
