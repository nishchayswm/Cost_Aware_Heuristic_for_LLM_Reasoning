import json, ollama, hashlib, os

Model = "qwen2.5:7b"
stats = {"calls": 0, "tokens": 0, "hits":0}

def llm_call(prompt, maxx = 500, temp = 0.7):
    stats["calls"] += 1
    key = hashlib.md5((Model + prompt + str(temp)).encode()).hexdigest()

    path = f"cache/{key}.json"

    if (os.path.exists(path)):
        with open(path) as f:
            saved = json.load(f)
        stats["hits"] += 1
        stats["tokens"] += saved["tokens"]
        return saved["text"]

    r = ollama.chat(model = Model, messages = [{"role": "user", "content": prompt}], options = {"temperature": temp, "num_predict": maxx})

    text = r["message"]["content"]
    tokens= r.get("prompt_eval_count", 0) + r.get("eval_count", 0)
    stats["tokens"] += tokens

    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w") as f:
        json.dump({"text" : text, "tokens": tokens}, f)

    return text