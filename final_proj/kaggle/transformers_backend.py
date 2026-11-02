"""In-process Transformers adapter for the existing request/accounting interface.
No HTTP server. Generation uses JSON prompting, not Ollama's constrained grammar.
"""
import hashlib, json, time

class TransformersBackend:
    def __init__(self, model, tokenizer, torch, name, revision, runtime):
        self.model=model;self.tokenizer=tokenizer;self.torch=torch;self.name=name
        self.runtime=runtime
        self.digest=hashlib.sha256(json.dumps(dict(model=name,revision=revision,runtime=runtime),sort_keys=True).encode()).hexdigest()

    def __call__(self,base,path,payload=None,timeout=15):
        if path=='/api/tags':return {'models':[dict(name=self.name,digest=self.digest,details={'quantization_level':'bitsandbytes NF4 (not GGUF)'})]}
        if path=='/api/version':return dict(version='transformers-local-adapter',runtime=self.runtime)
        if path!='/api/chat':raise ValueError('Unsupported endpoint '+path)
        torch=self.torch;opts=payload['options']
        messages=[dict(m) for m in payload['messages']]
        fmt=payload.get('format')
        instruction='\nReturn one JSON object only. No markdown fences or text outside JSON.'
        if isinstance(fmt,dict):instruction+=' Required JSON schema: '+json.dumps(fmt,separators=(',',':'))
        messages[-1]['content']+=instruction
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(self.model.device)
        n=inputs['input_ids'].shape[1];limit=int(opts['num_predict'])
        if n+limit>int(opts['num_ctx']):raise ValueError('Actual tokenizer context guard; no silent truncation')
        torch.manual_seed(int(opts['seed']));torch.cuda.manual_seed_all(int(opts['seed']))
        temperature=float(opts.get('temperature',0))
        kwargs=dict(max_new_tokens=limit,do_sample=temperature>0,repetition_penalty=float(opts.get('repeat_penalty',1.1)),pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id)
        if temperature>0:kwargs.update(temperature=temperature,top_p=1.0,top_k=0)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():output=self.model.generate(**inputs,**kwargs)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        generated=output[0,n:];text=self.tokenizer.decode(generated,skip_special_tokens=True)
        # Raw output is preserved. No answer repair or fabricated token counts.
        return dict(message={'content':text},prompt_eval_count=n,eval_count=len(generated),
                    done_reason='length' if len(generated)>=limit else 'stop',inference_seconds=elapsed)
