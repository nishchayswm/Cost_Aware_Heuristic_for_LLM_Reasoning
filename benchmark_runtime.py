"""Ollama transport for engine v3. Caching is disabled for measured runs."""
def ollama_backend(prompt, schema, temperature, seed, max_tokens):
    import ollama
    from llm import Model
    kwargs = {'model': Model, 'messages': [{'role': 'user', 'content': prompt}],
              'options': {'temperature': temperature, 'seed': seed, 'num_predict': max_tokens}}
    if schema is not None:
        kwargs['format'] = schema
    response = ollama.chat(**kwargs)
    return {'text': response['message']['content'],
            'prompt_tokens': response.get('prompt_eval_count', 0),
            'completion_tokens': response.get('eval_count', 0),
            'done_reason': response.get('done_reason', 'unknown')}
