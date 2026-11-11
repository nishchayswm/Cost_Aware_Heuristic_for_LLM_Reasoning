"""The same constrained reasoning format is used by all methods."""
import copy
import re
STEP = {"type":"object","properties":{
    "claim":{"type":"string","maxLength":120},
    "expression":{"type":"string","maxLength":180,
                  "pattern":r"^[0-9s+*/(). \t-]+$"}},
    "required":["claim","expression"],"additionalProperties":False}


def obj(properties):
    return {"type":"object","properties":properties,"required":list(properties),"additionalProperties":False}


def response_schema(kind, prompt):
    if kind=='game24_rank':
        score={'type':'number','minimum':0,'maximum':1}
        return obj({'ratings':{'type':'array','maxItems':12,'items':obj({
            'id':{'type':'integer','minimum':0},'reason':{'type':'string'},'quality':score,'success':score})}})
    if kind=='game24_choice':
        return obj({'reason':{'type':'string'},'choice':{'type':'integer','minimum':0}})
    if kind=='game24_expression':
        return obj({'reasoning':{'type':'string'},'expression':{'type':'string','maxLength':500}})
    if kind=='game24_moves':
        move=obj({'i':{'type':'integer','minimum':0,'maximum':3},
                  'j':{'type':'integer','minimum':0,'maximum':3},
                  'op':{'type':'string','enum':['+','-','*','/']}})
        return obj({'reasoning':{'type':'string'},'moves':{'type':'array','items':move,'maxItems':8}})
    if kind == "judge":
        score={"type":"number","minimum":0,"maximum":1}
        check={"type":"string","enum":["pass","fail","uncertain","pending"]}
        return obj({"evaluations":{"type":"array","items":obj({
            "coverage":check,"interpretation":check,"units":check,"completion":check,
            "quality":score,"success":score})}})
    step=copy.deepcopy(STEP)
    if "NUMERIC EXPRESSIONS ONLY." in prompt:
        step["properties"]["expression"]["pattern"]=r"^[0-9+*/(). \t-]+$"
    elif kind=="extend":
        label=re.search(r"Python will label it s(\d+)\.",prompt)
        if label:
            previous=[f"s{i}" for i in range(1,int(label.group(1)))]
            atoms=[r"[0-9+*/(). \t-]"]+previous
            step["properties"]["expression"]["pattern"]="^(?:"+"|".join(atoms)+")+$"
    return obj({"steps":{"type":"array","items":step,"minItems":1,"maxItems":1 if kind=="extend" else 10},
                "complete":{"type":"boolean"}})
