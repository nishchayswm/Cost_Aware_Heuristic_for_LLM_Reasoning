import ast, operator

Ops = {
    ast.Add : operator.add,
    ast.Sub : operator.sub,
    ast.Mult : operator.mul,
    ast.Div : operator.truediv,
    ast.FloorDiv : operator.floordiv,
    ast.Mod : operator.mod,
    ast.Pow : operator.pow,
    ast.USub : operator.neg,
}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in Ops:
        return Ops[type(node.op)](_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp) and type(node.op) in Ops:
        return Ops[type(node.op)](_eval(node.operand))
    else:
        raise ValueError("_eval function failed!")

def safe_eval(expr):
    from benchmark_engine import safe_arithmetic
    return safe_arithmetic(expr)

def check_equation(step):
    if "=" not in step:
        return None
    left, right = step.split("=", 1)
    a, b = safe_eval(left), safe_eval(right)
    if a is None or b is None:
        return None
    return abs(a - b) < 1e-6
