"""Restricted arithmetic and dependency validation. Never executes generated Python."""
from __future__ import annotations

import ast
import copy
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction


def number(value) -> str | None:
    """Strict numeric answer normalization; no last-number guessing."""
    text = str(value).strip().replace(",", "").replace("−", "-")
    if len(text) > 100:
        return None
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", text):
        return None
    try:
        v = Fraction(text)
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    except (ValueError, ZeroDivisionError):
        return None


def arithmetic(expression: str, values: dict[str, Fraction]) -> Fraction:
    if not isinstance(expression, str) or len(expression) > 300:
        raise ValueError("Expression too long")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 70:
        raise ValueError("Expression too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = Fraction(str(node.value))
        elif isinstance(node, ast.Name) and node.id in values:
            result = values[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): result = a + b
            elif isinstance(node.op, ast.Sub): result = a - b
            elif isinstance(node.op, ast.Mult): result = a * b
            elif isinstance(node.op, ast.Pow):
                if b.denominator != 1 or abs(b.numerator)>32:
                    raise ValueError("Exponent must be an integer between -32 and 32")
                result = a ** b.numerator
            else: result = a / b
        else:
            raise ValueError("Only numbers, prior step IDs, + - * / and parentheses are allowed")
        if abs(result.numerator) > 10**30 or result.denominator > 10**30:
            raise ValueError("Arithmetic magnitude limit")
        return result

    return visit(tree.body)


def references(expression):
    return {n.id for n in ast.walk(ast.parse(expression, mode="eval")) if isinstance(n, ast.Name)}


@dataclass
class Candidate:
    steps: list[dict] = field(default_factory=list)
    answer: str | None = None
    final_step: str = ""
    complete: bool = False
    issues: list[str] = field(default_factory=list)
    raw: str = ""
    tokens: int = 0
    origin: int = 0
    values: dict[str, Fraction] = field(default_factory=dict)
    corrections: int = 0
    metadata_repairs: list[str] = field(default_factory=list)

    def view(self, partial=False):
        steps = self.steps
        if partial and len(steps) > 3:
            steps = steps[:1] + steps[-2:]
        return {"steps": steps, "answer": self.answer, "final_step": self.final_step,
                "complete": self.complete, "arithmetic_issues": self.issues}

    def signature(self):
        return json.dumps([(s.get("claim"), s.get("expression"), s.get("source"),
                            s.get("depends_on")) for s in self.steps], sort_keys=True)


def evaluate(payload, raw="", tokens=0, origin=0) -> Candidate:
    c = Candidate(raw=raw, tokens=tokens, origin=origin)
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        c.issues = ["invalid_schema"]
        return c
    if len(payload["steps"]) > 16:
        c.issues = ["too_many_steps"]
        return c
    if not isinstance(payload.get("complete"), bool):
        c.issues.append("complete_must_be_boolean")
    c.complete = payload.get("complete") is True
    for index, original in enumerate(payload["steps"], 1):
        if not isinstance(original, dict):
            c.issues.append("invalid_step")
            continue
        step = copy.deepcopy(original)
        # IDs are positional bookkeeping, not a reasoning task for the LLM.
        # Expression references are NEVER guessed/remapped: only prior canonical IDs exist.
        sid = f"s{index}"
        if "id" in step and step["id"] != sid:
            c.metadata_repairs.append(f"{sid}:assigned_canonical_id")
        step["id"] = sid
        expression = step.get("expression", "")
        try:
            if not isinstance(step.get("claim"),str) or not step["claim"].strip():
                raise ValueError("missing_quantity_description")
            if not isinstance(expression,str) or len(expression)>300:
                raise ValueError("invalid_expression")
            # Accept only a redundant assignment to THIS canonical step. Never
            # guess aliases, rename references, or execute assignment statements.
            assignment=re.fullmatch(r"\s*"+re.escape(sid)+r"\s*(?::=|=)\s*([^=;\n]+)\s*",expression)
            if assignment:
                step["original_expression"]=expression
                expression=assignment.group(1).strip()
                step["expression"]=expression
                c.metadata_repairs.append(f"{sid}:removed_redundant_assignment")
            refs = references(expression)
            deps = sorted(refs)
            if "depends_on" in step and step["depends_on"] != deps:
                c.metadata_repairs.append(f"{sid}:dependencies_derived_from_expression")
            step["depends_on"] = deps
            if not refs.issubset(c.values):
                raise ValueError("missing_or_forward_dependency")
            value = arithmetic(expression, c.values)
            c.values[sid] = value
            supplied = number(step.get("value"))
            corrected = "value" in step and supplied != str(value)
            if corrected:
                c.corrections += 1
            step["value"] = str(value)
            step["arithmetic_checked"] = True
            step["arithmetic_corrected"] = corrected
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
            c.issues.append(f"{sid}:{exc}")
            step["arithmetic_checked"] = False
        c.steps.append(step)
    # New protocol: a completed solution's last step MUST be the requested quantity.
    c.final_step = c.steps[-1]["id"] if c.complete and c.steps else ""
    # Identical local arithmetic service for all methods.
    # It establishes expression correctness only, never semantic correctness.
    if c.complete and c.final_step in c.values and not c.issues:
        c.answer = str(c.values[c.final_step])
    # No model-answer fallback for an invalid graph. Unknown symbols remain invalid.
    return c


def vote(candidates):
    answers = [c.answer for c in candidates if c.complete and c.answer is not None]
    if not answers: return None
    counts = Counter(answers)
    # Deterministic first-generated tie break, fixed before evaluation.
    return max(answers, key=lambda a: counts[a])
