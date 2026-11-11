"""Evaluator-only reviewed labels. Never passed to generation or search."""
import json
from pathlib import Path
from .graph import number


def load_corrections(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Label corrections must be a list")
    result = {}
    for r in records:
        if r["id"] in result or not r["reason"] or not r["question"]:
            raise ValueError("Invalid or duplicate label correction")
        if any(number(r[k]) is None for k in ("original", "reviewed")):
            raise ValueError("Invalid reviewed label")
        result[r["id"]] = r
    return result


def score(question, prediction, corrections):
    original = number(question["answer"])
    reviewed = original
    correction = corrections.get(question["id"])
    if correction:
        if correction["question"] != question["question"] or number(correction["original"]) != original:
            raise ValueError("Label correction does not match the exact question and original answer")
        reviewed = number(correction["reviewed"])
    predicted = number(prediction) if prediction is not None else None
    return {"gold": original, "gold_reviewed": reviewed,
            "correct": predicted is not None and predicted == original,
            "correct_original": predicted is not None and predicted == original,
            "correct_reviewed": predicted is not None and predicted == reviewed,
            "label_correction_applied": bool(correction),
            "label_correction_reason": correction["reason"] if correction else ""}
