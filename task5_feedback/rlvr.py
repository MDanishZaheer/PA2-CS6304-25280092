# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Course-supplied exact final-answer verifier for Task 5."""

from __future__ import annotations

import re

_FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def extract_designated_final(text: str):
    """Extract the final answer only from the designated `#### <number>` field.

    The last designated final is used. Numbers that appear only in intermediate
    reasoning are ignored.
    """
    matches = _FINAL_RE.findall(str(text))
    if not matches:
        return None
    return matches[-1].replace(",", "")


def numerically_equal(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def exact_reward(response: str, gold_final: str) -> float:
    pred = extract_designated_final(response)
    return float(pred is not None and numerically_equal(pred, gold_final))


def format_compliance(response: str) -> float:
    """Return one only when the designated final-answer field is present."""
    return float(extract_designated_final(response) is not None)
