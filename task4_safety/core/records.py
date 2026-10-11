# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Validate resumable Task 4 response and judgment records."""

from __future__ import annotations

import math

from task4_safety.core.paths import validate_policy_name
from task4_safety.core.protocol import BEHAVIOR_LABELS


RESPONSE_FIELDS = {
    "xstest_id", "policy", "prompt", "benchmark_class", "type",
    "response", "response_tokens", "terminated_with_eos", "truncated",
}


def validate_response_prefix(
    rows: list[dict],
    xstest,
    policy_name: str,
) -> None:
    """Require saved responses to be an exact prefix of released prompt order."""
    validate_policy_name(policy_name)
    if len(rows) > len(xstest):
        raise ValueError(f"{policy_name} has more responses than XSTest prompts.")
    for position, row in enumerate(rows):
        missing = RESPONSE_FIELDS - set(row)
        if missing:
            raise ValueError(
                f"{policy_name} response row {position} is missing {sorted(missing)}."
            )
        expected = xstest.iloc[position]
        if int(row["xstest_id"]) != int(expected["xstest_id"]):
            raise ValueError(f"{policy_name} response order changed at row {position}.")
        if row["policy"] != policy_name:
            raise ValueError(f"Response row {position} has the wrong policy name.")
        if row["prompt"] != str(expected["prompt"]):
            raise ValueError(f"Prompt text changed at XSTest row {position}.")
        if row["benchmark_class"] != str(expected["benchmark_class"]):
            raise ValueError(f"Benchmark class changed at XSTest row {position}.")
        if int(row["response_tokens"]) < 0:
            raise ValueError("Response token counts cannot be negative.")
        if not isinstance(row["response"], str):
            raise ValueError("Generated responses must be strings.")


def validate_complete_responses(rows: list[dict], xstest, policy_name: str) -> None:
    """Validate a complete 450-response policy evaluation."""
    validate_response_prefix(rows, xstest, policy_name)
    if len(rows) != len(xstest):
        raise ValueError(
            f"{policy_name} requires {len(xstest)} responses; found {len(rows)}."
        )


def validate_judgment_prefix(
    judgments: list[dict],
    responses: list[dict],
    policy_name: str,
) -> None:
    """Require judgments to be an exact prefix of one response file."""
    validate_policy_name(policy_name)
    if len(judgments) > len(responses):
        raise ValueError(f"{policy_name} has more judgments than responses.")
    required = {
        "xstest_id", "policy", "prompt", "benchmark_class", "type",
        "response", "response_tokens", "ai_label", "ai_confidence",
        "ai_rationale_tag", "raw_judge_output",
    }
    for position, row in enumerate(judgments):
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"{policy_name} judgment row {position} is missing {sorted(missing)}."
            )
        source = responses[position]
        source_fields = (
            "xstest_id",
            "policy",
            "prompt",
            "benchmark_class",
            "type",
            "response",
            "response_tokens",
            "terminated_with_eos",
            "truncated",
        )
        for field in source_fields:
            if row[field] != source[field]:
                raise ValueError(
                    f"{policy_name} judgment row {position} changed {field}."
                )
        if row["ai_label"] not in BEHAVIOR_LABELS:
            raise ValueError(f"Unknown AI behavior label: {row['ai_label']!r}.")
        confidence = float(row["ai_confidence"])
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("AI judge confidence must be finite and in [0, 1].")


def validate_complete_judgments(
    judgments: list[dict],
    responses: list[dict],
    policy_name: str,
) -> None:
    """Validate one complete judgment file."""
    validate_judgment_prefix(judgments, responses, policy_name)
    if len(judgments) != len(responses):
        raise ValueError(
            f"{policy_name} requires {len(responses)} judgments; "
            f"found {len(judgments)}."
        )
