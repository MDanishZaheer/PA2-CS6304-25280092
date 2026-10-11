# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Validate resumable Task 5 generation and pairwise-judge records."""

from __future__ import annotations

from task5_feedback.core.paths import validate_policy_name
from task5_feedback.core.protocol import row_id
from task5_feedback.rlvr import exact_reward, extract_designated_final


def validate_response_prefix(
    records: list[dict], rows: list[dict], dataset: str, policy_name: str
) -> None:
    validate_policy_name(policy_name)
    if len(records) > len(rows):
        raise ValueError(f"{dataset}/{policy_name} has too many responses.")
    required = {
        "dataset", "prompt_id", "source_index", "policy", "question",
        "messages", "gold_final", "response", "response_tokens",
        "terminated_with_eos", "truncated", "predicted_final",
        "format_compliant", "exact_reward",
    }
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise ValueError(f"Response row {index} is missing {sorted(missing)}.")
        source = rows[index]
        if record["prompt_id"] != row_id(source, dataset):
            raise ValueError(f"{dataset}/{policy_name} response order changed.")
        if record["dataset"] != dataset or record["policy"] != policy_name:
            raise ValueError(f"Response row {index} has incorrect condition labels.")
        if record["question"] != str(source["question"]):
            raise ValueError(f"Response question changed at row {index}.")
        if record["messages"] != source["messages"]:
            raise ValueError(f"Response messages changed at row {index}.")
        if str(record["gold_final"]) != str(source["gold_final"]):
            raise ValueError(f"Gold answer changed at row {index}.")
        if int(record["response_tokens"]) < 0:
            raise ValueError("Response length cannot be negative.")
        if float(record["exact_reward"]) not in {0.0, 1.0}:
            raise ValueError("Exact reward must be binary.")
        predicted = extract_designated_final(record["response"])
        if record["predicted_final"] != predicted:
            raise ValueError(f"Stored final-answer parse changed at row {index}.")
        if bool(record["format_compliant"]) != (predicted is not None):
            raise ValueError(f"Stored format flag changed at row {index}.")
        expected_reward = exact_reward(record["response"], record["gold_final"])
        if float(record["exact_reward"]) != expected_reward:
            raise ValueError(f"Stored exact reward changed at row {index}.")


def validate_complete_responses(
    records: list[dict], rows: list[dict], dataset: str, policy_name: str
) -> None:
    validate_response_prefix(records, rows, dataset, policy_name)
    if len(records) != len(rows):
        raise ValueError(
            f"{dataset}/{policy_name} requires {len(rows)} responses; "
            f"found {len(records)}."
        )


def validate_pairwise_prefix(records: list[dict], expected: list[dict]) -> None:
    """Require pairwise cache rows to match the exact expected comparison order."""
    if len(records) > len(expected):
        raise ValueError("Pairwise cache has more rows than expected.")
    identity_fields = (
        "dataset", "prompt_id", "pair_name", "candidate_a", "candidate_b",
        "response_a_sha256", "response_b_sha256", "exact_reward_a",
        "exact_reward_b",
    )
    for index, record in enumerate(records):
        source = expected[index]
        required = {
            *identity_fields,
            "judge_preference",
            "verifier_preference",
            "verifier_judge_agree",
            "raw_judge_output",
            "judge_parse_failure",
            "judge_cache_key",
        }
        missing = required - set(record)
        if missing:
            raise ValueError(
                f"Pairwise row {index} is missing {sorted(missing)}."
            )
        for field in identity_fields:
            if record.get(field) != source.get(field):
                raise ValueError(
                    f"Pairwise cache changed {field} at comparison {index}."
                )
        if record.get("judge_preference") not in {"A", "B", "TIE"}:
            raise ValueError(f"Invalid judge preference at comparison {index}.")
        if record.get("verifier_preference") not in {"A", "B", "TIE"}:
            raise ValueError(f"Invalid verifier preference at comparison {index}.")
        agrees = record["judge_preference"] == record["verifier_preference"]
        if bool(record["verifier_judge_agree"]) != agrees:
            raise ValueError(f"Agreement flag changed at comparison {index}.")
