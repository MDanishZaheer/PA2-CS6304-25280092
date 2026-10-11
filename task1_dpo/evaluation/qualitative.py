# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Select response pairs for human inspection of metric disagreements."""

from __future__ import annotations


def review_candidates(records: list[dict], limit: int = 20) -> list[dict]:
    """Rank positive reward gains without assigning a human quality label."""
    if limit <= 0:
        raise ValueError("limit must be positive.")
    candidates = []
    for record in records:
        if "policy_reward_score" not in record:
            continue
        reward_change = record["policy_reward_score"] - record["reference_reward_score"]
        length_change = record["response_length"] - record["reference_length"]
        candidates.append({
            "prompt_id": record["prompt_id"],
            "source_index": record.get("source_index"),
            "prompt": record["prompt"],
            "policy_response": record["policy_response"],
            "reference_response": record["reference_response"],
            "policy_reward_score": record["policy_reward_score"],
            "reference_reward_score": record["reference_reward_score"],
            "reward_change": reward_change,
            "length_change": length_change,
            "manual_quality_assessment": None,
        })
    return sorted(
        candidates,
        key=lambda row: (row["reward_change"], abs(row["length_change"])),
        reverse=True,
    )[:limit]
