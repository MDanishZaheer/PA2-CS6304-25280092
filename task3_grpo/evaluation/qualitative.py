# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Prepare Task 3 examples for the student's manual quality assessment."""

from __future__ import annotations


def review_candidates(records: list[dict], limit: int = 24) -> list[dict]:
    """Select high-reward, high-drift, and long responses without judging them."""
    groups = [
        ("high_reward", sorted(
            records, key=lambda row: row["reward_model_score"], reverse=True
        )),
        ("high_kl", sorted(
            records, key=lambda row: row["sampled_kl_token_mean"], reverse=True
        )),
        ("long_or_truncated", sorted(
            records,
            key=lambda row: (row["truncated"], row["response_length"]),
            reverse=True,
        )),
    ]
    selected = []
    used = set()
    per_group = max(1, limit // len(groups))
    for reason, candidates in groups:
        added = 0
        for row in candidates:
            if row["prompt_id"] in used:
                continue
            selected.append({
                "candidate_reason": reason,
                "prompt_id": row["prompt_id"],
                "source_index": row.get("source_index"),
                "prompt": row["prompt"],
                "response": row["response"],
                "reward_model_score": row["reward_model_score"],
                "sampled_kl_token_mean": row["sampled_kl_token_mean"],
                "entropy_token_mean": row["entropy_token_mean"],
                "response_length": row["response_length"],
                "truncated": row["truncated"],
                "manual_quality_assessment": None,
            })
            used.add(row["prompt_id"])
            added += 1
            if added >= per_group or len(selected) >= limit:
                break
    return selected[:limit]


def compare_normalization_responses(
    canonical: list[dict], dr_grpo: list[dict]
) -> list[dict]:
    """Align matched held-out responses and rank behavior differences."""
    canonical_map = {row["prompt_id"]: row for row in canonical}
    dr_map = {row["prompt_id"]: row for row in dr_grpo}
    if list(canonical_map) != list(dr_map):
        raise ValueError("Normalization comparison needs identical held-out prompts.")
    comparisons = []
    for prompt_id in canonical_map:
        left = canonical_map[prompt_id]
        right = dr_map[prompt_id]
        comparisons.append({
            "prompt_id": prompt_id,
            "source_index": left.get("source_index"),
            "prompt": left["prompt"],
            "reward_difference_dr_minus_canonical": (
                right["reward_model_score"] - left["reward_model_score"]
            ),
            "length_difference_dr_minus_canonical": (
                right["response_length"] - left["response_length"]
            ),
            "canonical": {
                "response": left["response"],
                "reward_model_score": left["reward_model_score"],
                "sampled_kl_token_mean": left["sampled_kl_token_mean"],
                "response_length": left["response_length"],
            },
            "dr_grpo": {
                "response": right["response"],
                "reward_model_score": right["reward_model_score"],
                "sampled_kl_token_mean": right["sampled_kl_token_mean"],
                "response_length": right["response_length"],
            },
            "manual_quality_comparison": None,
        })
    return sorted(
        comparisons,
        key=lambda row: (
            abs(row["reward_difference_dr_minus_canonical"]),
            abs(row["length_difference_dr_minus_canonical"]),
        ),
        reverse=True,
    )
