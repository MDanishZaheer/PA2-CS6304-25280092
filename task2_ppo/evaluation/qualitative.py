# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Prepare PPO responses for human reward-versus-quality inspection."""

from __future__ import annotations


def review_candidates(records: list[dict], limit: int = 24) -> list[dict]:
    """Select high-reward, high-drift, and long responses without judging quality."""
    if limit <= 0:
        raise ValueError("Qualitative candidate limit must be positive.")
    ranked_groups = [
        ("high_reward", sorted(records, key=lambda row: row["reward_model_score"], reverse=True)),
        ("high_kl", sorted(records, key=lambda row: row["sampled_kl_token_mean"], reverse=True)),
        ("long_or_truncated", sorted(records, key=lambda row: (row["truncated"], row["response_length"]), reverse=True)),
    ]
    selected = []
    used = set()
    per_group = max(1, limit // len(ranked_groups))
    for reason, rows in ranked_groups:
        added = 0
        for row in rows:
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


def compare_condition_responses(
    records_by_run: dict[str, list[dict]], control_run: str
) -> list[dict]:
    """Align held-out responses across controlled forks for manual comparison."""
    if control_run not in records_by_run:
        raise KeyError(f"Missing control run: {control_run}")
    maps = {
        name: {row["prompt_id"]: row for row in records}
        for name, records in records_by_run.items()
    }
    prompt_ids = list(maps[control_run])
    if any(set(run_map) != set(prompt_ids) for run_map in maps.values()):
        raise ValueError("Qualitative comparison requires identical held-out prompt IDs.")
    comparisons = []
    for prompt_id in prompt_ids:
        control = maps[control_run][prompt_id]
        conditions = {}
        for run_name, run_map in maps.items():
            row = run_map[prompt_id]
            conditions[run_name] = {
                "response": row["response"],
                "reward_model_score": row["reward_model_score"],
                "sampled_kl_token_mean": row["sampled_kl_token_mean"],
                "entropy_token_mean": row["entropy_token_mean"],
                "response_length": row["response_length"],
            }
        reward_range = max(
            item["reward_model_score"] for item in conditions.values()
        ) - min(item["reward_model_score"] for item in conditions.values())
        comparisons.append({
            "prompt_id": prompt_id,
            "source_index": control.get("source_index"),
            "prompt": control["prompt"],
            "control_run": control_run,
            "reward_score_range": reward_range,
            "conditions": conditions,
            "manual_reward_quality_relationship": None,
        })
    return sorted(comparisons, key=lambda row: row["reward_score_range"], reverse=True)
