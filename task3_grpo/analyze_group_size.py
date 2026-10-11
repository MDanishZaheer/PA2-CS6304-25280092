# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Run the fixed-cache, equal-generation-budget GRPO group-size study."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task3_grpo.core.checkpoints import file_sha256
from task3_grpo.core.paths import group_size_directory
from task3_grpo.core.protocol import validate_shared_protocol
from task3_grpo.grpo import group_relative_advantages


INFORMATIVE_TOLERANCE = 1e-6


def load_k8_cache(path) -> dict[str, list[dict]]:
    """Load and strictly validate the instructor's eight-completion groups."""
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    required = {
        "source_index", "prompt_id", "generation_index", "completion",
        "completion_tokens", "reward",
    }
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"GRPO cache row is missing fields: {sorted(missing)}")
        if not math.isfinite(float(row["reward"])):
            raise ValueError("The GRPO cache contains a non-finite reward.")
        by_prompt[str(row["source_index"])].append(row)
    if not by_prompt:
        raise ValueError("The GRPO group-size cache is empty.")
    for prompt_key, group in by_prompt.items():
        group.sort(key=lambda item: int(item["generation_index"]))
        indices = [int(item["generation_index"]) for item in group]
        if len(group) != 8 or indices != list(range(8)):
            raise ValueError(
                f"Prompt {prompt_key} must contain exactly generation indices 0..7."
            )
        if len({item["prompt_id"] for item in group}) != 1:
            raise ValueError(f"Prompt {prompt_key} has inconsistent prompt IDs.")
    return dict(by_prompt)


def regroup_equal_generation_budget(
    by_prompt: dict[str, list[dict]], k: int
) -> list[list[dict]]:
    """Partition every eight-completion prompt cache into consecutive K groups."""
    if k not in {2, 4, 8}:
        raise ValueError("The released Task 3 study only supports K=2, 4, or 8.")
    groups = []
    for prompt_key in sorted(by_prompt, key=lambda value: int(value)):
        rows = by_prompt[prompt_key]
        if len(rows) != 8:
            raise ValueError(f"Prompt {prompt_key} does not have exactly 8 rows.")
        groups.extend(rows[start : start + k] for start in range(0, 8, k))
    expected = sum(len(rows) for rows in by_prompt.values())
    if sum(len(group) for group in groups) != expected:
        raise RuntimeError("Regrouping changed the fixed completion budget.")
    return groups


def assign_difficulty_bins(
    by_prompt: dict[str, list[dict]],
) -> tuple[dict[str, str], dict]:
    """Rank prompts by their mean reward across all eight cached completions."""
    means = {
        key: float(np.mean([float(row["reward"]) for row in rows]))
        for key, rows in by_prompt.items()
    }
    ordered = sorted(means, key=lambda key: (means[key], int(key)))
    labels = ["hard", "medium", "easy"]
    mapping = {
        key: labels[min(index * 3 // len(ordered), 2)]
        for index, key in enumerate(ordered)
    }
    counts = {
        label: sum(value == label for value in mapping.values()) for label in labels
    }
    if any(count == 0 for count in counts.values()):
        raise ValueError("Difficulty tertiles require at least one prompt in every bin.")
    ranges = {
        label: {
            "minimum_prompt_mean_reward": min(
                means[key] for key, value in mapping.items() if value == label
            ),
            "maximum_prompt_mean_reward": max(
                means[key] for key, value in mapping.items() if value == label
            ),
            "prompt_count": counts[label],
        }
        for label in labels
    }
    definition = {
        "rule": (
            "Compute each prompt's mean reward over its 8 cached completions, "
            "sort ascending with source_index as the tie-breaker, and split ranks "
            "into three near-equal bins: hard, medium, easy."
        ),
        "source": "fixed K=8 cache; identical labels reused for every K",
        "ranges": ranges,
        "prompt_assignments": [
            {
                "source_index": key,
                "prompt_id": by_prompt[key][0]["prompt_id"],
                "mean_reward_k8": means[key],
                "difficulty": mapping[key],
            }
            for key in ordered
        ],
    }
    return mapping, definition


def _group_record(group: list[dict], k: int, difficulty: str) -> dict:
    rewards = torch.tensor(
        [float(row["reward"]) for row in group], dtype=torch.float32
    )
    advantages = group_relative_advantages(
        rewards, torch.zeros(len(group), dtype=torch.long)
    )
    reward_std = float(rewards.std(unbiased=False))
    first = group[0]
    chunk = int(first["generation_index"]) // k
    return {
        "k": int(k),
        "group_id": f"{first['source_index']}:k{k}:chunk{chunk}",
        "source_index": first["source_index"],
        "prompt_id": first["prompt_id"],
        "chunk_index": chunk,
        "difficulty": difficulty,
        "completion_count": len(group),
        "reward_mean": float(rewards.mean()),
        "within_group_reward_std": reward_std,
        "informative": bool(reward_std > INFORMATIVE_TOLERANCE),
        "relative_signal_variance": float(advantages.var(unbiased=False)),
        "generation_indices": [int(row["generation_index"]) for row in group],
        "rewards": [float(row["reward"]) for row in group],
    }


def _summarize(records: list[dict], difficulty: str = "all") -> dict:
    selected = (
        records
        if difficulty == "all"
        else [row for row in records if row["difficulty"] == difficulty]
    )
    if not selected:
        raise ValueError(f"No group-size records for difficulty={difficulty!r}.")
    relative_signals = []
    for row in selected:
        relative_signals.extend(group_relative_advantages(
            torch.tensor(row["rewards"], dtype=torch.float32),
            torch.zeros(len(row["rewards"]), dtype=torch.long),
        ).tolist())
    informative_rate = float(np.mean([row["informative"] for row in selected]))
    return {
        "k": int(selected[0]["k"]),
        "difficulty": difficulty,
        "prompt_count": len({row["prompt_id"] for row in selected}),
        "group_count": len(selected),
        "completion_count": int(sum(row["completion_count"] for row in selected)),
        "informative_group_rate": informative_rate,
        "uninformative_group_rate": 1.0 - informative_rate,
        "mean_within_group_reward_std": float(np.mean([
            row["within_group_reward_std"] for row in selected
        ])),
        "variance_group_relative_signal": float(np.var(relative_signals)),
    }


def _qualitative_examples(
    records: list[dict],
    by_prompt: dict[str, list[dict]],
    prompt_lookup: dict[str, dict],
) -> list[dict]:
    """Select auditable high-spread and zero-signal cache examples."""
    examples = []
    for difficulty in ("hard", "medium", "easy"):
        candidates = [row for row in records if row["difficulty"] == difficulty]
        choices = [("largest_reward_spread", max(
            candidates, key=lambda row: row["within_group_reward_std"]
        ))]
        zero_signal = [row for row in candidates if not row["informative"]]
        if zero_signal:
            choices.append(("uninformative_equal_reward_group", zero_signal[0]))
        for reason, selected in choices:
            source_key = str(selected["source_index"])
            rows = [
                row for row in by_prompt[source_key]
                if int(row["generation_index"]) in selected["generation_indices"]
            ]
            prompt_row = prompt_lookup.get(source_key, {})
            examples.append({
                "selection_reason": reason,
                "difficulty": difficulty,
                "k": selected["k"],
                "group_id": selected["group_id"],
                "prompt_id": selected["prompt_id"],
                "messages": prompt_row.get("messages"),
                "prompt": prompt_row.get("prompt"),
                "within_group_reward_std": selected["within_group_reward_std"],
                "informative": selected["informative"],
                "completions": [
                    {
                        "generation_index": int(row["generation_index"]),
                        "reward": float(row["reward"]),
                        "completion_tokens": int(row["completion_tokens"]),
                        "completion": row["completion"],
                    }
                    for row in rows
                ],
                "manual_observation": None,
            })
    return examples


def run_group_size_study(config_path: str = "configs/grpo.yaml") -> dict:
    """Produce all numerical and qualitative evidence for Task 3 Part 2."""
    config = load_yaml(config_path)
    validate_shared_protocol(config)
    by_prompt = load_k8_cache(config["group_cache"])
    # The supplied cache was sampled from the fixed held-out prompt pool.
    prompt_rows = read_jsonl(config["paths"]["rl_prompt_eval"])
    prompt_lookup = {
        str(row["source_index"]): row
        for row in prompt_rows
        if "source_index" in row
    }
    missing_prompts = sorted(set(by_prompt).difference(prompt_lookup))
    if missing_prompts:
        raise ValueError(
            f"Cached source indices are missing from the training prompt pool: {missing_prompts}"
        )
    mismatched_ids = [
        key for key, rows in by_prompt.items()
        if prompt_lookup[key].get("prompt_id") != rows[0]["prompt_id"]
    ]
    if mismatched_ids:
        raise ValueError(
            f"Cache and held-out prompt IDs disagree at source indices: {mismatched_ids}"
        )
    difficulty, definition = assign_difficulty_bins(by_prompt)
    result_dir = group_size_directory(config)
    result_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    summaries = []
    difficulty_summaries = []
    completion_budget = sum(len(rows) for rows in by_prompt.values())
    for k in [int(value) for value in config["group_sizes"]]:
        records = [
            _group_record(group, k, difficulty[str(group[0]["source_index"])])
            for group in regroup_equal_generation_budget(by_prompt, k)
        ]
        if sum(row["completion_count"] for row in records) != completion_budget:
            raise RuntimeError("A K condition changed the fixed generation budget.")
        all_records.extend(records)
        summaries.append(_summarize(records))
        difficulty_summaries.extend(
            _summarize(records, label) for label in ("hard", "medium", "easy")
        )
    write_jsonl(result_dir / "per_group_metrics.jsonl", all_records)
    pd.DataFrame(summaries).to_csv(result_dir / "summary.csv", index=False)
    pd.DataFrame(difficulty_summaries).to_csv(
        result_dir / "difficulty_summary.csv", index=False
    )
    save_json(result_dir / "difficulty_bins.json", definition)
    write_jsonl(
        result_dir / "qualitative_examples.jsonl",
        _qualitative_examples(
            [row for row in all_records if row["k"] == 8],
            by_prompt,
            prompt_lookup,
        ),
    )
    protocol = {
        "cache": config["group_cache"],
        "cache_sha256": file_sha256(config["group_cache"]),
        "prompt_pool": config["paths"]["rl_prompt_eval"],
        "prompt_pool_sha256": file_sha256(config["paths"]["rl_prompt_eval"]),
        "cached_prompts": len(by_prompt),
        "cached_completions": completion_budget,
        "group_sizes": [int(value) for value in config["group_sizes"]],
        "regrouping": "consecutive generation_index chunks within each prompt",
        "equal_completion_budget_per_k": completion_budget,
        "informative_tolerance": INFORMATIVE_TOLERANCE,
        "relative_signal": "within-group population-std standardized reward",
        "difficulty_rule_file": (
            result_dir.relative_to(repo_path(".")).as_posix()
            + "/difficulty_bins.json"
        ),
    }
    save_json(result_dir / "protocol.json", protocol)
    from task3_grpo.analysis.plots import save_available_figures

    save_available_figures(config)
    return {
        "summary": summaries,
        "difficulty_summary": difficulty_summaries,
        "protocol": protocol,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    args = parser.parse_args()
    result = run_group_size_study(args.config)
    print(pd.DataFrame(result["summary"]).to_string(index=False))


if __name__ == "__main__":
    main()
