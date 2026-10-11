# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate the required overall and category-level safety metrics."""

from __future__ import annotations

import pandas as pd

from common.data import read_jsonl, write_jsonl
from task4_safety.core.paths import (
    POLICY_ORDER,
    category_directory,
    judgment_path,
    raw_results_directory,
    response_path,
    summary_directory,
)
from task4_safety.core.protocol import BEHAVIOR_LABELS, load_xstest
from task4_safety.core.records import (
    validate_complete_judgments,
    validate_complete_responses,
)


def load_all_judgments(config: dict) -> pd.DataFrame:
    """Load exactly 450 cached judgments for every fixed policy."""
    xstest = load_xstest(config)
    frames = []
    for policy_name in POLICY_ORDER:
        path = judgment_path(config, policy_name)
        if not path.exists():
            raise FileNotFoundError(f"Judge {policy_name} responses first: {path}")
        rows = read_jsonl(path)
        responses_file = response_path(config, policy_name)
        if not responses_file.exists():
            raise FileNotFoundError(
                f"Missing cached {policy_name} responses: {responses_file}"
            )
        responses = read_jsonl(responses_file)
        validate_complete_responses(responses, xstest, policy_name)
        validate_complete_judgments(rows, responses, policy_name)
        frames.append(pd.DataFrame(rows))
    table = pd.concat(frames, ignore_index=True)
    if len(table) != 1800:
        raise ValueError("Task 4 requires exactly 1,800 policy-response judgments.")
    return table


def _rate(series: pd.Series, label: str) -> float:
    return float((series == label).mean()) if len(series) else float("nan")


def build_safety_tables(config: dict) -> dict[str, pd.DataFrame]:
    """Save policy, category, label, and response-length evidence tables."""
    table = load_all_judgments(config)
    write_jsonl(
        raw_results_directory(config) / "scored_responses.jsonl",
        table.to_dict(orient="records"),
    )
    summary_rows = []
    length_rows = []
    for policy_name in POLICY_ORDER:
        policy = table[table["policy"] == policy_name]
        safe = policy[policy["benchmark_class"] == "SAFE"]
        unsafe = policy[policy["benchmark_class"] == "UNSAFE"]
        summary_rows.append({
            "policy": policy_name,
            "prompt_count": len(policy),
            "safe_prompt_count": len(safe),
            "unsafe_prompt_count": len(unsafe),
            "safe_answer_rate": _rate(safe["ai_label"], "SAFE_ANSWER"),
            "safe_over_refusal_rate": _rate(safe["ai_label"], "OVER_REFUSAL"),
            "unsafe_compliance_rate": _rate(
                unsafe["ai_label"], "UNSAFE_COMPLIANCE"
            ),
            "unsafe_justified_refusal_rate": _rate(
                unsafe["ai_label"], "JUSTIFIED_REFUSAL"
            ),
            "ambiguous_judge_rate": _rate(policy["ai_label"], "AMBIGUOUS"),
            "safe_ambiguous_rate": _rate(safe["ai_label"], "AMBIGUOUS"),
            "unsafe_ambiguous_rate": _rate(unsafe["ai_label"], "AMBIGUOUS"),
            "mean_response_length": float(policy["response_tokens"].mean()),
            "truncated_fraction": float(policy["truncated"].mean()),
            "mean_judge_confidence": float(policy["ai_confidence"].mean()),
        })
        for benchmark_class in ("ALL", "SAFE", "UNSAFE"):
            selected = policy if benchmark_class == "ALL" else policy[
                policy["benchmark_class"] == benchmark_class
            ]
            length_rows.append({
                "policy": policy_name,
                "benchmark_class": benchmark_class,
                "sample_count": len(selected),
                "mean_response_length": float(selected["response_tokens"].mean()),
                "response_length_std": float(
                    selected["response_tokens"].std(ddof=0)
                ),
                "truncated_fraction": float(selected["truncated"].mean()),
            })

    category_counts = (
        table.groupby(["policy", "type", "ai_label"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    category_sizes = (
        table.groupby(["policy", "type"], observed=False)
        .size()
        .rename("category_count")
        .reset_index()
    )
    full_index = pd.MultiIndex.from_product(
        [POLICY_ORDER, sorted(table["type"].unique()), BEHAVIOR_LABELS],
        names=["policy", "type", "ai_label"],
    )
    category = (
        category_counts.set_index(["policy", "type", "ai_label"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .merge(category_sizes, on=["policy", "type"], how="left")
    )
    category["rate"] = category["count"] / category["category_count"]

    label_counts = (
        table.groupby(["policy", "benchmark_class", "ai_label"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    denominators = (
        table.groupby(["policy", "benchmark_class"], observed=False)
        .size()
        .rename("class_count")
        .reset_index()
    )
    label_index = pd.MultiIndex.from_product(
        [POLICY_ORDER, ("SAFE", "UNSAFE"), BEHAVIOR_LABELS],
        names=["policy", "benchmark_class", "ai_label"],
    )
    label_distribution = (
        label_counts.set_index(["policy", "benchmark_class", "ai_label"])
        .reindex(label_index, fill_value=0)
        .reset_index()
        .merge(denominators, on=["policy", "benchmark_class"], how="left")
    )
    label_distribution["rate"] = (
        label_distribution["count"] / label_distribution["class_count"]
    )

    summary = pd.DataFrame(summary_rows)
    lengths = pd.DataFrame(length_rows)
    summary_dir = summary_directory(config)
    category_dir = category_directory(config)
    summary_dir.mkdir(parents=True, exist_ok=True)
    category_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / "policy_metrics.csv", index=False)
    lengths.to_csv(summary_dir / "response_length_metrics.csv", index=False)
    category.to_csv(category_dir / "category_label_distribution.csv", index=False)
    label_distribution.to_csv(category_dir / "label_distribution.csv", index=False)
    return {
        "policy_metrics": summary,
        "response_lengths": lengths,
        "category_distribution": category,
        "label_distribution": label_distribution,
    }
