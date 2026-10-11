# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Compare hidden manual labels with cached AI-judge labels."""

from __future__ import annotations

import math

import pandas as pd

from common.logging_utils import save_json
from task4_safety.core.paths import POLICY_ORDER, judgment_path, manual_directory
from task4_safety.core.protocol import BEHAVIOR_LABELS
from task4_safety.evaluation.manual_audit import (
    manual_sheet_long,
    validate_manual_sheet,
)


def _finite_or_none(value: float):
    return float(value) if math.isfinite(float(value)) else None


def _summary_row(table: pd.DataFrame, policy_name: str) -> dict:
    agreement = table["manual_label"] == table["ai_label"]
    manual_rates = table["manual_label"].value_counts(normalize=True)
    ai_rates = table["ai_label"].value_counts(normalize=True)
    expected_agreement = sum(
        float(manual_rates.get(label, 0.0)) * float(ai_rates.get(label, 0.0))
        for label in BEHAVIOR_LABELS
    )
    observed_agreement = float(agreement.mean())
    denominator = 1.0 - expected_agreement
    kappa = (
        (observed_agreement - expected_agreement) / denominator
        if not math.isclose(denominator, 0.0)
        else float("nan")
    )
    return {
        "policy": policy_name,
        "sample_count": len(table),
        "agreement_count": int(agreement.sum()),
        "agreement_rate": observed_agreement,
        "cohen_kappa": _finite_or_none(kappa),
        "ai_ambiguous_rate": float((table["ai_label"] == "AMBIGUOUS").mean()),
        "manual_ambiguous_rate": float(
            (table["manual_label"] == "AMBIGUOUS").mean()
        ),
    }


def build_manual_agreement(config: dict, require_complete: bool = False):
    """Save completion status, and agreement evidence once all labels exist."""
    directory = manual_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    status = validate_manual_sheet(config, require_complete=False)
    save_json(directory / "manual_completion.json", status)
    if not status["complete"]:
        if require_complete:
            raise ValueError(
                "Manual audit is incomplete: "
                f"{status['filled_response_labels']}/"
                f"{status['required_response_labels']} labels filled."
            )
        print(
            "Manual agreement is pending: "
            f"{status['filled_response_labels']}/"
            f"{status['required_response_labels']} labels filled."
        )
        return None

    manual = manual_sheet_long(config, require_complete=True)
    ai_frames = []
    for policy_name in POLICY_ORDER:
        path = judgment_path(config, policy_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing AI judgments: {path}")
        ai = pd.read_json(path, lines=True)[
            ["xstest_id", "policy", "ai_label", "ai_confidence", "ai_rationale_tag"]
        ]
        ai_frames.append(ai)
    ai = pd.concat(ai_frames, ignore_index=True)
    comparison = manual.merge(
        ai, on=["xstest_id", "policy"], how="left", validate="one_to_one"
    )
    if comparison["ai_label"].isna().any():
        raise ValueError("Manual audit rows are missing matching AI judgments.")
    comparison["agrees"] = comparison["manual_label"] == comparison["ai_label"]
    comparison.to_csv(directory / "audit_comparison.csv", index=False)

    summaries = [_summary_row(comparison, "all")]
    for policy_name in POLICY_ORDER:
        summaries.append(_summary_row(
            comparison[comparison["policy"] == policy_name], policy_name
        ))
    agreement = pd.DataFrame(summaries)
    agreement.to_csv(directory / "agreement_summary.csv", index=False)

    class_summaries = []
    for policy_name in POLICY_ORDER:
        policy = comparison[comparison["policy"] == policy_name]
        for benchmark_class in ("SAFE", "UNSAFE"):
            row = _summary_row(
                policy[policy["benchmark_class"] == benchmark_class],
                policy_name,
            )
            row["benchmark_class"] = benchmark_class
            class_summaries.append(row)
    agreement_by_class = pd.DataFrame(class_summaries)
    agreement_by_class.to_csv(
        directory / "agreement_by_policy_and_class.csv", index=False
    )

    full_index = pd.MultiIndex.from_product(
        [BEHAVIOR_LABELS, BEHAVIOR_LABELS],
        names=["manual_label", "ai_label"],
    )
    confusion = (
        comparison.groupby(["manual_label", "ai_label"], observed=False)
        .size()
        .reindex(full_index, fill_value=0)
        .rename("count")
        .reset_index()
    )
    confusion.to_csv(directory / "confusion_matrix.csv", index=False)

    by_policy_index = pd.MultiIndex.from_product(
        [POLICY_ORDER, BEHAVIOR_LABELS, BEHAVIOR_LABELS],
        names=["policy", "manual_label", "ai_label"],
    )
    by_policy = (
        comparison.groupby(
            ["policy", "manual_label", "ai_label"], observed=False
        )
        .size()
        .reindex(by_policy_index, fill_value=0)
        .rename("count")
        .reset_index()
    )
    by_policy.to_csv(directory / "confusion_by_policy.csv", index=False)
    disagreements = comparison[~comparison["agrees"]].copy()
    disagreements.to_json(
        directory / "judge_disagreements.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return {
        "comparison": comparison,
        "agreement": agreement,
        "agreement_by_class": agreement_by_class,
        "confusion": confusion,
        "confusion_by_policy": by_policy,
        "disagreements": disagreements,
    }
