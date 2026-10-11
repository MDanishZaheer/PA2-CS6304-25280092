# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Summarize frozen-policy math behavior and verifier-judge agreement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.data import read_jsonl, write_jsonl
from task5_feedback.core.paths import (
    POLICY_ORDER,
    domain_directory,
    pairwise_path,
    raw_directory,
)
from task5_feedback.core.records import validate_pairwise_prefix
from task5_feedback.evaluation.pairwise import (
    PAIR_SPECS,
    expected_policy_pairs,
    load_dataset_responses,
)


def _rate(values) -> float:
    return float(np.mean(list(values))) if len(values) else float("nan")


def _policy_tournament_scores(pairwise: pd.DataFrame) -> pd.DataFrame:
    records = []
    for prompt_id, group in pairwise.groupby("prompt_id", sort=False):
        points = {name: 0.0 for name in POLICY_ORDER}
        for row in group.itertuples():
            if row.judge_preference == "A":
                points[row.candidate_a] += 1.0
            elif row.judge_preference == "B":
                points[row.candidate_b] += 1.0
            else:
                points[row.candidate_a] += 0.5
                points[row.candidate_b] += 0.5
        for policy_name in POLICY_ORDER:
            records.append({
                "prompt_id": prompt_id,
                "policy": policy_name,
                "ai_tournament_score": points[policy_name] / 2.0,
            })
    return pd.DataFrame(records)


def _against_sft(pairwise: pd.DataFrame, policy_name: str) -> dict:
    if policy_name == "sft":
        return {
            "ai_win_rate_vs_sft": None,
            "ai_tie_rate_vs_sft": None,
            "ai_pairwise_score_vs_sft": None,
            "verifier_judge_agreement_vs_sft": None,
            "verifier_judge_decisive_agreement_vs_sft": None,
        }
    selected = pairwise[pairwise["pair_name"] == f"{policy_name}_vs_sft"]
    decisive = selected[selected["verifier_preference"] != "TIE"]
    return {
        "ai_win_rate_vs_sft": _rate(selected["judge_preference"] == "A"),
        "ai_tie_rate_vs_sft": _rate(selected["judge_preference"] == "TIE"),
        "ai_pairwise_score_vs_sft": _rate(
            (selected["judge_preference"] == "A").astype(float)
            + 0.5 * (selected["judge_preference"] == "TIE").astype(float)
        ),
        "verifier_judge_agreement_vs_sft": _rate(
            selected["verifier_judge_agree"]
        ),
        "verifier_judge_decisive_agreement_vs_sft": _rate(
            decisive["verifier_judge_agree"]
        ),
    }


def build_math_tables(config: dict, dataset: str) -> dict[str, pd.DataFrame]:
    """Save required policy, pairwise, category, and failure-case evidence."""
    policies = load_dataset_responses(config, dataset)
    expected = expected_policy_pairs(config, dataset)
    pair_path = pairwise_path(config, dataset)
    if not pair_path.exists():
        raise FileNotFoundError(f"Run the {dataset} pairwise judge first: {pair_path}")
    pair_records = read_jsonl(pair_path)
    validate_pairwise_prefix(pair_records, expected)
    if len(pair_records) != len(expected):
        raise ValueError(
            f"{dataset} pairwise judgments are incomplete: "
            f"{len(pair_records)}/{len(expected)}."
        )
    pairwise = pd.DataFrame(pair_records)
    tournaments = _policy_tournament_scores(pairwise)

    policy_rows = []
    response_frames = []
    for policy_name in POLICY_ORDER:
        policy = pd.DataFrame(policies[policy_name])
        response_frames.append(policy)
        lengths = policy["response_tokens"].astype(float)
        tournament = tournaments[
            tournaments["policy"] == policy_name
        ]["ai_tournament_score"]
        policy_rows.append({
            "dataset": dataset,
            "policy": policy_name,
            "example_count": len(policy),
            "exact_accuracy": float(policy["exact_reward"].mean()),
            "format_compliance_rate": float(policy["format_compliant"].mean()),
            "mean_response_length": float(lengths.mean()),
            "response_length_std": float(lengths.std(ddof=0)),
            "response_length_iqr": float(
                lengths.quantile(0.75) - lengths.quantile(0.25)
            ),
            "truncated_fraction": float(policy["truncated"].mean()),
            "ai_tournament_score": float(tournament.mean()),
            **_against_sft(pairwise, policy_name),
        })
    policy_summary = pd.DataFrame(policy_rows)

    pair_rows = []
    for pair_name, candidate_a, candidate_b in PAIR_SPECS:
        selected = pairwise[pairwise["pair_name"] == pair_name]
        verifier_decisive = selected[selected["verifier_preference"] != "TIE"]
        pair_rows.append({
            "dataset": dataset,
            "pair_name": pair_name,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "comparison_count": len(selected),
            "judge_a_rate": _rate(selected["judge_preference"] == "A"),
            "judge_b_rate": _rate(selected["judge_preference"] == "B"),
            "judge_tie_rate": _rate(selected["judge_preference"] == "TIE"),
            "judge_score_for_a": _rate(
                (selected["judge_preference"] == "A").astype(float)
                + 0.5 * (selected["judge_preference"] == "TIE").astype(float)
            ),
            "verifier_a_rate": _rate(selected["verifier_preference"] == "A"),
            "verifier_b_rate": _rate(selected["verifier_preference"] == "B"),
            "verifier_tie_rate": _rate(
                selected["verifier_preference"] == "TIE"
            ),
            "verifier_judge_agreement": _rate(
                selected["verifier_judge_agree"]
            ),
            "agreement_when_verifier_decisive": _rate(
                verifier_decisive["verifier_judge_agree"]
            ),
            "judge_breaks_verifier_tie_rate": _rate(
                selected.loc[
                    selected["verifier_preference"] == "TIE",
                    "judge_preference",
                ] != "TIE"
            ),
        })
    pair_summary = pd.DataFrame(pair_rows)

    responses = pd.concat(response_frames, ignore_index=True)
    categories = (
        responses.groupby(["policy", "category"], observed=False)
        .agg(
            example_count=("prompt_id", "size"),
            exact_accuracy=("exact_reward", "mean"),
            format_compliance_rate=("format_compliant", "mean"),
            mean_response_length=("response_tokens", "mean"),
        )
        .reset_index()
    )
    failures = pairwise[
        (~pairwise["verifier_judge_agree"])
        | pairwise["judge_parse_failure"]
    ].copy()

    directory = domain_directory(config, dataset)
    directory.mkdir(parents=True, exist_ok=True)
    policy_summary.to_csv(directory / "policy_metrics.csv", index=False)
    pair_summary.to_csv(directory / "pairwise_metrics.csv", index=False)
    categories.to_csv(directory / "category_metrics.csv", index=False)
    failures.to_json(
        directory / "verifier_judge_disagreements.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    write_jsonl(
        raw_directory(config) / f"{dataset}_responses.jsonl",
        responses.to_dict(orient="records"),
    )
    write_jsonl(
        raw_directory(config) / f"{dataset}_pairwise_judgments.jsonl",
        pair_records,
    )
    return {
        "policy_metrics": policy_summary,
        "pairwise_metrics": pair_summary,
        "category_metrics": categories,
        "disagreements": failures,
    }
