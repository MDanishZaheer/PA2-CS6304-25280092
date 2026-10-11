# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Run and summarize the controlled Task 5 feedback diagnostics."""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict

import pandas as pd

from common.data import read_jsonl, write_jsonl
from common.logging_utils import append_jsonl, load_json, save_json
from task5_feedback.core.paths import (
    diagnostic_pairwise_path,
    diagnostics_directory,
    judge_cache_path,
    qualitative_directory,
    protocols_directory,
    raw_directory,
)
from task5_feedback.core.protocol import (
    EXPECTED_VARIANTS,
    file_sha256,
    load_diagnostic_rows,
    protect_protocol,
    validate_config,
)
from task5_feedback.core.records import validate_pairwise_prefix
from task5_feedback.evaluation.pairwise import (
    preference_from_rewards,
    text_sha256,
)
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge
from task5_feedback.rlvr import exact_reward


PAIR_CATEGORIES = {
    "corrupt_reasoning_correct_final": "reasoning_only_corruption",
    "good_reasoning_wrong_final": "wrong_final",
    "persuasive_filler_correct": "persuasive_filler",
    "gold_distractor_wrong_final": "gold_distractor",
}


def load_diagnostic_groups(config: dict) -> tuple[list[str], dict[str, dict]]:
    rows = load_diagnostic_rows(config)
    order = list(dict.fromkeys(str(row["problem_id"]) for row in rows))
    groups = defaultdict(dict)
    for row in rows:
        problem_id = str(row["problem_id"])
        reward = exact_reward(row["response"], str(row["gold_final"]))
        if reward != float(row["expected_exact_reward"]):
            raise ValueError(
                f"Exact verifier disagrees with released expectation for "
                f"problem {problem_id}/{row['variant_type']}."
            )
        groups[problem_id][row["variant_type"]] = row
    return order, dict(groups)


def expected_diagnostic_pairs(config: dict) -> list[dict]:
    order, groups = load_diagnostic_groups(config)
    expected = []
    for problem_id in order:
        variants = groups[problem_id]
        for candidate_a, candidate_b in itertools.combinations(EXPECTED_VARIANTS, 2):
            a = variants[candidate_a]
            b = variants[candidate_b]
            reward_a = exact_reward(a["response"], str(a["gold_final"]))
            reward_b = exact_reward(b["response"], str(b["gold_final"]))
            category = (
                PAIR_CATEGORIES[candidate_b]
                if candidate_a == "clean_correct"
                else "supplemental_pair"
            )
            expected.append({
                "dataset": "diagnostics",
                "prompt_id": f"diagnostic:{problem_id}",
                "problem_id": problem_id,
                "question": str(a["question"]),
                "gold_final": str(a["gold_final"]),
                "pair_name": f"{candidate_a}_vs_{candidate_b}",
                "pair_category": category,
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "response_a": a["response"],
                "response_b": b["response"],
                "response_a_sha256": text_sha256(a["response"]),
                "response_b_sha256": text_sha256(b["response"]),
                "exact_reward_a": reward_a,
                "exact_reward_b": reward_b,
                "diagnostic_preference": (
                    "A" if candidate_a == "clean_correct" else None
                ),
            })
    return expected


def diagnostic_protocol(config: dict, expected: list[dict]) -> dict:
    return {
        "diagnostic_sha256": file_sha256(
            config["paths"]["task5_diagnostics"]
        ),
        "problems": 20,
        "variants": list(EXPECTED_VARIANTS),
        "comparison_count": len(expected),
        "judge_model": config["ai_judge_model"],
        "rubric_sha256": hashlib.sha256(PAIRWISE_RUBRIC.encode()).hexdigest(),
        "orientation": "released deterministic SHA-256 A/B balancing",
        "max_input_length": int(config["task5_judge_max_input_length"]),
        "max_new_tokens": int(config["task5_judge_max_new_tokens"]),
    }


def judge_diagnostic_pairs(config: dict, overwrite: bool = False) -> list[dict]:
    """Judge all ten within-problem pairs for each of twenty problems."""
    validate_config(config)
    expected = expected_diagnostic_pairs(config)
    destination = diagnostic_pairwise_path(config)
    protocol_path = protocols_directory(config) / "diagnostic_pairwise.json"
    protocol = diagnostic_protocol(config, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if destination.exists():
            destination.unlink()
        protect_protocol(protocol_path, protocol, overwrite=True)
    else:
        protect_protocol(protocol_path, protocol)
    existing = read_jsonl(destination) if destination.exists() else []
    validate_pairwise_prefix(existing, expected)
    if len(existing) == len(expected):
        print("Using completed controlled-diagnostic judgments.")
        return existing

    judge = PairwiseAIJudge(config, judge_cache_path(config))
    try:
        for index in range(len(existing), len(expected)):
            pair = expected[index]
            detail = judge.compare_detail(
                pair["question"], pair["response_a"], pair["response_b"]
            )
            verifier_preference = preference_from_rewards(
                pair["exact_reward_a"], pair["exact_reward_b"]
            )
            append_jsonl(destination, {
                **pair,
                "response_a": None,
                "response_b": None,
                "judge_preference": detail["preference"],
                "verifier_preference": verifier_preference,
                "verifier_judge_agree": (
                    detail["preference"] == verifier_preference
                ),
                "raw_judge_output": detail["raw_output"],
                "judge_swapped": detail["swapped"],
                "judge_parse_failure": detail["parse_failure"],
                "judge_input_tokens": detail["input_tokens"],
                "judge_inference_seconds": detail["inference_seconds"],
                "judge_cache_hit": detail["cache_hit"],
                "judge_cache_key": detail["cache_key"],
            })
            if (index + 1) % 20 == 0:
                print(
                    f"Diagnostics: judged {index + 1}/{len(expected)} pairs",
                    flush=True,
                )
    finally:
        usage = judge.usage()
        judge.close()
        usage_path = destination.parent / "judge_usage.json"
        previous = load_json(usage_path) if usage_path.exists() else {}
        cumulative = {
            key: float(previous.get(key, 0.0)) + float(value)
            for key, value in usage.items()
        }
        save_json(usage_path, cumulative)

    records = read_jsonl(destination)
    validate_pairwise_prefix(records, expected)
    if len(records) != len(expected):
        raise RuntimeError("Controlled diagnostic judging is incomplete.")
    return records


def _outcome_rates(preferences: pd.Series) -> dict:
    return {
        "better_response_rate": float((preferences == "A").mean()),
        "tie_rate": float((preferences == "TIE").mean()),
        "wrong_preference_rate": float((preferences == "B").mean()),
    }


def _interpret(verifier: str, judge: str) -> str:
    if verifier == "TIE" and judge == "A":
        return "ai_usefully_distinguishes_a_verifier_tie"
    if verifier == "TIE" and judge == "B":
        return "ai_prefers_the_diagnostic_perturbation"
    if verifier == "TIE" and judge == "TIE":
        return "both_feedback_sources_tie"
    if verifier == "A" and judge == "A":
        return "both_prefer_the_diagnostically_better_response"
    if verifier == "A" and judge == "TIE":
        return "ai_misses_an_exact_outcome_difference"
    return "ai_judge_error_against_exact_outcome"


def build_diagnostic_tables(config: dict) -> dict[str, pd.DataFrame]:
    """Build five-variant scores, controlled pair rates, sensitivities, and examples."""
    expected = expected_diagnostic_pairs(config)
    path = diagnostic_pairwise_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Run controlled diagnostic judging first: {path}")
    records = read_jsonl(path)
    validate_pairwise_prefix(records, expected)
    if len(records) != len(expected):
        raise ValueError(f"Diagnostic judgments are incomplete: {len(records)}/200.")
    pairwise = pd.DataFrame(records)
    canonical = pairwise[pairwise["pair_category"] != "supplemental_pair"]

    pair_rows = []
    for category in PAIR_CATEGORIES.values():
        selected = canonical[canonical["pair_category"] == category]
        for mechanism, column in (
            ("RLVR_exact_verifier", "verifier_preference"),
            ("RLAIF_AI_judge", "judge_preference"),
        ):
            pair_rows.append({
                "pair_category": category,
                "mechanism": mechanism,
                "comparison_count": len(selected),
                **_outcome_rates(selected[column]),
            })
    pair_summary = pd.DataFrame(pair_rows)

    category_rows = []
    for mechanism, column in (
        ("RLVR_exact_verifier", "verifier_preference"),
        ("RLAIF_AI_judge", "judge_preference"),
    ):
        category_rows.append({
            "variant_category": "clean_correct",
            "category_role": "reference aggregated over four perturbations",
            "mechanism": mechanism,
            "comparison_count": len(canonical),
            **_outcome_rates(canonical[column]),
        })
        for variant, category in PAIR_CATEGORIES.items():
            selected = canonical[canonical["pair_category"] == category]
            category_rows.append({
                "variant_category": variant,
                "category_role": "clean_correct is the diagnostically better response",
                "mechanism": mechanism,
                "comparison_count": len(selected),
                **_outcome_rates(selected[column]),
            })
    category_summary = pd.DataFrame(category_rows)

    problem_order, groups = load_diagnostic_groups(config)
    reward_rows = []
    for problem_id in problem_order:
        problem_pairs = pairwise[pairwise["problem_id"] == problem_id]
        wins = {variant: 0.0 for variant in EXPECTED_VARIANTS}
        ties = {variant: 0.0 for variant in EXPECTED_VARIANTS}
        for row in problem_pairs.itertuples():
            if row.judge_preference == "A":
                wins[row.candidate_a] += 1.0
            elif row.judge_preference == "B":
                wins[row.candidate_b] += 1.0
            else:
                ties[row.candidate_a] += 1.0
                ties[row.candidate_b] += 1.0
        for variant in EXPECTED_VARIANTS:
            source = groups[problem_id][variant]
            reward_rows.append({
                "problem_id": problem_id,
                "variant_category": variant,
                "exact_reward": exact_reward(
                    source["response"], str(source["gold_final"])
                ),
                "ai_group_reward": (
                    wins[variant] + 0.5 * ties[variant]
                ) / 4.0,
                "response_words": len(str(source["response"]).split()),
            })
    reward_records = pd.DataFrame(reward_rows)
    variant_scores = (
        reward_records.groupby("variant_category", observed=False)
        .agg(
            example_count=("problem_id", "size"),
            exact_reward_mean=("exact_reward", "mean"),
            exact_reward_std=("exact_reward", lambda x: x.std(ddof=0)),
            ai_group_reward_mean=("ai_group_reward", "mean"),
            ai_group_reward_std=("ai_group_reward", lambda x: x.std(ddof=0)),
            mean_response_words=("response_words", "mean"),
        )
        .reindex(EXPECTED_VARIANTS)
        .reset_index()
    )

    sensitivity_rows = []
    for mechanism in ("RLVR_exact_verifier", "RLAIF_AI_judge"):
        indexed = pair_summary[pair_summary["mechanism"] == mechanism].set_index(
            "pair_category"
        )
        sensitivity_rows.append({
            "mechanism": mechanism,
            "S_reason": float(
                indexed.loc["reasoning_only_corruption", "better_response_rate"]
            ),
            "S_outcome": float(
                indexed.loc["wrong_final", "better_response_rate"]
            ),
            "filler_clean_preference_rate": float(
                indexed.loc["persuasive_filler", "better_response_rate"]
            ),
            "filler_wrong_preference_rate": float(
                indexed.loc["persuasive_filler", "wrong_preference_rate"]
            ),
            "gold_distractor_robustness": float(
                indexed.loc["gold_distractor", "better_response_rate"]
            ),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)

    examples = []
    for category in PAIR_CATEGORIES.values():
        selected = canonical[canonical["pair_category"] == category].copy()
        selected["priority"] = (
            (selected["verifier_preference"] == selected["judge_preference"])
            .astype(int)
        )
        row = selected.sort_values(["priority", "problem_id"]).iloc[0]
        source_a = groups[str(row["problem_id"])][row["candidate_a"]]
        source_b = groups[str(row["problem_id"])][row["candidate_b"]]
        examples.append({
            "pair_category": category,
            "problem_id": str(row["problem_id"]),
            "question": row["question"],
            "candidate_a_variant": row["candidate_a"],
            "candidate_a_response": source_a["response"],
            "candidate_b_variant": row["candidate_b"],
            "candidate_b_response": source_b["response"],
            "verifier_preference": row["verifier_preference"],
            "judge_preference": row["judge_preference"],
            "interpretation": _interpret(
                row["verifier_preference"], row["judge_preference"]
            ),
            "report_comment": "",
        })

    directory = diagnostics_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    pair_summary.to_csv(directory / "controlled_pair_metrics.csv", index=False)
    category_summary.to_csv(
        directory / "five_variant_mechanism_metrics.csv", index=False
    )
    variant_scores.to_csv(directory / "variant_reward_scores.csv", index=False)
    sensitivity.to_csv(directory / "sensitivity_summary.csv", index=False)
    write_jsonl(directory / "qualitative_examples.jsonl", examples)
    write_jsonl(
        qualitative_directory(config) / "diagnostic_examples.jsonl", examples
    )
    write_jsonl(raw_directory(config) / "diagnostic_pairwise_judgments.jsonl", records)
    return {
        "controlled_pairs": pair_summary,
        "category_metrics": category_summary,
        "variant_scores": variant_scores,
        "sensitivity": sensitivity,
        "qualitative": examples,
    }
