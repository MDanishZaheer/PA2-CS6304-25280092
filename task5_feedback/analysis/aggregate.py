# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Combine Task 5 domain, diagnostic, transfer, and research-question evidence."""

from __future__ import annotations

import pandas as pd

from common.data import read_jsonl
from common.logging_utils import load_json, save_json
from task5_feedback.core.paths import (
    diagnostic_pairwise_path,
    domain_directory,
    pairwise_path,
    results_root,
)
from task5_feedback.evaluation.diagnostics import build_diagnostic_tables
from task5_feedback.evaluation.math_metrics import build_math_tables


def build_transfer_comparison(
    config: dict, gsm: pd.DataFrame, transfer: pd.DataFrame
) -> pd.DataFrame:
    """Save in-domain versus out-of-domain changes using matched policy rows."""
    left = gsm.set_index("policy")
    right = transfer.set_index("policy")
    records = []
    for policy_name in left.index:
        record = {
            "policy": policy_name,
            "gsm_exact_accuracy": float(left.loc[policy_name, "exact_accuracy"]),
            "transfer_exact_accuracy": float(
                right.loc[policy_name, "exact_accuracy"]
            ),
            "exact_accuracy_drop": float(
                left.loc[policy_name, "exact_accuracy"]
                - right.loc[policy_name, "exact_accuracy"]
            ),
            "gsm_format_compliance": float(
                left.loc[policy_name, "format_compliance_rate"]
            ),
            "transfer_format_compliance": float(
                right.loc[policy_name, "format_compliance_rate"]
            ),
            "format_compliance_drop": float(
                left.loc[policy_name, "format_compliance_rate"]
                - right.loc[policy_name, "format_compliance_rate"]
            ),
            "gsm_mean_response_length": float(
                left.loc[policy_name, "mean_response_length"]
            ),
            "transfer_mean_response_length": float(
                right.loc[policy_name, "mean_response_length"]
            ),
            "response_length_change": float(
                right.loc[policy_name, "mean_response_length"]
                - left.loc[policy_name, "mean_response_length"]
            ),
            "gsm_ai_tournament_score": float(
                left.loc[policy_name, "ai_tournament_score"]
            ),
            "transfer_ai_tournament_score": float(
                right.loc[policy_name, "ai_tournament_score"]
            ),
            "ai_tournament_score_drop": float(
                left.loc[policy_name, "ai_tournament_score"]
                - right.loc[policy_name, "ai_tournament_score"]
            ),
        }
        if policy_name != "sft":
            record.update({
                "gsm_ai_pairwise_score_vs_sft": float(
                    left.loc[policy_name, "ai_pairwise_score_vs_sft"]
                ),
                "transfer_ai_pairwise_score_vs_sft": float(
                    right.loc[policy_name, "ai_pairwise_score_vs_sft"]
                ),
                "ai_pairwise_score_vs_sft_drop": float(
                    left.loc[policy_name, "ai_pairwise_score_vs_sft"]
                    - right.loc[policy_name, "ai_pairwise_score_vs_sft"]
                ),
            })
        else:
            record.update({
                "gsm_ai_pairwise_score_vs_sft": None,
                "transfer_ai_pairwise_score_vs_sft": None,
                "ai_pairwise_score_vs_sft_drop": None,
            })
        records.append(record)
    table = pd.DataFrame(records)
    destination = domain_directory(config, "transfer") / "domain_shift_metrics.csv"
    table.to_csv(destination, index=False)
    return table


def _unique_judge_cost(config: dict) -> dict:
    records = []
    for path in (
        pairwise_path(config, "gsm"),
        pairwise_path(config, "transfer"),
        diagnostic_pairwise_path(config),
    ):
        records.extend(read_jsonl(path))
    table = pd.DataFrame(records).drop_duplicates("judge_cache_key")
    elapsed = pd.to_numeric(
        table["judge_inference_seconds"], errors="coerce"
    ).fillna(0.0)
    load_seconds = 0.0
    for usage_path in (
        pairwise_path(config, "gsm").parent / "judge_usage.json",
        pairwise_path(config, "transfer").parent / "judge_usage.json",
        diagnostic_pairwise_path(config).parent / "judge_usage.json",
    ):
        if usage_path.exists():
            load_seconds += float(load_json(usage_path).get("model_load_seconds", 0.0))
    return {
        "unique_pairwise_comparisons": len(table),
        "recorded_model_load_seconds": load_seconds,
        "recorded_inference_seconds": float(elapsed.sum()),
        "recorded_total_judge_seconds": float(elapsed.sum()) + load_seconds,
        "mean_recorded_seconds_per_comparison": float(elapsed.mean()),
        "parse_failure_rate": float(table["judge_parse_failure"].mean()),
    }


def build_feedback_source_summary(
    config: dict, diagnostics: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Compare coverage, noise, exploitability, and inference cost."""
    pairs = diagnostics["controlled_pairs"]
    sensitivity = diagnostics["sensitivity"].set_index("mechanism")
    rows = []
    for mechanism in ("RLVR_exact_verifier", "RLAIF_AI_judge"):
        selected = pairs[pairs["mechanism"] == mechanism]
        weights = selected["comparison_count"]
        weighted = lambda column: float(
            (selected[column] * weights).sum() / weights.sum()
        )
        row = {
            "mechanism": mechanism,
            "feedback_coverage": 1.0,
            "diagnostic_decisive_rate": 1.0 - weighted("tie_rate"),
            "diagnostic_better_response_rate": weighted(
                "better_response_rate"
            ),
            "diagnostic_wrong_preference_rate": weighted(
                "wrong_preference_rate"
            ),
            "S_reason": float(sensitivity.loc[mechanism, "S_reason"]),
            "S_outcome": float(sensitivity.loc[mechanism, "S_outcome"]),
            "filler_wrong_preference_rate": float(
                sensitivity.loc[mechanism, "filler_wrong_preference_rate"]
            ),
            "gold_distractor_robustness": float(
                sensitivity.loc[mechanism, "gold_distractor_robustness"]
            ),
            "requires_model_inference": mechanism == "RLAIF_AI_judge",
            "utility_scale_note": (
                "Binary designated-final correctness; sparse but exact to parser."
                if mechanism == "RLVR_exact_verifier"
                else "Pairwise relative preference; not calibrated to RLVR reward."
            ),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    judge_cost = _unique_judge_cost(config)
    for key, value in judge_cost.items():
        table.loc[
            table["mechanism"] == "RLAIF_AI_judge", key
        ] = value
    table.loc[
        table["mechanism"] == "RLVR_exact_verifier",
        [
            "unique_pairwise_comparisons",
            "recorded_model_load_seconds",
            "recorded_inference_seconds",
            "recorded_total_judge_seconds",
            "mean_recorded_seconds_per_comparison",
            "parse_failure_rate",
        ],
    ] = [0, 0.0, 0.0, 0.0, 0.0, 0.0]
    table["valid_feedback_rate"] = 1.0
    table.loc[
        table["mechanism"] == "RLAIF_AI_judge", "valid_feedback_rate"
    ] = 1.0 - judge_cost["parse_failure_rate"]
    table.to_csv(results_root(config) / "feedback_source_summary.csv", index=False)
    indexed = table.set_index("mechanism")
    verifier = indexed.loc["RLVR_exact_verifier"]
    judge = indexed.loc["RLAIF_AI_judge"]
    discussion = [
        "# Feedback-source discussion",
        "",
        "RLVR provides deterministic designated-final coverage with no model "
        "inference, but it is decisive on only "
        f"{100 * verifier['diagnostic_decisive_rate']:.1f}% of the controlled "
        "pairs because equal final answers receive identical reward.",
        "",
        "RLAIF is decisive on "
        f"{100 * judge['diagnostic_decisive_rate']:.1f}% of controlled pairs. "
        f"Its wrong-preference rate is "
        f"{100 * judge['diagnostic_wrong_preference_rate']:.1f}%, and its "
        f"persuasive-filler wrong-preference rate is "
        f"{100 * judge['filler_wrong_preference_rate']:.1f}%. These quantities "
        "measure judge noise and surface-style exploitability rather than exact "
        "outcome specification.",
        "",
        f"The saved unique AI comparisons record approximately "
        f"{judge['recorded_total_judge_seconds']:.1f} seconds of judge loading "
        "plus inference. "
        "RLVR uses a local parser instead. Binary RLVR reward and AI pairwise "
        "reward are not calibrated utility scales and are not compared as raw "
        "magnitudes.",
    ]
    (results_root(config) / "feedback_source_discussion.md").write_text(
        "\n".join(discussion) + "\n", encoding="utf-8"
    )
    return table


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _maximum_categories(series: pd.Series) -> tuple[list[str], float]:
    maximum = float(series.max())
    categories = [str(name) for name, value in series.items() if value == maximum]
    return categories, maximum


def build_research_question_evidence(
    config: dict,
    diagnostics: dict[str, pd.DataFrame],
    transfer: pd.DataFrame,
    feedback: pd.DataFrame,
) -> dict:
    """Write concise evidence-backed prompts for all four research questions."""
    sensitivity = diagnostics["sensitivity"].set_index("mechanism")
    ai = sensitivity.loc["RLAIF_AI_judge"]
    verifier = sensitivity.loc["RLVR_exact_verifier"]
    ai_feedback = feedback.set_index("mechanism").loc["RLAIF_AI_judge"]
    pairs = diagnostics["controlled_pairs"]
    wrong = pairs.pivot(
        index="pair_category",
        columns="mechanism",
        values="wrong_preference_rate",
    )
    transfer_index = transfer.set_index("policy")
    strongest_ai_errors, strongest_ai_error_rate = _maximum_categories(
        wrong["RLAIF_AI_judge"]
    )
    strongest_verifier_errors, strongest_verifier_error_rate = (
        _maximum_categories(wrong["RLVR_exact_verifier"])
    )
    exact_rank = transfer.sort_values(
        "transfer_exact_accuracy", ascending=False
    )["policy"].tolist()
    evidence = {
        "rq1": {
            "question": "Where does AI feedback distinguish verifier ties?",
            "S_reason_verifier": float(verifier["S_reason"]),
            "S_reason_ai": float(ai["S_reason"]),
            "filler_wrong_ai": float(ai["filler_wrong_preference_rate"]),
            "unique_ai_comparisons": int(
                ai_feedback["unique_pairwise_comparisons"]
            ),
            "recorded_ai_inference_seconds": float(
                ai_feedback["recorded_inference_seconds"]
            ),
            "interpretation": (
                "AI feedback adds information on equal-final-answer reasoning "
                "pairs, but its usefulness must be discounted by filler/style errors."
            ),
        },
        "rq2": {
            "question": "When is the judge influenced by style or framing?",
            "filler_wrong_preference_rate": float(
                ai["filler_wrong_preference_rate"]
            ),
            "filler_clean_preference_rate": float(
                ai["filler_clean_preference_rate"]
            ),
            "interpretation": (
                "The persuasive-filler condition isolates surface-style bias "
                "because both responses retain the correct designated answer."
            ),
        },
        "rq3": {
            "question": "Which source is vulnerable to each diagnostic?",
            "largest_ai_wrong_preference_categories": strongest_ai_errors,
            "largest_ai_wrong_preference_rate": strongest_ai_error_rate,
            "largest_verifier_wrong_preference_categories": (
                strongest_verifier_errors
            ),
            "largest_verifier_wrong_preference_rate": (
                strongest_verifier_error_rate
            ),
            "verifier_reasoning_sensitivity": float(verifier["S_reason"]),
            "verifier_outcome_sensitivity": float(verifier["S_outcome"]),
            "interpretation": (
                "Verifier ties expose specification blindness; AI wrong "
                "preferences expose judge noise or style bias."
            ),
        },
        "rq4": {
            "question": "Which behavior transfers to SVAMP?",
            "transfer_exact_accuracy_ranking": exact_rank,
            "exact_accuracy_drop": {
                policy: float(transfer_index.loc[policy, "exact_accuracy_drop"])
                for policy in transfer_index.index
            },
            "interpretation": (
                "Use exact and AI-pairwise drops together; neither metric alone "
                "establishes general reasoning transfer."
            ),
        },
    }
    save_json(results_root(config) / "research_question_evidence.json", evidence)

    markdown = [
        "# Task 5 research-question evidence",
        "",
        "## RQ1 - Extra flexibility of AI feedback",
        (
            f"The exact verifier's reasoning sensitivity is "
            f"{_percent(verifier['S_reason'])}, while the AI judge's is "
            f"{_percent(ai['S_reason'])}. The AI filler wrong-preference rate is "
            f"{_percent(ai['filler_wrong_preference_rate'])}. AI feedback is "
            "useful when it correctly separates reasoning quality inside verifier "
            "ties, but filler errors show that not every distinction is useful."
        ),
        "",
        "## RQ2 - Style, verbosity, and persuasive framing",
        (
            f"On clean-versus-persuasive-filler pairs, the judge preferred the "
            f"clean response {_percent(ai['filler_clean_preference_rate'])} of "
            f"the time and incorrectly preferred filler "
            f"{_percent(ai['filler_wrong_preference_rate'])} of the time. Use the "
            "saved examples to determine whether the remaining decisions reflect "
            "verbosity, confidence, or another surface cue."
        ),
        "",
        "## RQ3 - Reward-source vulnerabilities",
        (
            f"The AI judge's largest wrong-preference rate occurs for "
            f"{', '.join(f'`{value}`' for value in strongest_ai_errors)} "
            f"at {_percent(strongest_ai_error_rate)}. The verifier has S_reason="
            f"{_percent(verifier['S_reason'])} and S_outcome="
            f"{_percent(verifier['S_outcome'])}: this contrast quantifies its "
            "outcome sensitivity and reasoning blindness."
        ),
        "",
        "## RQ4 - Out-of-domain transfer",
        (
            "SVAMP exact-accuracy ranking: " + " > ".join(exact_rank) + ". "
            + "; ".join(
                f"{policy} exact drop={_percent(value)}"
                for policy, value in evidence["rq4"]["exact_accuracy_drop"].items()
            )
            + ". Compare these changes with the saved AI-pairwise and response-"
            "length changes before making a transfer claim."
        ),
    ]
    (results_root(config) / "research_question_evidence.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return evidence


def build_all_evidence(config: dict) -> dict:
    """Build every Task 5 table from completed cached inference."""
    gsm = build_math_tables(config, "gsm")
    diagnostics = build_diagnostic_tables(config)
    svamp = build_math_tables(config, "transfer")
    transfer = build_transfer_comparison(
        config, gsm["policy_metrics"], svamp["policy_metrics"]
    )
    feedback = build_feedback_source_summary(config, diagnostics)
    research_questions = build_research_question_evidence(
        config, diagnostics, transfer, feedback
    )
    manifest = {
        "gsm_policy_rows": len(gsm["policy_metrics"]),
        "transfer_policy_rows": len(svamp["policy_metrics"]),
        "diagnostic_pair_rows": len(diagnostics["controlled_pairs"]),
        "diagnostic_variant_rows": len(diagnostics["variant_scores"]),
        "research_questions": len(research_questions),
    }
    save_json(results_root(config) / "evidence_manifest.json", manifest)
    return {
        "gsm": gsm,
        "diagnostics": diagnostics,
        "transfer": svamp,
        "domain_shift": transfer,
        "feedback_sources": feedback,
        "research_questions": research_questions,
    }
