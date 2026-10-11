# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Turn Task 3 logs into compact, report-ready evidence tables."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common.data import read_jsonl, repo_path, write_jsonl
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from task3_grpo.core.paths import result_directory
from task3_grpo.evaluation.qualitative import compare_normalization_responses


NORMALIZATION_RUNS = ["norm_canonical", "norm_dr_grpo"]


def _coefficient_of_variation(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    mean = abs(float(array.mean()))
    return float(array.std(ddof=0) / mean) if mean > 1e-12 else float("nan")


def _finite_or_none(value: float) -> float | None:
    """Keep saved JSON standards-compliant when a statistic is undefined."""
    return float(value) if np.isfinite(value) else None


def standard_summary(config: dict) -> pd.DataFrame:
    """Create a single row joining standard training and held-out behavior."""
    directory = result_directory(config, "standard")
    history = read_jsonl(directory / "train_metrics.jsonl")
    training = load_json(directory / "train_summary.json")
    evaluation = load_json(directory / "eval_metrics.json")
    last = history[-1]
    row = {
        "run_name": "standard",
        "updates": training["updates"],
        "num_generations": training["num_generations"],
        "training_generated_tokens": training["generated_tokens"],
        "wall_time_seconds": training["wall_time_seconds"],
        "peak_vram_gib": training["peak_vram_gib"],
        "final_train_reward": last["reward_mean"],
        "final_train_sampled_kl": last["sampled_kl_token_mean"],
        "mean_within_group_reward_std": float(np.mean([
            item["within_group_reward_std_mean"] for item in history
        ])),
        "mean_uninformative_group_fraction": float(np.mean([
            item["uninformative_group_fraction"] for item in history
        ])),
        "final_policy_loss": last["policy_loss"],
        "final_gradient_norm": last["gradient_norm"],
        "final_entropy": last["entropy_token_mean"],
        "final_response_length": last["response_length_mean"],
        "heldout_reward": evaluation["reward_model_score_mean"],
        "heldout_reward_std": evaluation["reward_model_score_std"],
        "heldout_sampled_kl": evaluation["sampled_kl_token_mean"],
        "heldout_entropy": evaluation["entropy_token_mean"],
        "heldout_response_length": evaluation["response_length_mean"],
        "heldout_response_length_std": evaluation["response_length_std"],
        "heldout_truncated_fraction": evaluation["truncated_fraction"],
    }
    table = pd.DataFrame([row])
    table.to_csv(directory / "summary.csv", index=False)
    return table


def standard_stability_diagnostics(config: dict) -> dict:
    """Quantify candidate sources of sample inefficiency without interpreting them."""
    directory = result_directory(config, "standard")
    history = read_jsonl(directory / "train_metrics.jsonl")
    summary = load_json(directory / "train_summary.json")
    informative_groups = float(sum(
        1.0 - row["uninformative_group_fraction"] for row in history
    ))
    diagnostics = {
        "updates": len(history),
        "generated_tokens": int(summary["generated_tokens"]),
        "mean_uninformative_group_fraction": float(np.mean([
            row["uninformative_group_fraction"] for row in history
        ])),
        "mean_truncated_completion_fraction": float(np.mean([
            row["truncated_fraction"] for row in history
        ])),
        "informative_groups_across_updates": informative_groups,
        "generated_tokens_per_informative_group": (
            float(summary["generated_tokens"]) / max(informative_groups, 1.0)
        ),
        "within_group_reward_std_cv": _finite_or_none(_coefficient_of_variation([
            row["within_group_reward_std_mean"] for row in history
        ])),
        "gradient_norm_cv": _finite_or_none(_coefficient_of_variation([
            row["gradient_norm"] for row in history
        ])),
        "policy_loss_standard_deviation": float(np.std([
            row["policy_loss"] for row in history
        ])),
        "reward_standard_deviation_across_updates": float(np.std([
            row["reward_mean"] for row in history
        ])),
        "note": (
            "These are candidate mechanisms for RQ3. The student should identify "
            "the dominant one only after comparing the numerical evidence and examples."
        ),
    }
    save_json(directory / "stability_diagnostics.json", diagnostics)
    return diagnostics


def _length_bin(length: int, maximum: int) -> str:
    first = maximum // 3
    second = (2 * maximum) // 3
    if length <= first:
        return "short"
    if length <= second:
        return "medium"
    return "long"


def length_conditioned_training(config: dict, run_name: str) -> pd.DataFrame:
    """Aggregate objective-weight proxies in fixed thirds of the length cap."""
    directory = result_directory(config, run_name)
    rows = read_jsonl(directory / "completion_metrics.jsonl")
    maximum = int(config["max_completion_length"])
    for row in rows:
        row["length_bin"] = _length_bin(int(row["response_length"]), maximum)
    records = []
    for label in ("short", "medium", "long"):
        selected = [row for row in rows if row["length_bin"] == label]
        trainable = [row for row in selected if int(row["training_tokens"]) > 0]
        records.append({
            "run_name": run_name,
            "loss_type": load_json(directory / "train_summary.json")["loss_type"],
            "length_bin": label,
            "lower_exclusive": 0 if label == "short" else (
                maximum // 3 if label == "medium" else (2 * maximum) // 3
            ),
            "upper_inclusive": (
                maximum // 3 if label == "short" else (
                    (2 * maximum) // 3 if label == "medium" else maximum
                )
            ),
            "completion_count": len(selected),
            "trainable_completion_count": len(trainable),
            "mean_response_length": float(np.mean([
                row["response_length"] for row in selected
            ])) if selected else float("nan"),
            "mean_reward": float(np.mean([
                row["reward"] for row in selected
            ])) if selected else float("nan"),
            "mean_absolute_advantage": float(np.mean([
                abs(row["advantage"]) for row in trainable
            ])) if trainable else float("nan"),
            "mean_per_token_gradient_weight": float(np.mean([
                row["per_token_gradient_weight"] for row in trainable
            ])) if trainable else float("nan"),
            "mean_sequence_gradient_weight_proxy": float(np.mean([
                row["sequence_gradient_weight_proxy"] for row in trainable
            ])) if trainable else float("nan"),
            "truncated_fraction": float(np.mean([
                row["truncated"] for row in selected
            ])) if selected else float("nan"),
        })
    table = pd.DataFrame(records)
    table.to_csv(directory / "length_conditioned_metrics.csv", index=False)
    return table


def normalization_evidence(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join matched training, held-out, length, and qualitative comparisons."""
    summary_rows = []
    length_tables = []
    evaluations = {}
    run_protocols = {}
    for run_name in NORMALIZATION_RUNS:
        directory = result_directory(config, run_name)
        run_protocols[run_name] = load_json(directory / "run_config.json")
        training = load_json(directory / "train_summary.json")
        evaluation = load_json(directory / "eval_metrics.json")
        history = read_jsonl(directory / "train_metrics.jsonl")
        completions = read_jsonl(directory / "completion_metrics.jsonl")
        length_table = length_conditioned_training(config, run_name)
        length_tables.append(length_table)
        evaluations[run_name] = read_jsonl(directory / "generations.jsonl")
        trainable = [row for row in completions if row["training_tokens"] > 0]
        summary_rows.append({
            "run_name": run_name,
            "loss_type": training["loss_type"],
            "updates": training["updates"],
            "maximum_generation_budget": training["maximum_generation_budget"],
            "actual_generated_tokens": training["generated_tokens"],
            "wall_time_seconds": training["wall_time_seconds"],
            "peak_vram_gib": training["peak_vram_gib"],
            "mean_train_reward": float(np.mean([row["reward_mean"] for row in history])),
            "mean_train_kl": float(np.mean([
                row["sampled_kl_token_mean"] for row in history
            ])),
            "mean_gradient_norm": float(np.mean([row["gradient_norm"] for row in history])),
            "gradient_norm_std": float(np.std([row["gradient_norm"] for row in history])),
            "gradient_weight_length_correlation": safe_corr(
                [row["response_length"] for row in trainable],
                [row["sequence_gradient_weight_proxy"] for row in trainable],
            ),
            "heldout_reward": evaluation["reward_model_score_mean"],
            "heldout_reward_std": evaluation["reward_model_score_std"],
            "heldout_sampled_kl": evaluation["sampled_kl_token_mean"],
            "heldout_entropy": evaluation["entropy_token_mean"],
            "heldout_response_length": evaluation["response_length_mean"],
            "heldout_response_length_std": evaluation["response_length_std"],
            "heldout_truncated_fraction": evaluation["truncated_fraction"],
        })

    canonical_protocol = run_protocols["norm_canonical"]
    dr_protocol = run_protocols["norm_dr_grpo"]
    matched_fields = (
        "seed",
        "base_model",
        "reward_model",
        "reward_tokenizer",
        "dtype",
        "training_reward_device",
        "training_reward_quantized",
        "midpoint_policy_sha256",
        "updates",
        "prompts_per_update",
        "policy_epochs",
        "num_generations",
        "prompt_ids",
        "prompt_pool_sha256",
        "learning_rate",
        "clip_epsilon",
        "kl_beta",
        "max_grad_norm",
        "max_prompt_length",
        "max_completion_length",
        "mask_truncated_completions",
        "reward_max_length",
        "lora",
        "generation",
    )
    mismatched = [
        field for field in matched_fields
        if canonical_protocol.get(field) != dr_protocol.get(field)
    ]
    if mismatched:
        raise ValueError(
            "Normalization forks differ in controlled fields: "
            f"{mismatched}"
        )
    if canonical_protocol.get("loss_type") != "grpo":
        raise ValueError("The canonical normalization fork has the wrong loss type.")
    if dr_protocol.get("loss_type") != "dr_grpo":
        raise ValueError("The Dr-GRPO fork has the wrong loss type.")
    if len({row["maximum_generation_budget"] for row in summary_rows}) != 1:
        raise ValueError("Normalization forks do not share a maximum token budget.")
    root = repo_path(config["results_dir"]) / "normalization"
    root.mkdir(parents=True, exist_ok=True)
    summary_table = pd.DataFrame(summary_rows)
    length_table = pd.concat(length_tables, ignore_index=True)
    summary_table.to_csv(root / "summary.csv", index=False)
    length_table.to_csv(root / "length_conditioned_metrics.csv", index=False)

    left = {row["prompt_id"]: row for row in evaluations["norm_canonical"]}
    right = {row["prompt_id"]: row for row in evaluations["norm_dr_grpo"]}
    if list(left) != list(right):
        raise ValueError("Normalization evaluations do not use identical prompt order.")
    paired = pd.DataFrame([{
        "prompt_id": prompt_id,
        "source_index": left[prompt_id].get("source_index"),
        "reward_dr_minus_canonical": (
            right[prompt_id]["reward_model_score"]
            - left[prompt_id]["reward_model_score"]
        ),
        "kl_dr_minus_canonical": (
            right[prompt_id]["sampled_kl_token_mean"]
            - left[prompt_id]["sampled_kl_token_mean"]
        ),
        "length_dr_minus_canonical": (
            right[prompt_id]["response_length"]
            - left[prompt_id]["response_length"]
        ),
    } for prompt_id in left])
    paired.to_csv(root / "paired_heldout_differences.csv", index=False)
    comparisons = compare_normalization_responses(
        evaluations["norm_canonical"], evaluations["norm_dr_grpo"]
    )
    write_jsonl(root / "qualitative_comparisons.jsonl", comparisons[:32])
    save_json(root / "comparison_protocol.json", {
        "control": "norm_canonical",
        "treatment": "norm_dr_grpo",
        "shared_midpoint": canonical_protocol["midpoint_policy"],
        "shared_midpoint_sha256": canonical_protocol["midpoint_policy_sha256"],
        "matched_training_prompt_ids": canonical_protocol["prompt_ids"],
        "matched_heldout_prompt_ids": list(left),
        "matched_updates": canonical_protocol["updates"],
        "matched_num_generations": canonical_protocol["num_generations"],
        "matched_reward_model": canonical_protocol["reward_model"],
        "matched_kl_beta": canonical_protocol["kl_beta"],
        "matched_clip_epsilon": canonical_protocol["clip_epsilon"],
        "matched_generation_settings": canonical_protocol["generation"],
        "matched_max_completion_length": canonical_protocol["max_completion_length"],
        "matched_maximum_generation_budget": summary_rows[0][
            "maximum_generation_budget"
        ],
        "actual_generated_tokens_are_reported_separately": True,
        "only_intended_difference": "completion-length normalization denominator",
        "manual_quality_fields_are_intentionally_blank": True,
    })
    return summary_table, length_table


def build_available_evidence(config: dict) -> dict:
    """Build only tables whose required experiment artifacts are complete."""
    built = {}
    standard_dir = result_directory(config, "standard")
    if all((standard_dir / name).exists() for name in (
        "train_metrics.jsonl", "train_summary.json", "eval_metrics.json"
    )):
        built["standard"] = standard_summary(config)
        built["stability"] = standard_stability_diagnostics(config)
    normalization_files = (
        "run_config.json",
        "train_summary.json",
        "train_metrics.jsonl",
        "completion_metrics.jsonl",
        "eval_metrics.json",
        "generations.jsonl",
    )
    if all(
        all((result_directory(config, run) / name).exists()
            for name in normalization_files)
        for run in NORMALIZATION_RUNS
    ):
        built["normalization"] = normalization_evidence(config)
    return built
