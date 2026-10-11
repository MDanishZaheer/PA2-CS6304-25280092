# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Join earlier alignment metrics to Task 4 without claiming causal effects."""

from __future__ import annotations

import pandas as pd

from common.data import repo_path
from common.logging_utils import load_json
from task4_safety.core.paths import POLICY_ORDER, summary_directory


SOURCE_RESULTS = {
    "dpo": "task1_dpo/results/standard/eval_metrics.json",
    "ppo": "task2_ppo/results/standard/eval_metrics.json",
    "grpo": "task3_grpo/results/standard/eval_metrics.json",
}


def build_cross_task_table(config: dict, safety_metrics: pd.DataFrame) -> pd.DataFrame:
    """Save traceable Task 1-3 metric context beside Task 4 safety rates.

    PPO and GRPO use the same held-out RL prompt pool. DPO uses its own held-out
    preference set, so its reward value is useful context but is not a strictly
    matched cross-policy comparison.
    """
    safety = safety_metrics.set_index("policy")
    records = []
    for policy_name in POLICY_ORDER:
        record = {
            "policy": policy_name,
            "source_task": {
                "sft": "starting policy",
                "dpo": "Task 1 standard DPO",
                "ppo": "Task 2 standard PPO",
                "grpo": "Task 3 standard GRPO",
            }[policy_name],
            "source_metrics_file": "",
            "source_eval_prompts": None,
            "preference_accuracy": None,
            "reward_model_score_mean": None,
            "sampled_kl_token_mean": None,
            "source_response_length_mean": None,
            "comparison_scope": (
                "No earlier aligned-policy metric for the untouched SFT baseline."
            ),
        }
        if policy_name in SOURCE_RESULTS:
            source_path = repo_path(SOURCE_RESULTS[policy_name])
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Run the standard {policy_name.upper()} evaluation first: "
                    f"{source_path}"
                )
            metrics = load_json(source_path)
            record.update({
                "source_metrics_file": SOURCE_RESULTS[policy_name],
                "source_eval_prompts": metrics.get(
                    "evaluation_prompts", metrics.get("generated_prompts")
                ),
                "preference_accuracy": metrics.get("preference_accuracy"),
                "reward_model_score_mean": metrics.get(
                    "reward_model_score_mean"
                ),
                "sampled_kl_token_mean": metrics.get(
                    "sampled_kl_token_mean"
                ),
                "source_response_length_mean": metrics.get(
                    "response_length_mean"
                ),
                "comparison_scope": (
                    "DPO uses its held-out preference set; do not treat its "
                    "reward value as strictly matched to PPO/GRPO."
                    if policy_name == "dpo"
                    else "PPO and GRPO use the same held-out RL prompt pool."
                ),
            })
        record.update({
            "safe_answer_rate": float(safety.loc[policy_name, "safe_answer_rate"]),
            "safe_over_refusal_rate": float(
                safety.loc[policy_name, "safe_over_refusal_rate"]
            ),
            "unsafe_compliance_rate": float(
                safety.loc[policy_name, "unsafe_compliance_rate"]
            ),
            "unsafe_justified_refusal_rate": float(
                safety.loc[policy_name, "unsafe_justified_refusal_rate"]
            ),
            "ambiguous_judge_rate": float(
                safety.loc[policy_name, "ambiguous_judge_rate"]
            ),
            "task4_response_length_mean": float(
                safety.loc[policy_name, "mean_response_length"]
            ),
        })
        records.append(record)
    table = pd.DataFrame(records)
    destination = summary_directory(config) / "cross_task_alignment_safety.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return table
