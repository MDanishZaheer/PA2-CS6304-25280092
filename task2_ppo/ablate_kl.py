# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Run matched PPO continuations for the KL-pressure study."""

from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task2_ppo.analysis.plots import save_task2_figures
from task2_ppo.core.paths import result_directory
from task2_ppo.core.protocol import validate_shared_protocol
from task2_ppo.core.trainer import train_ppo_continuation
from task2_ppo.evaluate import run_evaluation
from task2_ppo.evaluation.qualitative import compare_condition_responses
from task2_ppo.evaluation.stability import stability_summary


def run_kl_study(
    config_path: str = "configs/ppo.yaml", overwrite: bool = False
) -> pd.DataFrame:
    """Train and evaluate every fixed-beta fork from the same midpoint."""
    config = load_yaml(config_path)
    validate_shared_protocol(config)
    summaries = []
    records_by_run = {}
    for beta in [float(value) for value in config["kl_values"]]:
        run_name = f"kl_beta_{beta:.2f}"
        training = train_ppo_continuation(
            config_path,
            run_name=run_name,
            updates=int(config["fork_updates"]),
            clip_epsilon=float(config["clip_epsilon"]),
            kl_beta=beta,
            overwrite=overwrite,
        )
        evaluation = run_evaluation(
            config_path, name=run_name, overwrite=overwrite
        )
        run_directory = result_directory(config, run_name)
        history = read_jsonl(run_directory / "train_metrics.jsonl")
        stability = stability_summary(history)
        summaries.append({
            "run": run_name,
            "kl_beta": beta,
            "updates": training["updates"],
            "maximum_generation_budget": (
                training["updates"]
                * training["prompts_per_update"]
                * int(config["max_response_length"])
            ),
            "generated_tokens": training["generated_tokens"],
            "heldout_reward": evaluation["reward_model_score_mean"],
            "heldout_kl": evaluation["sampled_kl_token_mean"],
            "heldout_entropy": evaluation["entropy_token_mean"],
            "heldout_response_length": evaluation["response_length_mean"],
            **stability,
        })
        records_by_run[run_name] = read_jsonl(run_directory / "generations.jsonl")

    frame = pd.DataFrame(summaries).sort_values("kl_beta")
    study_dir = repo_path(config["results_dir"]) / "kl"
    study_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(study_dir / "summary.csv", index=False)
    save_json(study_dir / "summary.json", frame.to_dict(orient="records"))
    comparisons = compare_condition_responses(records_by_run, "kl_beta_0.10")
    write_jsonl(study_dir / "qualitative_comparison.jsonl", comparisons)
    save_task2_figures(config["results_dir"])
    return frame


def main() -> None:
    """Parse command-line arguments and run the KL-pressure study."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(run_kl_study(args.config, args.overwrite))


if __name__ == "__main__":
    main()
