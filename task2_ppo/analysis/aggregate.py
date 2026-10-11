# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Assemble Task 2 result tables without rerunning model inference."""

from __future__ import annotations

import pandas as pd

from common.data import repo_path
from common.logging_utils import load_json


def standard_summary_table(
    results_dir: str = "task2_ppo/results",
) -> pd.DataFrame:
    """Combine standard continuation cost and held-out behavior in one row."""
    directory = repo_path(results_dir) / "standard"
    training = load_json(directory / "train_summary.json")
    evaluation = load_json(directory / "eval_metrics.json")
    return pd.DataFrame([{
        "run": "standard",
        "updates": training["updates"],
        "maximum_generation_budget": training["maximum_generation_budget"],
        "generated_tokens": training["generated_tokens"],
        "clip_epsilon": training["clip_epsilon"],
        "kl_beta": training["kl_beta"],
        "heldout_reward": evaluation["reward_model_score_mean"],
        "heldout_kl": evaluation["sampled_kl_token_mean"],
        "heldout_entropy": evaluation["entropy_token_mean"],
        "heldout_response_length": evaluation["response_length_mean"],
        "peak_vram_gib": training["peak_vram_gib"],
        "wall_time_seconds": training["wall_time_seconds"],
    }])


def clipping_tables(
    results_dir: str = "task2_ppo/results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached-rollout geometry and matched clipping-fork outcomes."""
    directory = repo_path(results_dir) / "clipping"
    return (
        pd.read_csv(directory / "cached_rollout_metrics.csv"),
        pd.read_csv(directory / "summary.csv"),
    )


def kl_summary_table(results_dir: str = "task2_ppo/results") -> pd.DataFrame:
    """Load the matched KL-pressure fork outcomes."""
    return pd.read_csv(repo_path(results_dir) / "kl" / "summary.csv")
