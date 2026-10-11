# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Assemble Task 1 comparison tables from saved JSON and CSV results."""

from __future__ import annotations

import pandas as pd

from common.data import repo_path
from common.logging_utils import load_json


def dpo_summary_table(results_dir: str = "task1_dpo/results") -> pd.DataFrame:
    """Load standard and beta fork summaries with their unequal budgets visible."""
    root = repo_path(results_dir)
    rows = []
    for name, folder in [
        ("standard", root / "standard"),
        *((f"beta_{value:.2f}", root / "beta" / f"beta_{value:.2f}") for value in (0.03, 0.10, 0.30)),
    ]:
        metrics = load_json(folder / "eval_metrics.json")
        training = load_json(folder / "train_summary.json")
        rows.append({
            "run": name,
            "training_budget": "one epoch" if name == "standard" else "short 600-pair fork",
            "training_examples": training["training_examples"],
            "optimizer_updates": training["updates"],
            "beta": training["beta"],
            "heldout_pairs": metrics["preference_pairs"],
            "heldout_dpo_loss": metrics["heldout_dpo_loss"],
            "preference_accuracy": metrics["preference_accuracy"],
            "sampled_kl_token_mean": metrics["sampled_kl_token_mean"],
            "reward_model_score_mean": metrics["reward_model_score_mean"],
            "response_length_mean": metrics["response_length_mean"],
            "response_length_std": metrics["response_length_std"],
            "response_length_iqr": metrics["response_length_iqr"],
            "truncated_fraction": metrics["truncated_fraction"],
        })
    return pd.DataFrame(rows)


def length_tables(results_dir: str = "task1_dpo/results") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the saved per-stratum and common word-limit comparisons."""
    root = repo_path(results_dir) / "length"
    return (
        pd.read_csv(root / "stratum_metrics.csv"),
        pd.read_csv(root / "word_limit_metrics.csv"),
    )
