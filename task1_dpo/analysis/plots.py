# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Plot saved Task 1 training and evaluation results."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.data import read_jsonl, repo_path
from task1_dpo.analysis.aggregate import dpo_summary_table, length_tables


def plot_training_history(results_dir: str = "task1_dpo/results", run_name: str = "standard"):
    """Plot update loss and preference accuracy from the saved history."""
    path = repo_path(results_dir) / run_name / "train_metrics.jsonl"
    frame = pd.DataFrame(read_jsonl(path))
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(frame["update"], frame["train_loss"])
    axes[0].set(xlabel="Optimizer update", ylabel="DPO loss", title=run_name)
    axes[1].plot(frame["update"], frame["train_preference_accuracy"])
    axes[1].set(xlabel="Optimizer update", ylabel="Preference accuracy", ylim=(0, 1))
    figure.tight_layout()
    return figure


def plot_beta_comparison(results_dir: str = "task1_dpo/results"):
    """Plot all required generated-output metrics across short beta forks."""
    frame = dpo_summary_table(results_dir)
    forks = frame.loc[frame["run"] != "standard"].sort_values("beta")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    axes[0].plot(forks["beta"], forks["preference_accuracy"], marker="o")
    axes[0].set(xlabel="DPO beta", ylabel="Held-out preference accuracy")
    axes[1].plot(forks["beta"], forks["sampled_kl_token_mean"], marker="o", label="KL")
    axes[1].set(xlabel="DPO beta", ylabel="Sampled KL per response token")
    axes[2].plot(forks["beta"], forks["reward_model_score_mean"], marker="o")
    axes[2].set(xlabel="DPO beta", ylabel="Mean reward-model score")
    axes[3].plot(forks["beta"], forks["response_length_mean"], marker="o")
    axes[3].set(xlabel="DPO beta", ylabel="Mean generated response tokens")
    figure.tight_layout()
    return figure


def plot_length_strata(results_dir: str = "task1_dpo/results"):
    """Plot standard versus balanced accuracy by supplied length stratum."""
    strata, _ = length_tables(results_dir)
    pivot = strata.pivot(index="length_stratum", columns="policy", values="preference_accuracy")
    pivot = pivot.loc[["preferred_longer", "length_matched", "rejected_longer"]]
    pivot = pivot[["standard", "length_balanced"]]
    axes = pivot.plot(kind="bar", figsize=(8, 3.5), rot=0).get_figure()
    axes.axes[0].set(ylabel="Held-out preference accuracy", xlabel="Length stratum", ylim=(0, 1))
    axes.tight_layout()
    return axes


def plot_word_limit_comparison(results_dir: str = "task1_dpo/results"):
    """Plot response length and word-limit compliance on the common prompts."""
    _, word_limits = length_tables(results_dir)
    frame = word_limits.set_index("policy").loc[["standard", "length_balanced"]]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    frame["response_length_mean"].plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set(xlabel="Policy", ylabel="Mean generated response tokens")
    frame["word_limit_compliance"].plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set(xlabel="Policy", ylabel="Word-limit compliance", ylim=(0, 1))
    figure.tight_layout()
    return figure
