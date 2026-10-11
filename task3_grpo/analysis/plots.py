# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Create the compact figures required for Task 3 discussion."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.data import read_jsonl
from task3_grpo.core.paths import figures_directory, group_size_directory, result_directory


def _line_panel(axis, frame, column: str, title: str) -> None:
    axis.plot(frame["update"], frame[column], marker="o", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Update")
    axis.grid(alpha=0.25)


def plot_standard(config: dict) -> None:
    history = pd.DataFrame(read_jsonl(
        result_directory(config, "standard") / "train_metrics.jsonl"
    ))
    figure, axes = plt.subplots(4, 2, figsize=(12, 14))
    specifications = [
        ("reward_mean", "Reward"),
        ("sampled_kl_token_mean", "Sampled KL from reference"),
        ("within_group_reward_std_mean", "Within-group reward std"),
        ("uninformative_group_fraction", "Uninformative-group fraction"),
        ("policy_loss", "Policy loss"),
        ("gradient_norm", "Gradient norm"),
        ("entropy_token_mean", "Exact token entropy"),
        ("response_length_mean", "Response length"),
    ]
    for axis, (column, title) in zip(axes.flat, specifications):
        _line_panel(axis, history, column, title)
    figure.tight_layout()
    directory = figures_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / "standard_training.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_group_size(config: dict) -> None:
    directory = group_size_directory(config)
    overall = pd.read_csv(directory / "summary.csv")
    by_difficulty = pd.read_csv(directory / "difficulty_summary.csv")
    columns = [
        ("informative_group_rate", "Informative-group rate"),
        ("mean_within_group_reward_std", "Mean within-group reward std"),
        ("variance_group_relative_signal", "Relative-signal variance"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (column, title) in zip(axes, columns):
        axis.plot(overall["k"], overall[column], marker="o")
        axis.set_xticks([2, 4, 8])
        axis.set_xlabel("Group size K")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figures = figures_directory(config)
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "group_size_overall.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (column, title) in zip(axes, columns):
        for label in ("hard", "medium", "easy"):
            rows = by_difficulty[by_difficulty["difficulty"] == label]
            axis.plot(rows["k"], rows[column], marker="o", label=label)
        axis.set_xticks([2, 4, 8])
        axis.set_xlabel("Group size K")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(
        figures / "group_size_by_difficulty.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def plot_normalization(config: dict) -> None:
    root = result_directory(config, "norm_canonical").parent
    summary = pd.read_csv(root / "summary.csv")
    length = pd.read_csv(root / "length_conditioned_metrics.csv")
    labels = ["Canonical GRPO", "Dr-GRPO"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("heldout_reward", "Held-out reward"),
        ("heldout_sampled_kl", "Held-out sampled KL"),
        ("heldout_response_length", "Held-out response length"),
    ]
    for axis, (column, title) in zip(axes, metrics):
        axis.bar(labels, summary[column])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figures = figures_directory(config)
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        figures / "normalization_heldout.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run_name, label in zip(summary["run_name"], labels):
        rows = length[length["run_name"] == run_name]
        axes[0].plot(
            rows["length_bin"], rows["mean_per_token_gradient_weight"],
            marker="o", label=label,
        )
        axes[1].plot(
            rows["length_bin"], rows["mean_sequence_gradient_weight_proxy"],
            marker="o", label=label,
        )
    axes[0].set_title("Per-token objective weight")
    axes[1].set_title("Per-completion weight proxy")
    for axis in axes:
        axis.set_xlabel("Fixed response-length bin")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(
        figures / "normalization_length_allocation.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_available_figures(config: dict) -> None:
    standard = result_directory(config, "standard") / "train_metrics.jsonl"
    if standard.exists():
        plot_standard(config)
    group = group_size_directory(config) / "summary.csv"
    if group.exists():
        plot_group_size(config)
    normalization = result_directory(config, "norm_canonical").parent / "summary.csv"
    if normalization.exists():
        plot_normalization(config)
