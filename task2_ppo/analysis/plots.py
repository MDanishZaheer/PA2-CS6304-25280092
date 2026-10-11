# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Create Task 2 figures only from saved machine-readable results."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.data import read_jsonl, repo_path
from task2_ppo.analysis.aggregate import clipping_tables, kl_summary_table


def plot_standard_trajectories(results_dir: str = "task2_ppo/results"):
    """Plot every required standard PPO continuation diagnostic."""
    history = pd.DataFrame(
        read_jsonl(repo_path(results_dir) / "standard" / "train_metrics.jsonl")
    )
    if history.empty:
        raise ValueError("The standard PPO training history is empty.")
    panels = [
        ("effective_reward", "Mean learned reward"),
        ("sampled_kl_token_mean", "Sampled KL / token"),
        ("policy_loss", "Policy loss"),
        ("value_loss", "Value loss (MSE)"),
        ("entropy", "Policy entropy / token"),
        ("clip_fraction", "Clip fraction"),
        ("policy_gradient_norm", "Policy gradient norm"),
        ("response_length", "Mean response tokens"),
    ]
    figure, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for axis, (column, label) in zip(axes.ravel(), panels):
        axis.plot(
            history["update"],
            history[column],
            marker="o",
            markersize=3,
            label="Effective reward" if column == "effective_reward" else None,
        )
        if column == "effective_reward":
            axis.plot(
                history["update"],
                history["raw_reward"],
                linestyle="--",
                label="Raw reward-model score",
            )
            axis.legend(fontsize=8)
        axis.set(ylabel=label)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("PPO update")
    axes[-1, 1].set_xlabel("PPO update")
    figure.suptitle("Standard PPO continuation diagnostics")
    figure.tight_layout()
    return figure


def plot_cached_clipping(results_dir: str = "task2_ppo/results"):
    """Plot the immediate geometric effect of epsilon on one fixed batch."""
    cached, _ = clipping_tables(results_dir)
    cached = cached.sort_values("clip_epsilon")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].plot(
        cached["clip_epsilon"],
        cached["clip_fraction"],
        marker="o",
        label="Ratio outside interval",
    )
    axes[0].plot(
        cached["clip_epsilon"],
        cached["affected_token_fraction"],
        marker="s",
        label="Affected by clipping",
    )
    axes[0].set(
        xlabel="Clip epsilon",
        ylabel="Fraction of valid tokens",
        ylim=(0, 1),
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        cached["clip_epsilon"], cached["clipped_surrogate"], marker="o"
    )
    axes[1].set(xlabel="Clip epsilon", ylabel="Mean clipped surrogate")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Cached-rollout clipping geometry")
    figure.tight_layout()
    return figure


def plot_clipping_forks(results_dir: str = "task2_ppo/results"):
    """Plot held-out behavior and stability across matched epsilon forks."""
    _, frame = clipping_tables(results_dir)
    frame = frame.sort_values("clip_epsilon")
    panels = [
        ("heldout_reward", "Mean reward-model score"),
        ("heldout_kl", "Sampled KL / token"),
        ("heldout_entropy", "Entropy / token"),
        ("heldout_response_length", "Mean response tokens"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (column, label) in zip(axes.ravel(), panels):
        axis.plot(frame["clip_epsilon"], frame[column], marker="o")
        axis.set(xlabel="Clip epsilon", ylabel=label)
        axis.grid(alpha=0.25)
    figure.suptitle("Matched short PPO clipping forks")
    figure.tight_layout()
    return figure


def plot_clipping_trajectories(results_dir: str = "task2_ppo/results"):
    """Compare optimization trajectories across the three epsilon forks."""
    root = repo_path(results_dir) / "clipping"
    panels = [
        ("policy_loss", "Policy loss"),
        ("value_loss", "Value loss (MSE)"),
        ("policy_gradient_norm", "Policy gradient norm"),
        ("clip_fraction", "Clip fraction"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for epsilon in (0.05, 0.20, 0.50):
        name = f"clip_eps_{epsilon:.2f}"
        history = pd.DataFrame(read_jsonl(root / name / "train_metrics.jsonl"))
        for axis, (column, label) in zip(axes.ravel(), panels):
            axis.plot(history["update"], history[column], marker="o", label=name)
            axis.set(ylabel=label)
            axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("PPO update")
    axes[-1, 1].set_xlabel("PPO update")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Clipping-fork optimization trajectories")
    figure.tight_layout()
    return figure


def plot_kl_forks(results_dir: str = "task2_ppo/results"):
    """Plot required held-out metrics across matched KL-pressure forks."""
    frame = kl_summary_table(results_dir).sort_values("kl_beta")
    panels = [
        ("heldout_reward", "Mean reward-model score"),
        ("heldout_kl", "Sampled KL / token"),
        ("heldout_entropy", "Entropy / token"),
        ("heldout_response_length", "Mean response tokens"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (column, label) in zip(axes.ravel(), panels):
        axis.plot(frame["kl_beta"], frame[column], marker="o")
        axis.set(xlabel="KL beta", ylabel=label)
        axis.grid(alpha=0.25)
    figure.suptitle("Matched short PPO KL-pressure forks")
    figure.tight_layout()
    return figure


def plot_kl_trajectories(results_dir: str = "task2_ppo/results"):
    """Show which behavior changes first across KL-pressure forks."""
    root = repo_path(results_dir) / "kl"
    panels = [
        ("raw_reward", "Mean learned reward"),
        ("sampled_kl_token_mean", "Sampled KL / token"),
        ("entropy", "Policy entropy / token"),
        ("response_length", "Mean response tokens"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for beta in (0.00, 0.10, 0.20):
        name = f"kl_beta_{beta:.2f}"
        history = pd.DataFrame(read_jsonl(root / name / "train_metrics.jsonl"))
        for axis, (column, label) in zip(axes.ravel(), panels):
            axis.plot(history["update"], history[column], marker="o", label=name)
            axis.set(ylabel=label)
            axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("PPO update")
    axes[-1, 1].set_xlabel("PPO update")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("KL-pressure fork trajectories")
    figure.tight_layout()
    return figure


def save_task2_figures(results_dir: str = "task2_ppo/results") -> dict[str, str]:
    """Save all figures whose required result files are available."""
    root = repo_path(results_dir)
    figure_directory = root / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    builders = {
        "standard_trajectories.png": (
            root / "standard" / "train_metrics.jsonl",
            plot_standard_trajectories,
        ),
        "cached_clipping.png": (
            root / "clipping" / "cached_rollout_metrics.csv",
            plot_cached_clipping,
        ),
        "clipping_forks.png": (
            root / "clipping" / "summary.csv",
            plot_clipping_forks,
        ),
        "clipping_trajectories.png": (
            root / "clipping" / "summary.csv",
            plot_clipping_trajectories,
        ),
        "kl_forks.png": (root / "kl" / "summary.csv", plot_kl_forks),
        "kl_trajectories.png": (
            root / "kl" / "summary.csv",
            plot_kl_trajectories,
        ),
    }
    saved = {}
    for filename, (required_path, builder) in builders.items():
        if required_path.exists():
            figure = builder(results_dir)
            destination = figure_directory / filename
            figure.savefig(destination, dpi=180, bbox_inches="tight")
            plt.close(figure)
            saved[filename] = str(destination)
    return saved
