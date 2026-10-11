# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Create compact report figures from saved Task 4 tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task4_safety.core.paths import (
    POLICY_ORDER,
    category_directory,
    figures_directory,
    manual_directory,
    summary_directory,
)
from task4_safety.core.protocol import BEHAVIOR_LABELS


DISPLAY_NAMES = {"sft": "SFT", "dpo": "DPO", "ppo": "PPO", "grpo": "GRPO"}


def plot_policy_comparison(config: dict):
    """Plot the four required calibration rates plus ambiguity."""
    table = pd.read_csv(summary_directory(config) / "policy_metrics.csv")
    table = table.set_index("policy").loc[list(POLICY_ORDER)].reset_index()
    metrics = [
        ("safe_answer_rate", "Safe answer"),
        ("safe_over_refusal_rate", "Safe over-refusal"),
        ("unsafe_compliance_rate", "Unsafe compliance"),
        ("unsafe_justified_refusal_rate", "Justified refusal"),
        ("ambiguous_judge_rate", "Ambiguous"),
    ]
    x = np.arange(len(table))
    width = 0.16
    fig, ax = plt.subplots(figsize=(12, 5))
    for index, (column, label) in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, table[column], width, label=label)
    ax.set_xticks(x, [DISPLAY_NAMES[name] for name in table["policy"]])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("XSTest safety calibration by frozen policy")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    destination = figures_directory(config) / "policy_safety_comparison.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def plot_response_lengths(config: dict):
    """Compare response length separately for safe and unsafe prompts."""
    table = pd.read_csv(summary_directory(config) / "response_length_metrics.csv")
    x = np.arange(len(POLICY_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, benchmark_class in enumerate(("SAFE", "UNSAFE")):
        selected = (
            table[table["benchmark_class"] == benchmark_class]
            .set_index("policy")
            .loc[list(POLICY_ORDER)]
        )
        offset = (index - 0.5) * width
        ax.bar(
            x + offset,
            selected["mean_response_length"],
            width,
            label=benchmark_class.title(),
        )
    ax.set_xticks(x, [DISPLAY_NAMES[name] for name in POLICY_ORDER])
    ax.set_ylabel("Mean response length (tokens)")
    ax.set_title("Response length by XSTest benchmark class")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    destination = figures_directory(config) / "response_length_comparison.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def plot_category_behavior(config: dict):
    """Show every judge-label rate for every XSTest category and policy."""
    table = pd.read_csv(
        category_directory(config) / "category_label_distribution.csv"
    )
    categories = sorted(table["type"].unique())
    fig, axes = plt.subplots(
        1,
        len(POLICY_ORDER),
        figsize=(18, max(6, len(categories) * 0.36)),
        sharey=True,
    )
    image = None
    for ax, policy_name in zip(axes, POLICY_ORDER):
        selected = table[table["policy"] == policy_name]
        pivot = selected.pivot(index="type", columns="ai_label", values="rate")
        pivot = pivot.reindex(index=categories, columns=BEHAVIOR_LABELS, fill_value=0)
        image = ax.imshow(
            pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto"
        )
        ax.set_title(DISPLAY_NAMES[policy_name])
        ax.set_xticks(
            range(len(BEHAVIOR_LABELS)),
            [label.replace("_", "\n") for label in BEHAVIOR_LABELS],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(range(len(categories)), categories, fontsize=8)
    fig.colorbar(image, ax=axes, label="Within-category rate", fraction=0.015)
    fig.suptitle("AI-judge label distribution across XSTest categories")
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.20, top=0.92, wspace=0.08)
    destination = figures_directory(config) / "category_behavior.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def plot_manual_confusion(config: dict):
    """Plot the manual-versus-AI confusion matrix after annotation."""
    path = manual_directory(config) / "confusion_matrix.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    matrix = (
        table.pivot(index="manual_label", columns="ai_label", values="count")
        .reindex(index=BEHAVIOR_LABELS, columns=BEHAVIOR_LABELS, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    for row in range(len(BEHAVIOR_LABELS)):
        for column in range(len(BEHAVIOR_LABELS)):
            ax.text(
                column,
                row,
                int(matrix.iloc[row, column]),
                ha="center",
                va="center",
            )
    short = [label.replace("_", "\n") for label in BEHAVIOR_LABELS]
    ax.set_xticks(range(len(short)), short, rotation=45, ha="right")
    ax.set_yticks(range(len(short)), short)
    ax.set_xlabel("AI judge label")
    ax.set_ylabel("Manual label")
    ax.set_title("Manual audit versus AI safety judge")
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    destination = figures_directory(config) / "manual_ai_confusion.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def save_available_figures(config: dict) -> list:
    """Build automated figures and the manual figure when labels are complete."""
    destinations = [
        plot_policy_comparison(config),
        plot_response_lengths(config),
        plot_category_behavior(config),
    ]
    manual = plot_manual_confusion(config)
    if manual is not None:
        destinations.append(manual)
    return destinations
