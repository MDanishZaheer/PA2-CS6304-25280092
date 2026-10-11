# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Create compact report figures from saved Task 5 tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task5_feedback.core.paths import (
    POLICY_ORDER,
    diagnostics_directory,
    domain_directory,
    figures_directory,
)


DISPLAY = {"sft": "SFT", "rlvr": "RLVR", "rlaif": "RLAIF"}


def plot_domain_accuracy(config: dict):
    gsm = pd.read_csv(domain_directory(config, "gsm") / "policy_metrics.csv")
    transfer = pd.read_csv(
        domain_directory(config, "transfer") / "policy_metrics.csv"
    )
    x = np.arange(len(POLICY_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, (label, table) in enumerate((("GSM8K", gsm), ("SVAMP", transfer))):
        values = table.set_index("policy").loc[list(POLICY_ORDER), "exact_accuracy"]
        ax.bar(x + (index - 0.5) * width, values, width, label=label)
    ax.set_xticks(x, [DISPLAY[name] for name in POLICY_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact final-answer accuracy")
    ax.set_title("In-domain and transfer accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    destination = figures_directory(config) / "domain_accuracy.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def plot_ai_pairwise(config: dict):
    gsm = pd.read_csv(domain_directory(config, "gsm") / "policy_metrics.csv")
    transfer = pd.read_csv(
        domain_directory(config, "transfer") / "policy_metrics.csv"
    )
    policies = ("rlvr", "rlaif")
    x = np.arange(len(policies))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for index, (label, table) in enumerate((("GSM8K", gsm), ("SVAMP", transfer))):
        values = table.set_index("policy").loc[
            list(policies), "ai_pairwise_score_vs_sft"
        ]
        ax.bar(x + (index - 0.5) * width, values, width, label=label)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Tie level")
    ax.set_xticks(x, [DISPLAY[name] for name in policies])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AI pairwise score versus SFT")
    ax.set_title("Fixed-judge pairwise behavior")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    destination = figures_directory(config) / "ai_pairwise_vs_sft.png"
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def plot_diagnostic_outcomes(config: dict):
    table = pd.read_csv(
        diagnostics_directory(config) / "controlled_pair_metrics.csv"
    )
    categories = list(dict.fromkeys(table["pair_category"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    colors = ("#2a9d8f", "#e9c46a", "#e76f51")
    columns = ("better_response_rate", "tie_rate", "wrong_preference_rate")
    labels = ("Better", "Tie", "Wrong")
    for ax, mechanism in zip(
        axes, ("RLVR_exact_verifier", "RLAIF_AI_judge")
    ):
        selected = table[table["mechanism"] == mechanism].set_index(
            "pair_category"
        ).loc[categories]
        bottom = np.zeros(len(categories))
        for column, label, color in zip(columns, labels, colors):
            values = selected[column].to_numpy()
            ax.bar(categories, values, bottom=bottom, label=label, color=color)
            bottom += values
        ax.set_title(mechanism.replace("_", " "))
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Fraction of controlled pairs")
    axes[1].legend()
    fig.suptitle("Feedback behavior on controlled perturbations")
    fig.tight_layout()
    destination = figures_directory(config) / "diagnostic_outcomes.png"
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def plot_variant_scores(config: dict):
    table = pd.read_csv(
        diagnostics_directory(config) / "variant_reward_scores.csv"
    )
    x = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    axes[0].bar(x, table["exact_reward_mean"], color="#457b9d")
    axes[1].bar(x, table["ai_group_reward_mean"], color="#e76f51")
    labels = [value.replace("_", "\n") for value in table["variant_category"]]
    for ax, title in zip(
        axes, ("RLVR exact reward", "RLAIF-style AI group reward")
    ):
        ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean score (scales are not calibrated utilities)")
    fig.tight_layout()
    destination = figures_directory(config) / "diagnostic_variant_scores.png"
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def save_all_figures(config: dict) -> list:
    return [
        plot_domain_accuracy(config),
        plot_ai_pairwise(config),
        plot_diagnostic_outcomes(config),
        plot_variant_scores(config),
    ]
