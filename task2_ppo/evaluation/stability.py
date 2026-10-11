# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate clearly defined optimization-stability statistics."""

from __future__ import annotations

import math

import numpy as np


def stability_summary(history: list[dict]) -> dict:
    """Summarize update variability; lower standard deviations are more stable."""
    if not history:
        raise ValueError("A stability summary requires at least one update.")
    policy_losses = np.asarray([row["policy_loss"] for row in history], dtype=float)
    gradients = np.asarray([row["policy_gradient_norm"] for row in history], dtype=float)
    rewards = np.asarray([row["effective_reward"] for row in history], dtype=float)
    finite = [
        math.isfinite(float(value))
        for row in history
        for value in (
            row["policy_loss"], row["value_loss"], row["policy_gradient_norm"]
        )
    ]
    return {
        "updates": len(history),
        "policy_loss_std": float(policy_losses.std(ddof=0)),
        "policy_gradient_norm_std": float(gradients.std(ddof=0)),
        "reward_step_change_std": float(
            np.diff(rewards).std(ddof=0) if len(rewards) > 1 else 0.0
        ),
        "finite_metric_fraction": float(np.mean(finite)),
    }
