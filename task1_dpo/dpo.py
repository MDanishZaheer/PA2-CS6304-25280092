# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate the DPO objective from response-only sequence log-probabilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return the DPO loss and reference-relative preference diagnostics."""
    if beta <= 0:
        raise ValueError("DPO beta must be positive.")
    if not (
        policy_chosen_logp.shape
        == policy_rejected_logp.shape
        == ref_chosen_logp.shape
        == ref_rejected_logp.shape
    ):
        raise ValueError("Chosen and rejected log-probability tensors must have matching shapes.")
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    preference_margin = policy_margin - ref_margin
    logits = beta * preference_margin

    loss = -F.logsigmoid(logits).mean()
    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "preference_margin_mean": preference_margin.detach().mean(),
        "preference_accuracy": (preference_margin > 0).float().mean().detach(),
    }
