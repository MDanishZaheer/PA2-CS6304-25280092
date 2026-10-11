# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Small, testable functions that define the Task 3 GRPO objective."""

from __future__ import annotations

import torch

from common.metrics import masked_mean


def _validate_group_inputs(rewards: torch.Tensor, group_ids: torch.Tensor) -> None:
    """Validate the one-reward/one-group-label representation."""
    if rewards.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("rewards and group_ids must both be one-dimensional.")
    if rewards.numel() == 0 or rewards.numel() != group_ids.numel():
        raise ValueError("Every non-empty reward vector needs one group ID per item.")
    if not torch.isfinite(rewards).all():
        raise ValueError("GRPO rewards must be finite.")


def group_relative_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Standardize rewards independently inside every prompt group.

    An uninformative group has reward standard deviation at or below ``eps``.
    Its advantages are deliberately zero, so it contributes no reward-gradient
    signal. This fixes the starter bug that normalized all prompt groups together.
    """
    _validate_group_inputs(rewards, group_ids)
    if eps <= 0:
        raise ValueError("eps must be positive.")
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    for group_id in torch.unique(group_ids, sorted=True):
        selected = group_ids == group_id
        group_rewards = rewards[selected].float()
        standard_deviation = group_rewards.std(unbiased=False)
        if float(standard_deviation) > eps:
            advantages[selected] = (
                group_rewards - group_rewards.mean()
            ) / standard_deviation
    return advantages


def normalization_denominators(
    token_mask: torch.Tensor,
    loss_type: str,
    max_completion_length: int | None = None,
) -> torch.Tensor:
    """Return the per-completion divisor used by one GRPO variant."""
    if token_mask.ndim != 2:
        raise ValueError("token_mask must have shape [completions, tokens].")
    if loss_type == "grpo":
        return token_mask.sum(-1).clamp_min(1.0)
    if loss_type == "dr_grpo":
        if max_completion_length is None or int(max_completion_length) <= 0:
            raise ValueError("dr_grpo requires a positive max_completion_length.")
        return torch.full(
            (token_mask.shape[0],),
            float(max_completion_length),
            dtype=token_mask.dtype,
            device=token_mask.device,
        )
    raise ValueError(f"Unknown loss_type={loss_type!r}.")


def gradient_allocation_proxy(
    seq_adv: torch.Tensor,
    token_mask: torch.Tensor,
    loss_type: str,
    max_completion_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure the objective weight assigned per token and per completion.

    These are transparent analytical proxies, not measured parameter-gradient
    norms. The sequence proxy sums the absolute token coefficients and therefore
    exposes how the two normalizations allocate learning signal by response length.
    """
    if seq_adv.ndim != 1 or seq_adv.shape[0] != token_mask.shape[0]:
        raise ValueError("seq_adv needs one scalar per completion.")
    denominators = normalization_denominators(
        token_mask, loss_type, max_completion_length
    )
    active = (token_mask.sum(-1) > 0).float()
    per_token = seq_adv.abs().float() / denominators.float() * active
    per_sequence = per_token * token_mask.sum(-1).float()
    return per_token, per_sequence


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    if not (
        new_logp.shape
        == old_logp.shape
        == token_mask.shape
        == ref_logp.shape
    ):
        raise ValueError("All token-level GRPO tensors must have the same shape.")
    if new_logp.ndim != 2 or seq_adv.shape != (new_logp.shape[0],):
        raise ValueError("seq_adv needs one scalar per sampled completion.")
    if float(eps) <= 0 or float(beta) < 0:
        raise ValueError("Clip epsilon must be positive and beta non-negative.")

    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    denominators = normalization_denominators(
        token_mask, loss_type, max_completion_length
    )
    per_sequence = (objective * token_mask).sum(-1) / denominators
    policy_term = -per_sequence.mean()

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "kl_penalty": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "affected_token_fraction": masked_mean(
            (s2 < s1).float(), token_mask
        ).detach(),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    """Zero the complete loss mask for responses that hit the generation cap."""
    if token_mask.ndim != 2:
        raise ValueError("token_mask must have shape [completions, tokens].")
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    if truncated.shape != (token_mask.shape[0],):
        raise ValueError("truncated needs one flag per completion.")
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
