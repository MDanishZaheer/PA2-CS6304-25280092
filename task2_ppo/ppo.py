# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Implement the masked token-level PPO objective used in Task 2."""

from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions.

    rewards, values, mask: [batch, response_steps]. Padding positions must have mask=0.
    The final valid response position bootstraps with zero.
    """
    if rewards.shape != values.shape or rewards.shape != mask.shape:
        raise ValueError("Rewards, values, and mask must have identical shapes.")
    if rewards.ndim != 2:
        raise ValueError("GAE expects [batch, response_steps] tensors.")
    if not 0.0 <= float(gamma) <= 1.0 or not 0.0 <= float(lam) <= 1.0:
        raise ValueError("GAE gamma and lambda must lie in [0, 1].")
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward."""
    if policy_logp.shape != ref_logp.shape or policy_logp.shape != response_mask.shape:
        raise ValueError("Policy log-probs, reference log-probs, and mask must match.")
    if task_reward.ndim != 1 or task_reward.shape[0] != policy_logp.shape[0]:
        raise ValueError("Task reward must contain one scalar per response.")
    if float(beta_kl) < 0:
        raise ValueError("KL beta must be non-negative.")
    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Return the conservative PPO loss and both clipping diagnostics."""
    if not (
        new_logp.shape == old_logp.shape == advantage.shape == mask.shape
    ):
        raise ValueError("PPO log-probs, advantages, and mask must have matching shapes.")
    if float(eps) <= 0:
        raise ValueError("PPO clip epsilon must be positive.")
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage

    # PPO maximizes the smaller surrogate. The starter incorrectly used maximum here.
    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    outside = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    affected = (surr2 < surr1).float()
    clip_fraction = masked_mean(outside, mask)
    affected_fraction = masked_mean(affected, mask)
    return (
        loss,
        ratio.detach(),
        clip_fraction.detach(),
        affected_fraction.detach(),
    )


def value_mse_loss(predicted_values, returns, mask):
    if predicted_values.shape != returns.shape or predicted_values.shape != mask.shape:
        raise ValueError("Predicted values, returns, and mask must have matching shapes.")
    return masked_mean((predicted_values - returns) ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    if advantages.shape != mask.shape:
        raise ValueError("Advantages and mask must have matching shapes.")
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask
