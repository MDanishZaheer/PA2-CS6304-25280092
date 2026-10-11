# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Collect on-policy PPO rollouts and update the actor and critic."""

from __future__ import annotations

import numpy as np
import torch

from common.data import prompt_messages
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.metrics import masked_mean
from common.models import reference_mode
from task2_ppo.core.batching import response_token_values, token_entropy
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def _independent_reward_scores(
    reward_model,
    reward_tokenizer,
    prompts: list[list[dict]],
    responses: list[str],
    max_length: int,
) -> torch.Tensor:
    """Score each response alone so dynamic padding cannot change its score."""
    values = []
    for messages, response in zip(prompts, responses):
        score = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            [messages],
            [response],
            max_length=max_length,
        )
        values.append(float(score[0]))
    device = next(reward_model.parameters()).device
    scores = torch.tensor(values, dtype=torch.float32, device=device)
    if not torch.isfinite(scores).all():
        raise ValueError("The reward model produced a non-finite score.")
    return scores


def collect_rollout(
    policy,
    value_model,
    reward_model,
    tokenizer,
    reward_tokenizer,
    prompt_rows: list[dict],
    config: dict,
) -> dict:
    """Generate one on-policy batch and freeze its old-policy targets."""
    messages = [prompt_messages(row) for row in prompt_rows]
    was_cache_enabled = bool(getattr(policy.config, "use_cache", False))
    policy.config.use_cache = True
    policy.eval()
    generated = batch_generate(
        policy,
        tokenizer,
        messages,
        max_prompt_length=int(config["max_prompt_length"]),
        max_new_tokens=int(config["max_response_length"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
        do_sample=bool(config["generation"]["do_sample"]),
    )
    policy.config.use_cache = was_cache_enabled
    with torch.no_grad():
        old_logp, old_logits = response_token_logprobs(
            policy,
            generated["sequences"],
            generated["attention_mask"],
            generated["prompt_width"],
            generated["response_ids"],
        )
        rollout_entropy = token_entropy(old_logits, generated["response_mask"])
        del old_logits
        with reference_mode(policy):
            reference_logp, reference_logits = response_token_logprobs(
                policy,
                generated["sequences"],
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
            del reference_logits
    policy.train()

    value_model.eval()
    with torch.no_grad():
        old_values = response_token_values(
            value_model,
            generated["sequences"],
            generated["attention_mask"],
            generated["prompt_width"],
            generated["response_ids"].shape[1],
        )
    value_model.train()

    raw_rewards = _independent_reward_scores(
        reward_model,
        reward_tokenizer,
        messages,
        generated["responses"],
        int(config["reward_max_length"]),
    ).to(old_logp.device)
    terminated = torch.tensor(
        generated["terminated_with_eos"], dtype=torch.bool, device=old_logp.device
    )
    effective_rewards = raw_rewards - (~terminated).float() * float(
        config["missing_eos_penalty"]
    )
    rewards = shaped_rewards(
        effective_rewards,
        old_logp,
        reference_logp,
        generated["response_mask"],
        float(config["kl_beta"]),
    )
    advantages, returns = compute_gae(
        rewards,
        old_values,
        generated["response_mask"],
        gamma=float(config["gamma"]),
        lam=float(config["gae_lambda"]),
    )
    return {
        **generated,
        "prompt_rows": prompt_rows,
        "messages": messages,
        "old_logp": old_logp.detach(),
        "reference_logp": reference_logp.detach(),
        "old_values": old_values.detach(),
        "raw_rewards": raw_rewards.detach(),
        "effective_rewards": effective_rewards.detach(),
        "shaped_rewards": rewards.detach(),
        "advantages": advantages.detach(),
        "returns": returns.detach(),
        "rollout_entropy": float(rollout_entropy),
    }


def update_from_rollout(
    policy,
    value_model,
    policy_optimizer,
    value_optimizer,
    rollout: dict,
    config: dict,
) -> dict:
    """Run the fixed number of PPO epochs over one frozen rollout batch."""
    mask = rollout["response_mask"]
    normalized_advantages = normalize_advantages(
        rollout["advantages"], mask
    ).detach()
    epoch_records = []
    for _ in range(int(config["ppo_epochs"])):
        # Keep training mode for gradient checkpointing, but disable dropout so
        # the first importance ratio is exactly one before any parameter update.
        policy.train()
        for module in policy.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        policy_optimizer.zero_grad(set_to_none=True)
        new_logp, logits = response_token_logprobs(
            policy,
            rollout["sequences"],
            rollout["attention_mask"],
            rollout["prompt_width"],
            rollout["response_ids"],
        )
        policy_loss, ratio, clip_fraction, affected_fraction = ppo_policy_loss(
            new_logp,
            rollout["old_logp"],
            normalized_advantages,
            mask,
            eps=float(config["clip_epsilon"]),
        )
        update_entropy = token_entropy(logits, mask)
        del logits
        policy_loss.backward()
        policy_gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            float(config["max_grad_norm"]),
        )
        policy_optimizer.step()

        value_model.train()
        value_optimizer.zero_grad(set_to_none=True)
        predicted_values = response_token_values(
            value_model,
            rollout["sequences"],
            rollout["attention_mask"],
            rollout["prompt_width"],
            rollout["response_ids"].shape[1],
        )
        value_loss = value_mse_loss(predicted_values, rollout["returns"], mask)
        scaled_value_loss = value_loss * float(config["value_coef"])
        scaled_value_loss.backward()
        value_gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in value_model.parameters() if parameter.requires_grad],
            float(config["max_grad_norm"]),
        )
        value_optimizer.step()

        valid_ratios = ratio[mask.bool()]
        epoch_records.append({
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "scaled_value_loss": float(scaled_value_loss.detach()),
            "update_entropy": float(update_entropy.detach()),
            "clip_fraction": float(clip_fraction),
            "affected_token_fraction": float(affected_fraction),
            "policy_gradient_norm": float(policy_gradient_norm),
            "value_gradient_norm": float(value_gradient_norm),
            "ratio_mean": float(valid_ratios.mean()),
            "ratio_std": float(valid_ratios.std(unbiased=False)),
        })
        if not all(np.isfinite(value) for value in epoch_records[-1].values()):
            raise FloatingPointError("PPO produced a non-finite optimization metric.")

    def mean(name: str) -> float:
        return float(np.mean([row[name] for row in epoch_records]))

    return {name: mean(name) for name in epoch_records[0]}


def summarize_rollout(rollout: dict) -> dict:
    """Summarize frozen rollout quantities before optimization."""
    mask = rollout["response_mask"]
    sampled_kl = masked_mean(
        rollout["old_logp"] - rollout["reference_logp"], mask
    )
    valid_advantages = rollout["advantages"][mask.bool()]
    return {
        "raw_reward": float(rollout["raw_rewards"].mean()),
        "effective_reward": float(rollout["effective_rewards"].mean()),
        "sampled_kl_token_mean": float(sampled_kl),
        "entropy": float(rollout["rollout_entropy"]),
        "old_value_mean": float(masked_mean(rollout["old_values"], mask)),
        "return_mean": float(masked_mean(rollout["returns"], mask)),
        "advantage_mean": float(valid_advantages.mean()),
        "advantage_std": float(valid_advantages.std(unbiased=False)),
        "response_length": float(mask.sum(dim=1).float().mean()),
        "generated_tokens": int(mask.sum().item()),
        "terminated_fraction": float(
            np.mean(rollout["terminated_with_eos"])
        ),
        "truncated_fraction": float(np.mean(rollout["truncated"])),
    }
