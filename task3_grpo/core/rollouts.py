# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Generate GRPO groups, score them, and perform memory-safe policy updates."""

from __future__ import annotations

import math

import numpy as np
import torch

from common.data import prompt_messages
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.metrics import masked_mean, safe_corr
from task3_grpo.core.batching import frozen_policy_statistics
from task3_grpo.grpo import (
    gradient_allocation_proxy,
    group_relative_advantages,
    mask_truncated_sequences,
    normalization_denominators,
)


GROUP_STD_TOLERANCE = 1e-6


def independent_reward_scores(
    reward_model,
    reward_tokenizer,
    prompts: list[list[dict]],
    responses: list[str],
    max_length: int = 1024,
) -> torch.Tensor:
    """Score each response separately so dynamic padding cannot alter scores."""
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
    scores = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(scores).all():
        raise ValueError("The reward model produced a non-finite score.")
    return scores


def collect_rollout(
    policy,
    reward_model,
    tokenizer,
    reward_tokenizer,
    prompt_rows: list[dict],
    config: dict,
    loss_type: str,
) -> dict:
    """Sample K completions per prompt and freeze all rollout targets."""
    k = int(config["num_generations"])
    prompt_messages_once = [prompt_messages(row) for row in prompt_rows]
    messages = [messages for messages in prompt_messages_once for _ in range(k)]
    expanded_rows = [row for row in prompt_rows for _ in range(k)]
    group_ids = torch.arange(len(prompt_rows)).repeat_interleave(k)

    old_cache_setting = bool(getattr(policy.config, "use_cache", False))
    policy.config.use_cache = True
    policy.eval()
    generated = batch_generate(
        policy,
        tokenizer,
        messages,
        max_prompt_length=int(config["max_prompt_length"]),
        max_new_tokens=int(config["max_completion_length"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
        do_sample=bool(config["generation"]["do_sample"]),
    )
    policy.config.use_cache = old_cache_setting
    statistics = frozen_policy_statistics(policy, generated)
    device = statistics["old_logp"].device
    group_ids = group_ids.to(device)
    rewards = independent_reward_scores(
        reward_model,
        reward_tokenizer,
        messages,
        generated["responses"],
        max_length=int(config.get("reward_max_length", 1024)),
    ).to(device)
    advantages = group_relative_advantages(
        rewards, group_ids, eps=GROUP_STD_TOLERANCE
    )
    training_mask = (
        mask_truncated_sequences(
            generated["response_mask"], generated["truncated"]
        )
        if bool(config["mask_truncated_completions"])
        else generated["response_mask"]
    )
    per_token_weight, sequence_weight = gradient_allocation_proxy(
        advantages,
        training_mask,
        loss_type,
        int(config["max_completion_length"]),
    )
    policy.train()
    return {
        **generated,
        **statistics,
        "prompt_rows": expanded_rows,
        "messages": messages,
        "group_ids": group_ids,
        "rewards": rewards.detach(),
        "advantages": advantages.detach(),
        "training_mask": training_mask.detach(),
        "per_token_gradient_weight": per_token_weight.detach(),
        "sequence_gradient_weight_proxy": sequence_weight.detach(),
        "loss_type": loss_type,
    }


def _group_statistics(rewards: torch.Tensor, group_ids: torch.Tensor) -> tuple[float, float]:
    standard_deviations = []
    for group_id in torch.unique(group_ids, sorted=True):
        values = rewards[group_ids == group_id].float()
        standard_deviations.append(float(values.std(unbiased=False)))
    uninformative = [value <= GROUP_STD_TOLERANCE for value in standard_deviations]
    return float(np.mean(standard_deviations)), float(np.mean(uninformative))


def summarize_rollout(rollout: dict) -> dict:
    """Summarize frozen pre-update behavior and group informativeness."""
    response_mask = rollout["response_mask"]
    training_mask = rollout["training_mask"]
    reward_std, uninformative_fraction = _group_statistics(
        rollout["rewards"], rollout["group_ids"]
    )
    sampled_kl = masked_mean(
        rollout["old_logp"] - rollout["reference_logp"], response_mask
    )
    lengths = response_mask.sum(-1).float()
    sequence_weights = rollout["sequence_gradient_weight_proxy"].float()
    length_correlation = safe_corr(
        lengths.cpu().numpy(), sequence_weights.cpu().numpy()
    )
    return {
        "reward_mean": float(rollout["rewards"].mean()),
        "reward_std": float(rollout["rewards"].std(unbiased=False)),
        "sampled_kl_token_mean": float(sampled_kl),
        "entropy_token_mean": float(rollout["entropy"]),
        "within_group_reward_std_mean": reward_std,
        "uninformative_group_fraction": uninformative_fraction,
        "informative_group_fraction": 1.0 - uninformative_fraction,
        "advantage_mean": float(rollout["advantages"].mean()),
        "advantage_std": float(rollout["advantages"].std(unbiased=False)),
        "response_length_mean": float(lengths.mean()),
        "response_length_std": float(lengths.std(unbiased=False)),
        "generated_tokens": int(response_mask.sum()),
        "training_tokens": int(training_mask.sum()),
        "terminated_fraction": float(np.mean(rollout["terminated_with_eos"])),
        "truncated_fraction": float(np.mean(rollout["truncated"])),
        "gradient_weight_length_correlation": (
            length_correlation if np.isfinite(length_correlation) else None
        ),
        "sequence_gradient_weight_mean": float(sequence_weights.mean()),
    }


def completion_records(rollout: dict, update: int, run_name: str) -> list[dict]:
    """Return one auditable length/reward/gradient row per completion."""
    records = []
    group_positions: dict[int, int] = {}
    for index, row in enumerate(rollout["prompt_rows"]):
        group_id = int(rollout["group_ids"][index])
        generation_index = group_positions.get(group_id, 0)
        group_positions[group_id] = generation_index + 1
        records.append({
            "run_name": run_name,
            "update": int(update),
            "prompt_id": row["prompt_id"],
            "source_index": row.get("source_index"),
            "generation_index": generation_index,
            "group_id": group_id,
            "reward": float(rollout["rewards"][index]),
            "advantage": float(rollout["advantages"][index]),
            "response_length": int(rollout["response_mask"][index].sum()),
            "training_tokens": int(rollout["training_mask"][index].sum()),
            "terminated_with_eos": bool(rollout["terminated_with_eos"][index]),
            "truncated": bool(rollout["truncated"][index]),
            "per_token_gradient_weight": float(
                rollout["per_token_gradient_weight"][index]
            ),
            "sequence_gradient_weight_proxy": float(
                rollout["sequence_gradient_weight_proxy"][index]
            ),
        })
    return records


def update_from_rollout(
    policy,
    optimizer,
    rollout: dict,
    config: dict,
    loss_type: str,
) -> dict:
    """Apply one streamed GRPO update without retaining K large logits tensors."""
    policy.train()
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    optimizer.zero_grad(set_to_none=True)
    batch_size = int(rollout["sequences"].shape[0])
    total_training_tokens = float(rollout["training_mask"].sum().clamp_min(1.0))
    denominators = normalization_denominators(
        rollout["training_mask"],
        loss_type,
        int(config["max_completion_length"]),
    )
    sums = {
        "policy_term": 0.0,
        "kl_numerator": 0.0,
        "clip_numerator": 0.0,
        "affected_numerator": 0.0,
        "ratio_sum": 0.0,
        "ratio_square_sum": 0.0,
    }
    epsilon = float(config["clip_epsilon"])
    beta = float(config["kl_beta"])
    for index in range(batch_size):
        one = slice(index, index + 1)
        new_logp, logits = response_token_logprobs(
            policy,
            rollout["sequences"][one],
            rollout["attention_mask"][one],
            rollout["prompt_width"],
            rollout["response_ids"][one],
        )
        del logits
        old_logp = rollout["old_logp"][one]
        reference_logp = rollout["reference_logp"][one]
        mask = rollout["training_mask"][one]
        advantage = rollout["advantages"][index]
        ratio = torch.exp(new_logp - old_logp)
        unclipped = ratio * advantage
        clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantage
        objective = torch.minimum(unclipped, clipped)
        policy_piece = -(
            (objective * mask).sum() / denominators[index] / batch_size
        )
        log_ratio_reference_over_policy = reference_logp - new_logp
        kl_tokens = (
            torch.exp(log_ratio_reference_over_policy)
            - log_ratio_reference_over_policy
            - 1.0
        )
        kl_piece = (kl_tokens * mask).sum() / total_training_tokens
        loss_piece = policy_piece + beta * kl_piece
        loss_piece.backward()

        valid_count = float(mask.sum())
        sums["policy_term"] += float(policy_piece.detach())
        sums["kl_numerator"] += float((kl_tokens.detach() * mask).sum())
        sums["clip_numerator"] += float((
            ((ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)).float()
            * mask
        ).sum())
        sums["affected_numerator"] += float(((clipped < unclipped).float() * mask).sum())
        sums["ratio_sum"] += float((ratio.detach() * mask).sum())
        sums["ratio_square_sum"] += float(((ratio.detach() ** 2) * mask).sum())
        if valid_count == 0 and not bool(rollout["truncated"][index]):
            raise RuntimeError("A non-truncated completion unexpectedly has no training tokens.")

    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters, float(config["max_grad_norm"])
    )
    optimizer.step()
    kl_penalty = sums["kl_numerator"] / total_training_tokens
    ratio_mean = sums["ratio_sum"] / total_training_tokens
    ratio_variance = max(
        sums["ratio_square_sum"] / total_training_tokens - ratio_mean**2,
        0.0,
    )
    metrics = {
        "policy_loss": sums["policy_term"] + beta * kl_penalty,
        "policy_term": sums["policy_term"],
        "objective_kl_penalty": kl_penalty,
        "clip_fraction": sums["clip_numerator"] / total_training_tokens,
        "affected_token_fraction": sums["affected_numerator"] / total_training_tokens,
        "ratio_mean": ratio_mean,
        "ratio_std": math.sqrt(ratio_variance),
        "gradient_norm": float(gradient_norm),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("GRPO produced a non-finite optimization metric.")
    return metrics
