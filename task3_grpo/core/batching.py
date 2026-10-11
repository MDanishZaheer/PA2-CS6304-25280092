# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Memory-conscious token statistics for long GRPO completions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common.generation import response_token_logprobs
from common.models import reference_mode


def token_entropy(
    logits: torch.Tensor, mask: torch.Tensor, chunk_size: int = 32
) -> torch.Tensor:
    """Return exact categorical entropy in small token chunks."""
    if logits.shape[:2] != mask.shape:
        raise ValueError("Logits and response mask have incompatible shapes.")
    if int(chunk_size) <= 0:
        raise ValueError("Entropy chunk size must be positive.")
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    with torch.no_grad():
        detached = logits.detach()
        for start in range(0, logits.shape[1], int(chunk_size)):
            end = min(start + int(chunk_size), logits.shape[1])
            chunk = detached[:, start:end, :].float()
            log_normalizer = torch.logsumexp(chunk, dim=-1)
            probabilities = F.softmax(chunk, dim=-1)
            entropy = log_normalizer - (probabilities * chunk).sum(-1)
            total += (entropy * mask[:, start:end]).sum()
    return total / mask.sum().clamp_min(1.0)


@torch.no_grad()
def frozen_policy_statistics(policy, generated: dict) -> dict:
    """Compute old/reference log-probabilities one completion at a time.

    Sequential forwards avoid retaining a batch of full-vocabulary logits on a
    laptop GPU. They do not change the model, prompts, or assignment budgets.
    """
    old_rows = []
    entropy_weighted_sum = 0.0
    entropy_tokens = 0.0
    batch_size = int(generated["sequences"].shape[0])
    for index in range(batch_size):
        one = slice(index, index + 1)
        logp, logits = response_token_logprobs(
            policy,
            generated["sequences"][one],
            generated["attention_mask"][one],
            generated["prompt_width"],
            generated["response_ids"][one],
        )
        mask = generated["response_mask"][one]
        count = float(mask.sum())
        entropy_weighted_sum += float(token_entropy(logits, mask)) * count
        entropy_tokens += count
        old_rows.append(logp.detach())
        del logits

    reference_rows = []
    with reference_mode(policy):
        for index in range(batch_size):
            one = slice(index, index + 1)
            logp, logits = response_token_logprobs(
                policy,
                generated["sequences"][one],
                generated["attention_mask"][one],
                generated["prompt_width"],
                generated["response_ids"][one],
            )
            reference_rows.append(logp.detach())
            del logits
    return {
        "old_logp": torch.cat(old_rows, dim=0),
        "reference_logp": torch.cat(reference_rows, dim=0),
        "entropy": entropy_weighted_sum / max(entropy_tokens, 1.0),
    }
