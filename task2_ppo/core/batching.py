# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build response-aligned tensors for PPO training and diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common.models import token_values


def response_token_values(
    value_model,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
    response_steps: int,
) -> torch.Tensor:
    """Align critic state values with generated response actions."""
    all_values = token_values(value_model, sequences, attention_mask)
    values = all_values[:, prompt_width - 1 : prompt_width - 1 + response_steps]
    if values.shape[1] != response_steps:
        raise ValueError("Critic values do not align with the response tokens.")
    return values.float()


def token_entropy(
    logits: torch.Tensor, mask: torch.Tensor, chunk_size: int = 32
) -> torch.Tensor:
    """Calculate exact entropy in small token chunks to limit peak VRAM."""
    if logits.shape[:2] != mask.shape:
        raise ValueError("Policy logits and response mask must align.")
    if int(chunk_size) <= 0:
        raise ValueError("Entropy chunk size must be positive.")
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    # Entropy is a diagnostic, not part of the PPO loss, so its graph is not
    # retained. Chunking avoids several full [tokens, vocabulary] float copies.
    with torch.no_grad():
        detached_logits = logits.detach()
        for start in range(0, logits.shape[1], int(chunk_size)):
            end = min(start + int(chunk_size), logits.shape[1])
            chunk = detached_logits[:, start:end, :].float()
            log_normalizer = torch.logsumexp(chunk, dim=-1)
            probabilities = F.softmax(chunk, dim=-1)
            entropy = log_normalizer - (probabilities * chunk).sum(dim=-1)
            total += (entropy * mask[:, start:end]).sum()
    return total / mask.sum().clamp_min(1.0)


def encode_cached_response(
    tokenizer,
    messages: list[dict],
    response: str,
    max_prompt_length: int,
    expected_response_tokens: int,
    terminated_with_eos: bool,
) -> dict:
    """Reconstruct one cached rollout using the original generation limits."""
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )[-max_prompt_length:]
    suffix = tokenizer.eos_token if terminated_with_eos else ""
    response_ids = tokenizer(
        response + (suffix or ""), add_special_tokens=False
    )["input_ids"]
    if len(response_ids) != int(expected_response_tokens):
        raise ValueError(
            "Cached response token count could not be reconstructed: "
            f"expected {expected_response_tokens}, found {len(response_ids)}."
        )
    input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "response_ids": torch.tensor([response_ids], dtype=torch.long),
        "response_mask": torch.ones((1, len(response_ids)), dtype=torch.float32),
        "prompt_width": len(prompt_ids),
    }


def pad_token_vectors(vectors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad one-dimensional response tensors and return their valid mask."""
    if not vectors or any(vector.ndim != 1 for vector in vectors):
        raise ValueError("Expected a non-empty list of one-dimensional tensors.")
    width = max(int(vector.numel()) for vector in vectors)
    values = torch.zeros((len(vectors), width), dtype=vectors[0].dtype)
    mask = torch.zeros((len(vectors), width), dtype=torch.float32)
    for index, vector in enumerate(vectors):
        length = int(vector.numel())
        values[index, :length] = vector
        mask[index, :length] = 1.0
    return values, mask
