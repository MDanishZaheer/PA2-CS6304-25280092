# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate response-only sequence log-probabilities in small token chunks."""

from __future__ import annotations

import torch


def response_sequence_log_probs(model, batch: dict, chunk_size: int = 64) -> torch.Tensor:
    """Sum teacher-forced log-probabilities only where response_mask is one."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:]
    sequence_log_probs = torch.zeros(
        logits.shape[0], device=logits.device, dtype=torch.float32
    )
    for start in range(0, logits.shape[1], chunk_size):
        end = min(start + chunk_size, logits.shape[1])
        chunk = logits[:, start:end, :].float()
        chosen = chunk.gather(-1, labels[:, start:end].unsqueeze(-1)).squeeze(-1)
        token_log_probs = chosen - torch.logsumexp(chunk, dim=-1)
        sequence_log_probs = sequence_log_probs + (
            token_log_probs * mask[:, start:end]
        ).sum(dim=-1)
    return sequence_log_probs
