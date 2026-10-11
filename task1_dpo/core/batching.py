# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Prepare chosen and rejected responses with response-only loss masks."""

from __future__ import annotations

from common.data import (
    encode_prompt_response,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
)


def make_preference_collator(tokenizer, max_sequence_length: int):
    """Return a collator that pads preference pairs independently."""
    if max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least 2.")

    def collate(rows: list[dict]):
        chosen_examples = []
        rejected_examples = []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            chosen, rejected = preference_responses(row)
            chosen_examples.append(
                encode_prompt_response(tokenizer, prompt, chosen, max_sequence_length)
            )
            rejected_examples.append(
                encode_prompt_response(tokenizer, prompt, rejected, max_sequence_length)
            )
        chosen_batch = pad_batch(tokenizer, chosen_examples)
        rejected_batch = pad_batch(tokenizer, rejected_examples)
        for name, batch in (("chosen", chosen_batch), ("rejected", rejected_batch)):
            if (batch["response_mask"][:, 1:].sum(dim=1) == 0).any():
                raise ValueError(f"A {name} response has no tokens after truncation.")
        return chosen_batch, rejected_batch

    return collate
