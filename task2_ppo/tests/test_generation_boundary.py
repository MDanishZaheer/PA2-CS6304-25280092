# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check that generated rollout tensors can be reused by PPO autograd."""

from __future__ import annotations

import unittest

import torch

from common.generation import batch_generate


class TinyTokenizer:
    """Provide the tokenizer interface used by batch generation."""

    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return messages[-1]["content"]

    def __call__(self, texts, **_):
        return {
            "input_ids": torch.tensor([[1, 4]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, tokens, skip_special_tokens=True):
        return "tiny response"


class TinyGenerator(torch.nn.Module):
    """Return two fixed generated tokens under the helper's inference context."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, **_):
        suffix = torch.tensor([[3, 2]], device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


class GenerationBoundaryTests(unittest.TestCase):
    """Prevent inference-only tensors from reaching the PPO update graph."""

    def test_generated_sequences_are_normal_tensors(self):
        generated = batch_generate(
            TinyGenerator(),
            TinyTokenizer(),
            [[{"role": "user", "content": "question"}]],
            max_prompt_length=8,
            max_new_tokens=2,
            do_sample=False,
        )
        self.assertFalse(torch.is_inference(generated["sequences"]))
        self.assertFalse(torch.is_inference(generated["response_ids"]))


if __name__ == "__main__":
    unittest.main()
