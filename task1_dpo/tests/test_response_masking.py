# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check that teacher-forced preference scores ignore prompt-only tokens."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from common.generation import response_sequence_logprobs
from task1_dpo.core.log_probs import response_sequence_log_probs


class FixedLogitsModel(torch.nn.Module):
    """Return prescribed token logits for a response-mask unit test."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, **_):
        return SimpleNamespace(logits=self.logits)


class ResponseMaskTests(unittest.TestCase):
    """Ensure prompt probabilities cannot change a response-only score."""

    def test_prompt_logits_do_not_change_sequence_log_probability(self):
        batch = {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "response_mask": torch.tensor([[0, 0, 0, 1]], dtype=torch.float32),
        }
        logits_a = torch.zeros(1, 4, 4)
        logits_b = logits_a.clone()
        logits_b[:, 0, 0] = 100.0
        logits_b[:, 1, 0] = 100.0
        score_a, _, _ = response_sequence_logprobs(FixedLogitsModel(logits_a), batch)
        score_b, _, _ = response_sequence_logprobs(FixedLogitsModel(logits_b), batch)
        self.assertTrue(torch.allclose(score_a, score_b))
        chunked_score = response_sequence_log_probs(FixedLogitsModel(logits_b), batch, chunk_size=2)
        self.assertTrue(torch.allclose(score_a, chunked_score))

if __name__ == "__main__":
    unittest.main()
