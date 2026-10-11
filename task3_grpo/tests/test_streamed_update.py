# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Exercise the memory-safe per-completion GRPO update on a tiny model."""

import math
import types
import unittest

import torch

from common.generation import response_token_logprobs
from task3_grpo.core.rollouts import update_from_rollout


class TinyPolicy(torch.nn.Module):
    def __init__(self, vocabulary_size: int = 11, hidden_size: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
        del attention_mask, use_cache, return_dict
        return types.SimpleNamespace(logits=self.output(self.embedding(input_ids)))


class StreamedUpdateTests(unittest.TestCase):
    def test_streamed_update_is_finite_and_changes_parameters(self):
        torch.manual_seed(7)
        policy = TinyPolicy()
        sequences = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]])
        attention = torch.ones_like(sequences)
        responses = sequences[:, 2:]
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy, sequences, attention, 2, responses
            )
        rollout = {
            "sequences": sequences,
            "attention_mask": attention,
            "prompt_width": 2,
            "response_ids": responses,
            "old_logp": old_logp.detach(),
            "reference_logp": old_logp.detach().clone(),
            "training_mask": torch.ones(2, 2),
            "advantages": torch.tensor([-1.0, 1.0]),
            "truncated": [False, False],
        }
        optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
        before = policy.output.weight.detach().clone()
        metrics = update_from_rollout(
            policy,
            optimizer,
            rollout,
            {
                "clip_epsilon": 0.2,
                "kl_beta": 0.1,
                "max_completion_length": 2,
                "max_grad_norm": 1.0,
            },
            "grpo",
        )
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertFalse(torch.equal(before, policy.output.weight.detach()))
        self.assertAlmostEqual(metrics["ratio_mean"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
