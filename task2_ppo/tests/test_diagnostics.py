# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check exact entropy and cached clipping diagnostic definitions."""

from __future__ import annotations

import math
import unittest

import torch

from common.data import load_yaml
from task2_ppo.analyze_clipping import (
    _prompt_lookup,
    compute_cached_clipping_metrics,
    load_cached_rollouts,
)
from task2_ppo.core.batching import pad_token_vectors, token_entropy


class DiagnosticTests(unittest.TestCase):
    """Verify report metrics on small hand-calculated examples."""

    def test_uniform_policy_entropy(self):
        logits = torch.zeros(1, 2, 4)
        mask = torch.tensor([[1.0, 0.0]])
        self.assertAlmostEqual(float(token_entropy(logits, mask)), math.log(4), places=6)

    def test_chunked_entropy_matches_direct_definition(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 5, 11)
        mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        direct = -(
            log_probabilities.exp() * log_probabilities
        ).sum(dim=-1)
        direct = (direct * mask).sum() / mask.sum()
        self.assertTrue(torch.allclose(token_entropy(logits, mask, 2), direct))

    def test_token_vector_padding(self):
        values, mask = pad_token_vectors([
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0]),
        ])
        self.assertTrue(torch.equal(values, torch.tensor([[1.0, 2.0], [3.0, 0.0]])))
        self.assertTrue(torch.equal(mask, torch.tensor([[1.0, 1.0], [1.0, 0.0]])))

    def test_cached_clip_and_affected_fractions_are_distinct(self):
        old_logprobs = torch.zeros(1, 3)
        new_logprobs = torch.log(torch.tensor([[1.5, 0.5, 1.5]]))
        advantages = torch.tensor([[1.0, 1.0, -1.0]])
        mask = torch.ones(1, 3)
        metrics = compute_cached_clipping_metrics(
            new_logprobs, old_logprobs, advantages, mask, [0.2]
        )[0]
        self.assertEqual(metrics["clip_fraction"], 1.0)
        self.assertAlmostEqual(metrics["affected_token_fraction"], 1.0 / 3.0)

    def test_cached_prompts_are_recovered_from_the_heldout_pool(self):
        config = load_yaml("configs/ppo.yaml")
        cached_rows = load_cached_rollouts(config["cached_rollouts"])
        prompts = _prompt_lookup(config)
        self.assertEqual(len(cached_rows), 32)
        for row in cached_rows:
            self.assertIn(row["prompt_id"], prompts)
            self.assertEqual(
                int(row["source_index"]),
                int(prompts[row["prompt_id"]]["source_index"]),
            )


if __name__ == "__main__":
    unittest.main()
