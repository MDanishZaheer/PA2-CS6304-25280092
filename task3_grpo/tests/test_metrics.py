# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Tests for exact entropy and held-out aggregation."""

import math
import unittest

import torch

from task3_grpo.core.batching import token_entropy
from task3_grpo.evaluation.policy_metrics import summarize_policy


class MetricsTests(unittest.TestCase):
    def test_exact_entropy_of_uniform_logits(self):
        value = token_entropy(torch.zeros(1, 2, 4), torch.ones(1, 2))
        self.assertAlmostEqual(float(value), math.log(4.0), places=6)

    def test_heldout_kl_and_entropy_are_token_weighted(self):
        records = [
            {
                "response_length": 2,
                "reward_model_score": 1.0,
                "sampled_kl_sequence_sum": 2.0,
                "entropy_sequence_sum": 4.0,
                "terminated_with_eos": True,
                "truncated": False,
            },
            {
                "response_length": 6,
                "reward_model_score": 3.0,
                "sampled_kl_sequence_sum": 6.0,
                "entropy_sequence_sum": 18.0,
                "terminated_with_eos": False,
                "truncated": True,
            },
        ]
        summary = summarize_policy(records)
        self.assertAlmostEqual(summary["reward_model_score_mean"], 2.0)
        self.assertAlmostEqual(summary["sampled_kl_token_mean"], 1.0)
        self.assertAlmostEqual(summary["entropy_token_mean"], 2.75)
        self.assertAlmostEqual(summary["response_length_mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
