# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Protect Task 1 reward evaluation from batch-dependent scores."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from task1_dpo.evaluation.policy_metrics import add_reward_scores


class RewardScoringTests(unittest.TestCase):
    """Confirm policy and reference responses are scored independently."""

    def test_each_response_uses_a_one_item_batch(self):
        records = [{
            "messages": [{"role": "user", "content": "Explain DPO."}],
            "policy_response": "Policy answer",
            "reference_response": "Reference answer",
        }]
        with patch(
            "task1_dpo.evaluation.policy_metrics.score_reward_pairs",
            side_effect=[torch.tensor([1.25]), torch.tensor([0.50])],
        ) as scorer:
            add_reward_scores(records, object(), object())

        self.assertEqual(scorer.call_count, 2)
        for call in scorer.call_args_list:
            self.assertEqual(len(call.args[2]), 1)
            self.assertEqual(len(call.args[3]), 1)
        self.assertEqual(records[0]["policy_reward_score"], 1.25)
        self.assertEqual(records[0]["reference_reward_score"], 0.50)


if __name__ == "__main__":
    unittest.main()
