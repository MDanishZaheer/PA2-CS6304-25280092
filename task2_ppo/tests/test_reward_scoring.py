# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Protect Task 2 reward evaluation from batch-dependent scores."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from task2_ppo.evaluation.policy_metrics import add_reward_scores


class RewardScoringTests(unittest.TestCase):
    """Confirm every saved response uses a one-item reward-model batch."""

    def test_each_response_is_scored_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            records = [
                {
                    "prompt_id": "p1",
                    "messages": [{"role": "user", "content": "Question one"}],
                    "response": "Answer one",
                },
                {
                    "prompt_id": "p2",
                    "messages": [{"role": "user", "content": "Question two"}],
                    "response": "Answer two",
                },
            ]
            config = {
                "reward_model": "reward-model",
                "reward_tokenizer": "tokenizer",
                "base_model": "base-model",
                "reward_max_length": 1280,
                "quantize_frozen_models": True,
                "results_dir": str(Path(directory) / "results"),
            }
            dummy_model = torch.nn.Linear(1, 1)
            with patch(
                "task2_ppo.evaluation.policy_metrics.load_reward_model",
                return_value=(dummy_model, object()),
            ) as loader, patch(
                "task2_ppo.evaluation.policy_metrics.score_reward_pairs",
                side_effect=[torch.tensor([1.25]), torch.tensor([0.50])],
            ) as scorer:
                add_reward_scores(records, config)
            loader.assert_called_once_with(config)
            self.assertEqual(scorer.call_count, 2)
            for call in scorer.call_args_list:
                self.assertEqual(len(call.args[2]), 1)
                self.assertEqual(len(call.args[3]), 1)
            self.assertEqual(records[0]["reward_model_score"], 1.25)
            self.assertEqual(records[1]["reward_model_score"], 0.50)


if __name__ == "__main__":
    unittest.main()
