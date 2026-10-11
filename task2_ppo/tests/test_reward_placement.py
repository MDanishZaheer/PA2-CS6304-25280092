# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Protect the low-memory reward-model placement used during PPO training."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from common.models import promote_value_head
from task2_ppo.core.trainer import _load_models_and_optimizers


class RewardPlacementTests(unittest.TestCase):
    """Keep training rewards on CPU while evaluation retains auto placement."""

    def test_training_loads_reward_model_on_cpu(self):
        config = {
            "base_model": "base-model",
            "paths": {
                "ppo_midpoint_policy": "checkpoints/ppo_midpoint_policy",
                "ppo_midpoint_value": "checkpoints/ppo_midpoint_value",
            },
            "value_train_mode": "lora_head",
            "policy_learning_rate": 3.0e-6,
            "value_lora_learning_rate": 1.0e-4,
            "value_head_learning_rate": 3.0e-4,
        }
        policy = torch.nn.Linear(1, 1)
        critic = torch.nn.Linear(1, 1)
        reward_model = torch.nn.Linear(1, 1)

        with patch(
            "task2_ppo.core.trainer.load_policy", return_value=policy
        ), patch(
            "task2_ppo.core.trainer.load_value_model", return_value=critic
        ), patch(
            "task2_ppo.core.trainer.load_reward_model",
            return_value=(reward_model, object()),
        ) as reward_loader, patch(
            "task2_ppo.core.trainer.load_tokenizer", return_value=object()
        ), patch(
            "task2_ppo.core.trainer.value_parameter_groups",
            return_value=[{"params": list(critic.parameters()), "lr": 1.0e-4}],
        ):
            _load_models_and_optimizers(config)

        reward_loader.assert_called_once_with(config, device="cpu")

    def test_trainable_value_head_is_promoted_to_float32(self):
        model = torch.nn.Module()
        model.score = torch.nn.Linear(4, 1, bias=False).half()

        promote_value_head(model)

        self.assertEqual(model.score.weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
