# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Tests that protect the released Task 3 experimental settings."""

import copy
import unittest

from common.data import load_yaml
from task3_grpo.core.protocol import validate_shared_protocol, validate_training_protocol
from task3_grpo.core.trainer import fixed_prompt_schedule


class ProtocolTests(unittest.TestCase):
    def test_released_configuration_is_valid(self):
        config = load_yaml("configs/grpo.yaml")
        validate_shared_protocol(config)
        validate_training_protocol(config, "standard", 20, "grpo")
        validate_training_protocol(config, "norm_canonical", 8, "grpo")
        validate_training_protocol(config, "norm_dr_grpo", 8, "dr_grpo")

    def test_accidental_budget_change_is_rejected(self):
        config = copy.deepcopy(load_yaml("configs/grpo.yaml"))
        config["num_generations"] = 3
        with self.assertRaises(ValueError):
            validate_shared_protocol(config)

    def test_normalization_conditions_cannot_be_swapped(self):
        config = load_yaml("configs/grpo.yaml")
        with self.assertRaises(ValueError):
            validate_training_protocol(config, "norm_canonical", 8, "dr_grpo")
        with self.assertRaises(ValueError):
            validate_training_protocol(config, "norm_dr_grpo", 20, "dr_grpo")

    def test_prompt_schedule_is_deterministic(self):
        rows = [{"prompt_id": str(index)} for index in range(20)]
        first = fixed_prompt_schedule(rows, 8, 6304)
        second = fixed_prompt_schedule(rows, 8, 6304)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
