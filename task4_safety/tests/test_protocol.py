# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Protect fixed Task 4 models, prompt counts, and decoding settings."""

import copy
import unittest

from common.data import load_yaml
from task4_safety.core.protocol import load_xstest, validate_config


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_yaml("configs/feedback.yaml")

    def test_released_configuration_and_xstest_are_valid(self):
        validate_config(self.config)
        table = load_xstest(self.config)
        self.assertEqual(len(table), 450)
        self.assertEqual(int((table["benchmark_class"] == "SAFE").sum()), 250)
        self.assertEqual(int((table["benchmark_class"] == "UNSAFE").sum()), 200)

    def test_ablation_adapter_cannot_replace_standard_policy(self):
        changed = copy.deepcopy(self.config)
        changed["policies"]["grpo"] = "task3_grpo/outputs/norm_dr_grpo/policy"
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_stochastic_decoding_budget_cannot_be_changed(self):
        changed = copy.deepcopy(self.config)
        changed["safety_max_new_tokens"] = 128
        with self.assertRaises(ValueError):
            validate_config(changed)


if __name__ == "__main__":
    unittest.main()
