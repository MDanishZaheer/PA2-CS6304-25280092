# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Protect the fixed Task 5 datasets, policies, and budgets."""

import copy
import unittest

from common.data import load_yaml
from task5_feedback.core.protocol import (
    EXPECTED_VARIANTS,
    load_diagnostic_rows,
    load_math_rows,
    row_id,
    validate_config,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_yaml("configs/feedback.yaml")

    def test_released_data_counts_and_variants(self):
        gsm = load_math_rows(self.config, "gsm")
        transfer = load_math_rows(self.config, "transfer")
        self.assertEqual(len(gsm), 300)
        self.assertEqual(len(transfer), 100)
        self.assertEqual(row_id(gsm[0], "gsm"), f"gsm:{gsm[0]['source_index']}")
        self.assertEqual(row_id(transfer[0], "transfer"), transfer[0]["prompt_id"])
        diagnostics = load_diagnostic_rows(self.config)
        self.assertEqual(len(diagnostics), 100)
        self.assertEqual(
            tuple(row["variant_type"] for row in diagnostics[:5]),
            EXPECTED_VARIANTS,
        )

    def test_policy_substitution_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["policies"]["rlvr"] = "task3_grpo/outputs/standard/policy"
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_generation_budget_change_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["math_max_new_tokens"] = 256
        with self.assertRaises(ValueError):
            validate_config(changed)


if __name__ == "__main__":
    unittest.main()
