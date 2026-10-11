# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Ensure completed expensive stages return before loading any model."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from task4_safety.core.paths import POLICY_ORDER
from task4_safety.evaluation.ai_judge import judge_all_responses
from task4_safety.evaluation.generation import generate_policy_responses
from task4_safety.tests.helpers import (
    temporary_config,
    write_synthetic_judgments,
    write_synthetic_responses,
)


class ResumeTests(unittest.TestCase):
    def test_completed_generation_does_not_load_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = temporary_config(Path(temporary))
            write_synthetic_responses(config)
            with patch(
                "task4_safety.evaluation.generation.load_policy",
                side_effect=AssertionError("policy should not load"),
            ):
                rows = generate_policy_responses(config, "sft")
            self.assertEqual(len(rows), 450)

    def test_completed_judging_does_not_load_judge(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = temporary_config(Path(temporary))
            write_synthetic_responses(config)
            write_synthetic_judgments(config)
            with patch(
                "task4_safety.evaluation.ai_judge.load_judge",
                side_effect=AssertionError("judge should not load"),
            ):
                rows = judge_all_responses(config)
            self.assertEqual(set(rows), set(POLICY_ORDER))
            self.assertTrue(all(len(value) == 450 for value in rows.values()))


if __name__ == "__main__":
    unittest.main()
