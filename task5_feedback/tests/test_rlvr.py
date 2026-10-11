# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Verify the exact designated-final parser and distractor behavior."""

import unittest

from task5_feedback.rlvr import (
    exact_reward,
    extract_designated_final,
    format_compliance,
)


class ExactVerifierTests(unittest.TestCase):
    def test_intermediate_gold_number_is_not_rewarded(self):
        response = "I considered 42, rejected it, and finished with 41."
        self.assertEqual(exact_reward(response, "42"), 0.0)
        self.assertEqual(format_compliance(response), 0.0)

    def test_last_designated_final_is_used(self):
        response = "First attempt: #### 41\nCorrection: #### 42"
        self.assertEqual(extract_designated_final(response), "42")
        self.assertEqual(exact_reward(response, "42"), 1.0)

    def test_commas_signs_and_decimals_are_numeric(self):
        self.assertEqual(exact_reward("#### -1,234.50", "-1234.5"), 1.0)


if __name__ == "__main__":
    unittest.main()
