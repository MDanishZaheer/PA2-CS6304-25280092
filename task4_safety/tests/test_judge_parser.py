# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check the fixed judge parser and failure behavior."""

import unittest

from task4_safety.evaluation.ai_judge import parse_json


class JudgeParserTests(unittest.TestCase):
    def test_valid_json_is_parsed(self):
        result = parse_json(
            '{"label":"SAFE_ANSWER","confidence":0.8,"rationale_tag":"answers"}'
        )
        self.assertEqual(result["label"], "SAFE_ANSWER")
        self.assertAlmostEqual(result["confidence"], 0.8)

    def test_invalid_output_becomes_ambiguous(self):
        result = parse_json("This is not JSON")
        self.assertEqual(result["label"], "AMBIGUOUS")
        self.assertEqual(result["rationale_tag"], "parse_failure")

    def test_unknown_label_becomes_ambiguous(self):
        result = parse_json(
            '{"label":"REFUSAL","confidence":2,"rationale_tag":"bad label"}'
        )
        self.assertEqual(result["label"], "AMBIGUOUS")
        self.assertEqual(result["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
