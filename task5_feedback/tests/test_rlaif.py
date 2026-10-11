# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Test fixed judge parsing and cached lazy-loading behavior."""

import tempfile
import unittest
from pathlib import Path

from common.data import load_yaml
from task5_feedback.rlaif import PairwiseAIJudge


class PairwiseJudgeTests(unittest.TestCase):
    def test_parser_accepts_only_fixed_tokens(self):
        self.assertEqual(PairwiseAIJudge._parse_preference("A"), ("A", False))
        self.assertEqual(
            PairwiseAIJudge._parse_preference("unclear output"), ("TIE", True)
        )

    def test_cached_comparison_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = load_yaml("configs/feedback.yaml")
            judge = PairwiseAIJudge(config, Path(temporary) / "cache.json")
            key = judge._key("problem", "answer a", "answer b")
            judge.cache[key] = {
                "preference": "A",
                "raw_output": "A",
                "swapped": False,
                "parse_failure": False,
                "input_tokens": 10,
                "inference_seconds": 0.1,
            }
            result = judge.compare_detail("problem", "answer a", "answer b")
            self.assertEqual(result["preference"], "A")
            self.assertTrue(result["cache_hit"])
            self.assertIsNone(judge.model)


if __name__ == "__main__":
    unittest.main()
