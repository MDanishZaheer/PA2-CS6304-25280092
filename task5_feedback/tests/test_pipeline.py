# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Exercise Task 5 resume logic and complete synthetic evidence generation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from task5_feedback.analysis.aggregate import build_all_evidence
from task5_feedback.analysis.plots import save_all_figures
from task5_feedback.core.paths import POLICY_ORDER
from task5_feedback.evaluation.diagnostics import (
    build_diagnostic_tables,
    judge_diagnostic_pairs,
)
from task5_feedback.evaluation.generation import generate_policy_responses
from task5_feedback.evaluation.math_metrics import build_math_tables
from task5_feedback.evaluation.pairwise import judge_dataset_pairs
from task5_feedback.tests.helpers import (
    temporary_config,
    write_synthetic_diagnostic_pairs,
    write_synthetic_policy_pairs,
    write_synthetic_responses,
)


class SyntheticPipelineTests(unittest.TestCase):
    def test_completed_generation_skips_policy_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = temporary_config(Path(temporary))
            write_synthetic_responses(config, "gsm")
            with patch(
                "task5_feedback.evaluation.generation.load_policy",
                side_effect=AssertionError("policy should not load"),
            ):
                rows = generate_policy_responses(config, "gsm", "sft")
            self.assertEqual(len(rows), 300)

    def test_completed_pairwise_stage_skips_judge_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = temporary_config(Path(temporary))
            write_synthetic_responses(config, "transfer")
            write_synthetic_policy_pairs(config, "transfer")
            with patch(
                "task5_feedback.evaluation.pairwise.PairwiseAIJudge",
                side_effect=AssertionError("judge should not be created"),
            ):
                rows = judge_dataset_pairs(config, "transfer")
            self.assertEqual(len(rows), 300)

    def test_completed_diagnostic_stage_skips_judge_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = temporary_config(Path(temporary))
            write_synthetic_diagnostic_pairs(config)
            with patch(
                "task5_feedback.evaluation.diagnostics.PairwiseAIJudge",
                side_effect=AssertionError("judge should not be created"),
            ):
                rows = judge_diagnostic_pairs(config)
            self.assertEqual(len(rows), 200)

    def test_required_math_and_diagnostic_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = temporary_config(Path(temporary))
            write_synthetic_responses(config, "gsm")
            write_synthetic_policy_pairs(config, "gsm")
            math = build_math_tables(config, "gsm")
            self.assertEqual(
                math["policy_metrics"]["policy"].tolist(), list(POLICY_ORDER)
            )
            self.assertTrue(
                (math["policy_metrics"]["exact_accuracy"] == 1.0).all()
            )

            write_synthetic_diagnostic_pairs(config)
            diagnostics = build_diagnostic_tables(config)
            self.assertEqual(len(diagnostics["variant_scores"]), 5)
            self.assertEqual(len(diagnostics["controlled_pairs"]), 8)
            self.assertEqual(len(diagnostics["category_metrics"]), 10)
            verifier = diagnostics["sensitivity"].set_index("mechanism").loc[
                "RLVR_exact_verifier"
            ]
            self.assertEqual(verifier["S_reason"], 0.0)
            self.assertEqual(verifier["S_outcome"], 1.0)

    def test_complete_evidence_and_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = temporary_config(Path(temporary))
            for dataset in ("gsm", "transfer"):
                write_synthetic_responses(config, dataset)
                write_synthetic_policy_pairs(config, dataset)
            write_synthetic_diagnostic_pairs(config)
            evidence = build_all_evidence(config)
            self.assertEqual(len(evidence["research_questions"]), 4)
            figures = save_all_figures(config)
            self.assertEqual(len(figures), 4)
            self.assertTrue(all(path.exists() for path in figures))


if __name__ == "__main__":
    unittest.main()
