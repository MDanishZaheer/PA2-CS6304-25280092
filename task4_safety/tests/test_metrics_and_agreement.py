# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Exercise required rates, manual agreement, tables, and figures."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from common.data import read_jsonl, write_jsonl
from task4_safety.analysis.cross_task import build_cross_task_table
from task4_safety.analysis.plots import save_available_figures
from task4_safety.core.paths import (
    POLICY_ORDER,
    judgment_path,
    raw_results_directory,
)
from task4_safety.evaluation.agreement import build_manual_agreement
from task4_safety.evaluation.manual_audit import (
    audit_sheet_path,
    create_manual_audit_sheet,
)
from task4_safety.evaluation.qualitative import (
    qualitative_review_path,
    select_qualitative_examples,
)
from task4_safety.evaluation.safety_metrics import build_safety_tables
from task4_safety.tests.helpers import (
    temporary_config,
    write_synthetic_judgments,
    write_synthetic_responses,
)


class MetricsAndAgreementTests(unittest.TestCase):
    def test_required_metrics_and_complete_manual_agreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = temporary_config(Path(temporary))
            write_synthetic_responses(config)
            write_synthetic_judgments(config)
            tables = build_safety_tables(config)
            self.assertEqual(
                len(read_jsonl(
                    raw_results_directory(config) / "scored_responses.jsonl"
                )),
                1800,
            )
            summary = tables["policy_metrics"].set_index("policy")
            for policy_name in POLICY_ORDER:
                self.assertEqual(int(summary.loc[policy_name, "prompt_count"]), 450)
                self.assertAlmostEqual(
                    summary.loc[policy_name, "safe_answer_rate"], 1.0
                )
                self.assertAlmostEqual(
                    summary.loc[policy_name, "unsafe_justified_refusal_rate"], 1.0
                )
                self.assertAlmostEqual(
                    summary.loc[policy_name, "unsafe_compliance_rate"], 0.0
                )
            cross_task = build_cross_task_table(config, tables["policy_metrics"])
            self.assertEqual(cross_task["policy"].tolist(), list(POLICY_ORDER))
            self.assertTrue(
                cross_task.loc[
                    cross_task["policy"] == "dpo", "reward_model_score_mean"
                ].notna().all()
            )

            create_manual_audit_sheet(config)
            sheet_path = audit_sheet_path(config)
            sheet = pd.read_csv(sheet_path, keep_default_na=False)
            for index, row in sheet.iterrows():
                label = (
                    "SAFE_ANSWER"
                    if row["benchmark_class"] == "SAFE"
                    else "JUSTIFIED_REFUSAL"
                )
                for policy_name in POLICY_ORDER:
                    sheet.loc[index, f"{policy_name}_manual_label"] = label
            sheet.to_csv(sheet_path, index=False)
            result = build_manual_agreement(config, require_complete=True)
            overall = result["agreement"].set_index("policy").loc["all"]
            self.assertEqual(int(overall["sample_count"]), 240)
            self.assertAlmostEqual(overall["agreement_rate"], 1.0)
            figures = save_available_figures(config)
            self.assertEqual(len(figures), 4)
            self.assertTrue(all(path.exists() for path in figures))

    def test_qualitative_examples_prefer_the_fixed_manual_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = temporary_config(Path(temporary))
            write_synthetic_responses(config)
            write_synthetic_judgments(config)
            sheet = create_manual_audit_sheet(config)
            safe_id = int(
                sheet.loc[sheet["benchmark_class"] == "SAFE", "xstest_id"].iloc[0]
            )
            unsafe_id = int(
                sheet.loc[
                    sheet["benchmark_class"] == "UNSAFE", "xstest_id"
                ].iloc[0]
            )
            for policy_name in POLICY_ORDER:
                path = judgment_path(config, policy_name)
                rows = read_jsonl(path)
                for row in rows:
                    if row["xstest_id"] == safe_id and policy_name == "sft":
                        row["ai_label"] = "OVER_REFUSAL"
                    if row["xstest_id"] == unsafe_id and policy_name == "dpo":
                        row["ai_label"] = "UNSAFE_COMPLIANCE"
                write_jsonl(path, rows)

            sheet_path = audit_sheet_path(config)
            sheet = pd.read_csv(sheet_path, keep_default_na=False)
            labels = {
                (row["xstest_id"], row["policy"]): row["ai_label"]
                for policy_name in POLICY_ORDER
                for row in read_jsonl(judgment_path(config, policy_name))
            }
            for index, row in sheet.iterrows():
                for policy_name in POLICY_ORDER:
                    sheet.loc[index, f"{policy_name}_manual_label"] = labels[
                        (int(row["xstest_id"]), policy_name)
                    ]
            sheet.to_csv(sheet_path, index=False)
            agreement = build_manual_agreement(config, require_complete=True)
            examples = select_qualitative_examples(
                config, agreement["comparison"]
            )
            self.assertEqual(
                {row["selection_target"] for row in examples},
                {"UNSAFE_COMPLIANCE", "JUSTIFIED_REFUSAL", "OVER_REFUSAL"},
            )
            self.assertTrue(
                all(row["covered_by_fixed_manual_audit"] for row in examples)
            )
            review_path = qualitative_review_path(config)
            review = pd.read_csv(
                review_path, encoding="utf-8-sig", keep_default_na=False
            )
            original_response = review.loc[0, "sft_response"]
            review.loc[0, "review_complete"] = True
            review.loc[0, "report_comment"] = "Judge’s review is retained."
            review.loc[0, "sft_response"] = "accidentally changed"
            review.to_csv(review_path, index=False, encoding="cp1252")

            rerun = select_qualitative_examples(
                config, agreement["comparison"]
            )
            self.assertTrue(rerun[0]["review_complete"])
            self.assertEqual(
                rerun[0]["report_comment"], "Judge’s review is retained."
            )
            self.assertEqual(review_path.read_bytes()[:3], b"\xef\xbb\xbf")
            repaired = pd.read_csv(
                review_path, encoding="utf-8-sig", keep_default_na=False
            )
            self.assertEqual(repaired.loc[0, "sft_response"], original_response)


if __name__ == "__main__":
    unittest.main()
