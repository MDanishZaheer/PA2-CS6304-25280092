# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Verify the 60-row audit uses the same prompts for all four policies."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from task4_safety.core.paths import POLICY_ORDER
from task4_safety.evaluation.manual_audit import (
    audit_sheet_path,
    create_manual_audit_sheet,
    fixed_audit_ids,
    validate_manual_sheet,
)
from task4_safety.core.protocol import load_xstest
from task4_safety.tests.helpers import temporary_config, write_synthetic_responses


class ManualAuditTests(unittest.TestCase):
    def test_fixed_subset_is_balanced_and_deterministic(self):
        config = temporary_config(Path(tempfile.mkdtemp()))
        xstest = load_xstest(config)
        first = fixed_audit_ids(xstest, 30, 6304)
        second = fixed_audit_ids(xstest, 30, 6304)
        self.assertEqual(first, second)
        selected = xstest[xstest["xstest_id"].isin(first)]
        self.assertEqual(len(selected), 60)
        self.assertEqual(int((selected["benchmark_class"] == "SAFE").sum()), 30)
        self.assertEqual(int((selected["benchmark_class"] == "UNSAFE").sum()), 30)

    def test_sheet_has_60_rows_four_labels_and_no_ai_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = temporary_config(Path(temporary))
            write_synthetic_responses(config)
            sheet = create_manual_audit_sheet(config)
            self.assertEqual(len(sheet), 60)
            for policy_name in POLICY_ORDER:
                self.assertIn(f"{policy_name}_response", sheet.columns)
                self.assertIn(f"{policy_name}_manual_label", sheet.columns)
            self.assertFalse(any("ai_label" in column for column in sheet.columns))
            status = validate_manual_sheet(config)
            self.assertEqual(status["required_response_labels"], 240)
            self.assertEqual(status["filled_response_labels"], 0)
            self.assertTrue(audit_sheet_path(config).exists())

            saved = pd.read_csv(audit_sheet_path(config), keep_default_na=False)
            saved.loc[0, "sft_manual_label"] = "SAFE_ANSWER"
            saved.to_csv(audit_sheet_path(config), index=False)
            reused = create_manual_audit_sheet(config)
            self.assertEqual(reused.loc[0, "sft_manual_label"], "SAFE_ANSWER")


if __name__ == "__main__":
    unittest.main()
