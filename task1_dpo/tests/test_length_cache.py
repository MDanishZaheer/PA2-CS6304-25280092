# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check that a completed length study does not rerun model evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.logging_utils import save_json
from task1_dpo.analyze_length import (
    LENGTH_RESULT_FILES,
    _length_protocol,
    _load_completed_length_results,
    run_length_study,
)


class LengthCacheTests(unittest.TestCase):
    def test_completed_study_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standard = root / "standard"
            balanced = root / "balanced"
            for adapter in (standard, balanced):
                adapter.mkdir()
                (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            result_dir = root / "results" / "length"
            result_dir.mkdir(parents=True)
            config = {
                "standard_output": str(standard),
                "length_output": str(balanced),
                "results_dir": str(root / "results"),
                "paths": {
                    "dpo_standard_train": "standard-train",
                    "dpo_length_train": "balanced-train",
                    "dpo_length_eval": "length-eval",
                    "word_limit_prompts": "word-limits",
                },
                "seed": 6304,
                "base_model": "offline-model",
                "beta": 0.1,
                "max_sequence_length": 768,
                "max_generation_tokens": 256,
                "generation": {"temperature": 0.7, "top_p": 0.9, "do_sample": True},
            }
            rows = {
                "standard-train": [{"prompt_id": "s1"}],
                "balanced-train": [{"prompt_id": "b1"}],
                "length-eval": [{"prompt_id": "e1", "source_index": 1}],
                "word-limits": [{"prompt_id": "w1"}],
            }
            fake_hash = lambda path: "hash-" + Path(path).name
            results = {"dataset_lengths": [], "strata": [], "word_limits": []}
            with patch("task1_dpo.analyze_length.adapter_sha256", side_effect=fake_hash):
                protocol = _length_protocol(
                    config, standard, balanced, rows["length-eval"], rows["word-limits"]
                )
            for name in LENGTH_RESULT_FILES:
                (result_dir / name).write_text("saved", encoding="utf-8")
            save_json(result_dir / "study_summary.json", {"protocol": protocol, "results": results})

            with patch("task1_dpo.analyze_length.load_yaml", return_value=config), patch(
                "task1_dpo.analyze_length.read_jsonl", side_effect=lambda path: rows[path]
            ), patch("task1_dpo.analyze_length.validate_saved_run"), patch(
                "task1_dpo.analyze_length.adapter_sha256", side_effect=fake_hash
            ), patch("task1_dpo.analyze_length.train_dpo") as train, patch(
                "task1_dpo.analyze_length.load_tokenizer"
            ) as tokenizer, patch("task1_dpo.analyze_length.load_policy") as policy:
                self.assertEqual(run_length_study(), results)
                train.assert_not_called()
                tokenizer.assert_not_called()
                policy.assert_not_called()

            (result_dir / "stratum_metrics.csv").unlink()
            self.assertIsNone(_load_completed_length_results(result_dir, protocol))
            self.assertIsNone(_load_completed_length_results(result_dir, {"seed": 7}))


if __name__ == "__main__":
    unittest.main()
