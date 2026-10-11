# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check that saved Task 3 logs can produce every final table and figure."""

import tempfile
import unittest
from pathlib import Path

from common.data import load_yaml, write_jsonl
from common.logging_utils import save_json
from task3_grpo.analysis.aggregate import build_available_evidence
from task3_grpo.analysis.plots import save_available_figures
from task3_grpo.core.paths import figures_directory, result_directory


class EvidencePipelineTests(unittest.TestCase):
    def _evaluation_metrics(self) -> dict:
        return {
            "reward_model_score_mean": 1.5,
            "reward_model_score_std": 0.2,
            "sampled_kl_token_mean": 0.03,
            "entropy_token_mean": 2.0,
            "response_length_mean": 250.0,
            "response_length_std": 80.0,
            "truncated_fraction": 0.1,
        }

    def _training_protocol(self, loss_type: str) -> dict:
        return {
            "run_name": "synthetic",
            "loss_type": loss_type,
            "seed": 6304,
            "base_model": "base",
            "reward_model": "reward",
            "reward_tokenizer": "base",
            "dtype": "float16",
            "training_reward_device": "cpu",
            "training_reward_quantized": False,
            "midpoint_policy": "midpoint",
            "midpoint_policy_sha256": "abc",
            "updates": 8,
            "prompts_per_update": 1,
            "policy_epochs": 1,
            "num_generations": 4,
            "prompt_ids": ["train-prompt"],
            "prompt_pool_sha256": "pool",
            "learning_rate": 5e-6,
            "clip_epsilon": 0.2,
            "kl_beta": 0.1,
            "max_grad_norm": 1.0,
            "max_prompt_length": 256,
            "max_completion_length": 512,
            "mask_truncated_completions": True,
            "reward_max_length": 1024,
            "lora": {"r": 8},
            "generation": {"temperature": 0.7, "top_p": 0.9, "do_sample": True},
        }

    def _history_row(self, run_name: str) -> dict:
        return {
            "run_name": run_name,
            "update": 1,
            "reward_mean": 1.0,
            "sampled_kl_token_mean": 0.02,
            "within_group_reward_std_mean": 0.4,
            "uninformative_group_fraction": 0.0,
            "policy_loss": -0.1,
            "gradient_norm": 0.5,
            "entropy_token_mean": 2.1,
            "response_length_mean": 220.0,
            "truncated_fraction": 0.0,
        }

    def _generation_rows(self, reward_offset: float) -> list[dict]:
        return [{
            "prompt_id": "eval-prompt",
            "source_index": 1,
            "prompt": "Give a concise answer.",
            "response": "A concise answer.",
            "reward_model_score": 1.0 + reward_offset,
            "sampled_kl_token_mean": 0.02 + reward_offset,
            "entropy_token_mean": 2.0,
            "response_length": 20 + int(reward_offset * 10),
        }]

    def test_saved_artifacts_build_all_tables_and_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = load_yaml("configs/grpo.yaml")
            config["results_dir"] = str(Path(temporary) / "results")

            standard = result_directory(config, "standard")
            write_jsonl(standard / "train_metrics.jsonl", [
                self._history_row("standard")
            ])
            save_json(standard / "train_summary.json", {
                "updates": 20,
                "num_generations": 4,
                "generated_tokens": 1000,
                "wall_time_seconds": 100.0,
                "peak_vram_gib": 4.0,
            })
            save_json(standard / "eval_metrics.json", self._evaluation_metrics())

            for run_name, loss_type, reward_offset in (
                ("norm_canonical", "grpo", 0.0),
                ("norm_dr_grpo", "dr_grpo", 0.1),
            ):
                directory = result_directory(config, run_name)
                protocol = self._training_protocol(loss_type)
                protocol["run_name"] = run_name
                save_json(directory / "run_config.json", protocol)
                save_json(directory / "train_summary.json", {
                    "loss_type": loss_type,
                    "updates": 8,
                    "maximum_generation_budget": 16384,
                    "generated_tokens": 800,
                    "wall_time_seconds": 80.0,
                    "peak_vram_gib": 4.1,
                })
                write_jsonl(directory / "train_metrics.jsonl", [
                    self._history_row(run_name)
                ])
                write_jsonl(directory / "completion_metrics.jsonl", [
                    {
                        "response_length": length,
                        "training_tokens": length,
                        "reward": 1.0,
                        "advantage": 1.0,
                        "per_token_gradient_weight": 1.0 / length,
                        "sequence_gradient_weight_proxy": (
                            1.0 if loss_type == "grpo" else length / 512
                        ),
                        "truncated": False,
                    }
                    for length in (100, 250, 400)
                ])
                save_json(
                    directory / "eval_metrics.json", self._evaluation_metrics()
                )
                write_jsonl(
                    directory / "generations.jsonl",
                    self._generation_rows(reward_offset),
                )

            built = build_available_evidence(config)
            save_available_figures(config)
            self.assertIn("standard", built)
            self.assertIn("normalization", built)
            self.assertTrue((standard / "summary.csv").exists())
            normalization = result_directory(config, "norm_canonical").parent
            self.assertTrue((normalization / "summary.csv").exists())
            self.assertTrue(
                (normalization / "length_conditioned_metrics.csv").exists()
            )
            self.assertTrue(
                (normalization / "paired_heldout_differences.csv").exists()
            )
            self.assertTrue(
                (figures_directory(config) / "standard_training.png").exists()
            )
            self.assertTrue(
                (figures_directory(config) / "normalization_heldout.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
