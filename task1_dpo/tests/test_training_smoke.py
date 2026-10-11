# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Exercise one complete DPO optimizer update with a tiny offline model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from task1_dpo.core.trainer import train_dpo
from task1_dpo.evaluate import run_evaluation


class TinyTokenizer:
    """Provide the minimal tokenizer operations used by the DPO collator."""

    eos_token = "~"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2] if tokenize else "user: " + messages[-1]["content"]

    eos_token_id = 2

    def __call__(
        self, text, add_special_tokens=False, return_tensors=None,
        padding=False, truncation=False, max_length=None,
    ):
        if isinstance(text, str):
            return {"input_ids": [3 + (ord(character) % 20) for character in text]}
        sequences = [[3 + (ord(character) % 20) for character in item] for item in text]
        if truncation and max_length is not None:
            sequences = [tokens[-max_length:] for tokens in sequences]
        width = max(map(len, sequences))
        ids = [[0] * (width - len(tokens)) + tokens for tokens in sequences]
        mask = [[0] * (width - len(tokens)) + [1] * len(tokens) for tokens in sequences]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + int(token) % 26) for token in ids if int(token) != 0)

    def save_pretrained(self, path):
        Path(path, "tiny_tokenizer.txt").write_text("offline test", encoding="utf-8")


def tiny_policy(_config, trainable, fresh_lora=False, adapter_path=None):
    """Return a small PEFT model with a working disable_adapter context."""
    base = GPT2LMHeadModel(
        GPT2Config(vocab_size=32, n_positions=32, n_ctx=32, n_embd=16, n_layer=1, n_head=2)
    )
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=trainable)
    else:
        model = get_peft_model(
            base,
            LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"),
        )
    if torch.cuda.is_available():
        model = model.cuda()
    model.train() if trainable else model.eval()
    return model


class TinyReward(torch.nn.Module):
    """Return a deterministic scalar for the evaluation plumbing test."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=input_ids.float().sum(dim=1, keepdim=True) / 100)


class TrainingSmokeTests(unittest.TestCase):
    """Confirm the fixed-budget loop saves an adapter and machine-readable log."""

    def test_two_example_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_file = root / "pairs.jsonl"
            rows = [
                {"prompt_id": f"p{index}", "source_index": index, "prompt": "why?",
                 "chosen": "good", "rejected": "bad"}
                for index in range(2)
            ]
            data_file.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            config_file = root / "config.yaml"
            config_file.write_text(yaml.safe_dump({
                "seed": 6304,
                "base_model": "tiny-offline",
                "reward_model": "tiny-reward",
                "reward_tokenizer": "tiny-offline",
                "batch_size": 2,
                "grad_accum_steps": 1,
                "epochs": 1,
                "max_sequence_length": 24,
                "max_generation_tokens": 4,
                "generation": {"temperature": 0.7, "top_p": 0.9, "do_sample": True},
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "beta": 0.1,
                "standard_output": str(root / "adapter"),
                "results_dir": str(root / "results"),
                "paths": {"dpo_standard_train": str(data_file)},
            }), encoding="utf-8")
            with patch("task1_dpo.core.trainer.load_tokenizer", return_value=TinyTokenizer()), patch(
                "task1_dpo.core.trainer.load_policy", side_effect=tiny_policy
            ):
                summary = train_dpo(str(config_file), run_name="smoke")
            self.assertEqual(summary["updates"], 1)
            self.assertEqual(summary["training_examples"], 2)
            self.assertTrue((root / "adapter" / "adapter_model.safetensors").exists())
            records = (root / "results" / "smoke" / "train_metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(records), 1)
            self.assertTrue(torch.isfinite(torch.tensor(json.loads(records[0])["train_loss"])))
            with patch("task1_dpo.evaluate.load_tokenizer", return_value=TinyTokenizer()), patch(
                "task1_dpo.evaluate.load_policy", side_effect=tiny_policy
            ), patch(
                "task1_dpo.evaluate.load_reward_model",
                return_value=(TinyReward(), TinyTokenizer()),
            ):
                metrics = run_evaluation(
                    str(config_file), str(root / "adapter"), "smoke", str(data_file)
                )
                with patch(
                    "task1_dpo.evaluate.generate_comparison_records",
                    side_effect=AssertionError("Saved generations should be reused"),
                ):
                    cached_metrics = run_evaluation(
                        str(config_file), str(root / "adapter"), "smoke", str(data_file)
                    )
            self.assertEqual(metrics["preference_pairs"], 2)
            self.assertEqual(metrics["generated_prompts"], 2)
            self.assertEqual(metrics["sampled_kl_token_mean"], cached_metrics["sampled_kl_token_mean"])
            self.assertTrue((root / "results" / "smoke" / "generations.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
