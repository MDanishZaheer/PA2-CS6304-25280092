# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Validate assignment-fixed Task 2 models, budgets, and decoding settings."""

from __future__ import annotations

import math


def _same_numbers(actual, expected) -> bool:
    """Compare two short numeric lists without depending on YAML scalar types."""
    return len(actual) == len(expected) and all(
        math.isclose(float(left), float(right))
        for left, right in zip(actual, expected)
    )


def validate_shared_protocol(config: dict) -> None:
    """Reject accidental changes to the released Task 2 protocol."""
    expected_scalars = {
        "seed": 6304,
        "updates": 20,
        "fork_updates": 8,
        "prompts_per_update": 1,
        "ppo_epochs": 2,
        "policy_learning_rate": 3.0e-6,
        "value_lora_learning_rate": 1.0e-4,
        "value_head_learning_rate": 3.0e-4,
        "max_prompt_length": 256,
        "max_response_length": 512,
        "eval_max_response_length": 768,
        "reward_max_length": 1280,
        "clip_epsilon": 0.20,
        "kl_beta": 0.10,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "value_coef": 0.50,
        "missing_eos_penalty": 1.0,
        "max_grad_norm": 1.0,
    }
    for key, expected in expected_scalars.items():
        actual = config.get(key)
        if isinstance(expected, float):
            valid = actual is not None and math.isclose(float(actual), expected)
        else:
            valid = actual is not None and int(actual) == expected
        if not valid:
            raise ValueError(f"Task 2 requires {key}={expected!r}; found {actual!r}.")
    if config.get("base_model") != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("Task 2 base_model does not match the course release.")
    if config.get("reward_model") != "yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback":
        raise ValueError("Task 2 reward_model does not match the course release.")
    if config.get("reward_tokenizer") != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("Task 2 reward_tokenizer does not match the course release.")
    if str(config.get("dtype", "")).lower() != "float16":
        raise ValueError("Task 2 requires float16 model loading.")
    if not bool(config.get("quantize_frozen_models")):
        raise ValueError("Task 2 requires quantization of the frozen reward model.")
    if config.get("value_train_mode") != "lora_head":
        raise ValueError("Task 2 requires the released lora_head critic mode.")
    for section in ("lora", "value_lora"):
        settings = config.get(section, {})
        if (
            int(settings.get("r", -1)) != 8
            or int(settings.get("alpha", -1)) != 16
            or not math.isclose(float(settings.get("dropout", -1)), 0.05)
            or set(settings.get("target_modules", [])) != {"q_proj", "v_proj"}
        ):
            raise ValueError(f"Task 2 {section} settings do not match the release.")
    if not _same_numbers(config.get("clip_values", []), [0.05, 0.20, 0.50]):
        raise ValueError("Task 2 clipping conditions must be 0.05, 0.20, and 0.50.")
    if not _same_numbers(config.get("kl_values", []), [0.00, 0.10, 0.20]):
        raise ValueError("Task 2 KL conditions must be 0.00, 0.10, and 0.20.")
    generation = config.get("generation", {})
    if (
        not math.isclose(float(generation.get("temperature", -1)), 0.7)
        or not math.isclose(float(generation.get("top_p", -1)), 0.9)
        or generation.get("do_sample") is not True
    ):
        raise ValueError("Task 2 requires temperature=0.7, top_p=0.9 sampling.")
