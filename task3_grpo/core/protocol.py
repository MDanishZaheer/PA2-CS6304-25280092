# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Protect the assignment-fixed Task 3 experimental protocol."""

from __future__ import annotations

import math


def _same_numbers(actual, expected) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(float(left), float(right))
        for left, right in zip(actual, expected)
    )


def validate_shared_protocol(config: dict) -> None:
    """Reject accidental edits to released models, budgets, and decoding."""
    expected = {
        "seed": 6304,
        "updates": 20,
        "fork_updates": 8,
        "prompts_per_update": 1,
        "policy_epochs": 1,
        "learning_rate": 5.0e-6,
        "clip_epsilon": 0.20,
        "kl_beta": 0.10,
        "num_generations": 4,
        "max_prompt_length": 256,
        "max_completion_length": 512,
        "max_grad_norm": 1.0,
        "cache_generation_cap": 768,
    }
    for key, required in expected.items():
        actual = config.get(key)
        valid = (
            actual is not None
            and math.isclose(float(actual), float(required))
        )
        if not valid:
            raise ValueError(
                f"Task 3 requires {key}={required!r}; found {actual!r}."
            )
    if config.get("mask_truncated_completions") is not True:
        raise ValueError("Task 3 must mask max-length completions from training.")
    if not _same_numbers(config.get("group_sizes", []), [2, 4, 8]):
        raise ValueError("Task 3 group sizes must be K=2, 4, and 8.")
    if config.get("base_model") != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("Task 3 base model does not match the course release.")
    if config.get("reward_model") != "yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback":
        raise ValueError("Task 3 reward model does not match the course release.")
    if config.get("reward_tokenizer") != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("Task 3 reward tokenizer does not match the release.")
    if str(config.get("dtype", "")).lower() != "float16":
        raise ValueError("Task 3 requires float16 model loading.")
    if config.get("quantize_frozen_models") is not True:
        raise ValueError("Task 3 requires quantized frozen models when on GPU.")
    lora = config.get("lora", {})
    if (
        int(lora.get("r", -1)) != 8
        or int(lora.get("alpha", -1)) != 16
        or not math.isclose(float(lora.get("dropout", -1)), 0.05)
        or set(lora.get("target_modules", [])) != {"q_proj", "v_proj"}
    ):
        raise ValueError("Task 3 LoRA settings do not match the release.")
    generation = config.get("generation", {})
    if (
        not math.isclose(float(generation.get("temperature", -1)), 0.7)
        or not math.isclose(float(generation.get("top_p", -1)), 0.9)
        or generation.get("do_sample") is not True
    ):
        raise ValueError("Task 3 requires temperature=0.7, top_p=0.9 sampling.")


def validate_training_protocol(
    config: dict,
    run_name: str,
    updates: int,
    loss_type: str,
) -> None:
    """Validate one standard or normalization continuation."""
    validate_shared_protocol(config)
    conditions = {
        "standard": (int(config["updates"]), "grpo"),
        "norm_canonical": (int(config["fork_updates"]), "grpo"),
        "norm_dr_grpo": (int(config["fork_updates"]), "dr_grpo"),
    }
    if run_name not in conditions:
        raise ValueError(f"Unknown Task 3 run name: {run_name!r}.")
    required_updates, required_loss = conditions[run_name]
    if int(updates) != required_updates:
        raise ValueError(f"{run_name} requires exactly {required_updates} updates.")
    if loss_type != required_loss:
        raise ValueError(f"{run_name} requires loss_type={required_loss!r}.")
