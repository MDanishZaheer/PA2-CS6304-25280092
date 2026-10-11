# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Run resumable PPO continuations from the supplied course midpoint."""

from __future__ import annotations

import gc
import math
import random
import shutil
import time

import torch
from torch.optim import AdamW

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import append_jsonl, load_json, save_json, set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.core.checkpoints import (
    directory_sha256,
    file_sha256,
    latest_update_checkpoint,
    load_saved_value_model,
    optimizer_to_parameter_devices,
    save_update_checkpoint,
)
from task2_ppo.core.paths import output_directory, result_directory
from task2_ppo.core.protocol import validate_shared_protocol
from task2_ppo.core.rollouts import collect_rollout, summarize_rollout, update_from_rollout


def validate_training_protocol(
    config: dict, run_name: str, updates: int, clip_epsilon: float, kl_beta: float
) -> None:
    """Reject settings that violate the fixed Task 2 comparison."""
    validate_shared_protocol(config)
    required_updates = int(config["updates"] if run_name == "standard" else config["fork_updates"])
    if int(updates) != required_updates:
        raise ValueError(f"{run_name} requires exactly {required_updates} updates.")
    if int(config["prompts_per_update"]) <= 0 or int(config["ppo_epochs"]) <= 0:
        raise ValueError("PPO prompts_per_update and ppo_epochs must be positive.")
    if float(clip_epsilon) <= 0 or float(kl_beta) < 0:
        raise ValueError("Clip epsilon must be positive and KL beta non-negative.")
    if run_name == "standard":
        if not math.isclose(float(clip_epsilon), float(config["clip_epsilon"])):
            raise ValueError("The standard run must use the configured clip epsilon.")
        if not math.isclose(float(kl_beta), float(config["kl_beta"])):
            raise ValueError("The standard run must use the configured KL beta.")
    elif run_name.startswith("clip_eps_"):
        named_epsilon = float(run_name.removeprefix("clip_eps_"))
        allowed = [float(value) for value in config["clip_values"]]
        if not any(math.isclose(named_epsilon, value) for value in allowed):
            raise ValueError(f"Unsupported clipping condition: {run_name}.")
        if not math.isclose(float(clip_epsilon), named_epsilon):
            raise ValueError("The run name and clip epsilon do not match.")
        if not math.isclose(float(kl_beta), float(config["kl_beta"])):
            raise ValueError("Clipping forks must keep the standard KL beta fixed.")
    elif run_name.startswith("kl_beta_"):
        named_beta = float(run_name.removeprefix("kl_beta_"))
        allowed = [float(value) for value in config["kl_values"]]
        if not any(math.isclose(named_beta, value) for value in allowed):
            raise ValueError(f"Unsupported KL condition: {run_name}.")
        if not math.isclose(float(kl_beta), named_beta):
            raise ValueError("The run name and KL beta do not match.")
        if not math.isclose(float(clip_epsilon), float(config["clip_epsilon"])):
            raise ValueError("KL forks must keep the standard clip epsilon fixed.")
    else:
        raise ValueError(f"Unknown Task 2 run name: {run_name}.")


def fixed_prompt_schedule(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Choose one deterministic prompt schedule shared by every continuation."""
    if count > len(rows):
        raise ValueError("The fixed PPO prompt pool is smaller than the requested budget.")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = [rows[index] for index in indices[:count]]
    if len({row["prompt_id"] for row in selected}) != len(selected):
        raise ValueError("The selected PPO prompt IDs must be unique.")
    return selected


def _protocol(
    config: dict,
    run_name: str,
    output_dir,
    updates: int,
    clip_epsilon: float,
    kl_beta: float,
    prompts: list[dict],
) -> dict:
    """Record every setting that defines one controlled continuation."""
    return {
        "run_name": run_name,
        "seed": int(config["seed"]),
        "base_model": config["base_model"],
        "reward_model": config["reward_model"],
        "reward_tokenizer": config.get("reward_tokenizer", config["base_model"]),
        "dtype": config["dtype"],
        "quantize_frozen_models": bool(config["quantize_frozen_models"]),
        "training_reward_device": "cpu",
        "training_reward_quantized": False,
        "midpoint_policy": str(repo_path(config["paths"]["ppo_midpoint_policy"])),
        "midpoint_policy_sha256": directory_sha256(config["paths"]["ppo_midpoint_policy"]),
        "midpoint_value": str(repo_path(config["paths"]["ppo_midpoint_value"])),
        "midpoint_value_sha256": directory_sha256(config["paths"]["ppo_midpoint_value"]),
        "output_dir": str(output_dir),
        "updates": int(updates),
        "prompts_per_update": int(config["prompts_per_update"]),
        "ppo_epochs": int(config["ppo_epochs"]),
        "prompt_ids": [row["prompt_id"] for row in prompts],
        "prompt_pool_sha256": file_sha256(config["paths"]["rl_prompt_train"]),
        "clip_epsilon": float(clip_epsilon),
        "kl_beta": float(kl_beta),
        "gamma": float(config["gamma"]),
        "gae_lambda": float(config["gae_lambda"]),
        "missing_eos_penalty": float(config["missing_eos_penalty"]),
        "value_coef": float(config["value_coef"]),
        "max_grad_norm": float(config["max_grad_norm"]),
        "max_prompt_length": int(config["max_prompt_length"]),
        "prompt_truncation_side": "left",
        "max_response_length": int(config["max_response_length"]),
        "reward_max_length": int(config["reward_max_length"]),
        "policy_learning_rate": float(config["policy_learning_rate"]),
        "value_lora_learning_rate": float(config["value_lora_learning_rate"]),
        "value_head_learning_rate": float(config["value_head_learning_rate"]),
        "value_train_mode": config["value_train_mode"],
        "lora": config["lora"],
        "value_lora": config["value_lora"],
        "generation": config["generation"],
    }


def _load_models_and_optimizers(config: dict, checkpoint_dir=None):
    """Load either the supplied midpoint or one verified partial continuation."""
    policy_source = (
        checkpoint_dir / "policy"
        if checkpoint_dir is not None
        else config["paths"]["ppo_midpoint_policy"]
    )
    policy = load_policy(config, adapter_path=str(policy_source), trainable=True)
    value_model = (
        load_saved_value_model(config, checkpoint_dir / "value")
        if checkpoint_dir is not None
        else load_value_model(
            config,
            config["paths"]["ppo_midpoint_value"],
            train_mode=config["value_train_mode"],
        )
    )
    print("Loading the frozen reward model on CPU to preserve GPU memory.")
    reward_model, reward_tokenizer = load_reward_model(config, device="cpu")
    tokenizer = load_tokenizer(config["base_model"])
    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(config["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(config["value_lora_learning_rate"]),
            head_lr=float(config["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )
    return (
        policy,
        value_model,
        reward_model,
        tokenizer,
        reward_tokenizer,
        policy_optimizer,
        value_optimizer,
    )


def _validate_completed_run(output_dir, result_dir) -> dict | None:
    """Return a summary only when its saved policy and critic still match."""
    summary_path = result_dir / "train_summary.json"
    if not summary_path.exists():
        return None
    summary = load_json(summary_path)
    if summary.get("policy_sha256") != directory_sha256(output_dir / "policy"):
        raise RuntimeError("The completed PPO policy changed after training.")
    if summary.get("value_sha256") != directory_sha256(output_dir / "value"):
        raise RuntimeError("The completed PPO critic changed after training.")
    history = read_jsonl(result_dir / "train_metrics.jsonl")
    expected_updates = list(range(1, int(summary["updates"]) + 1))
    if [int(row["update"]) for row in history] != expected_updates:
        raise RuntimeError("The completed PPO history is missing or reordered updates.")
    return summary


def train_ppo_continuation(
    config_path: str = "configs/ppo.yaml",
    run_name: str = "standard",
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    overwrite: bool = False,
) -> dict:
    """Train, checkpoint, or resume one fixed PPO continuation."""
    config = load_yaml(config_path)
    update_budget = int(config["updates"] if updates is None else updates)
    epsilon = float(config["clip_epsilon"] if clip_epsilon is None else clip_epsilon)
    beta = float(config["kl_beta"] if kl_beta is None else kl_beta)
    validate_training_protocol(config, run_name, update_budget, epsilon, beta)
    output_dir = output_directory(config, run_name)
    result_dir = result_directory(config, run_name)
    if overwrite:
        for directory in (output_dir, result_dir):
            if directory.exists():
                repository_root = repo_path(".").resolve()
                resolved = directory.resolve()
                if resolved == repository_root or repository_root not in resolved.parents:
                    raise ValueError(f"Refusing to remove unsafe run path: {resolved}")
                shutil.rmtree(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    prompt_pool = read_jsonl(config["paths"]["rl_prompt_train"])
    schedule = fixed_prompt_schedule(
        prompt_pool,
        update_budget * int(config["prompts_per_update"]),
        int(config["seed"]),
    )
    protocol = _protocol(
        config, run_name, output_dir, update_budget, epsilon, beta, schedule
    )
    run_config_path = result_dir / "run_config.json"
    if run_config_path.exists():
        saved_protocol = load_json(run_config_path)
        if saved_protocol != protocol:
            # A model-loading failure can leave only the protocol files behind.
            # Upgrade that empty legacy attempt to record the new operational
            # CPU placement without discarding any completed PPO update.
            legacy_protocol = dict(protocol)
            legacy_protocol.pop("training_reward_device")
            legacy_protocol.pop("training_reward_quantized")
            has_progress = (
                (output_dir / "checkpoints").exists()
                or (result_dir / "train_metrics.jsonl").exists()
            )
            if saved_protocol == legacy_protocol and not has_progress:
                save_json(run_config_path, protocol)
            else:
                raise ValueError(
                    f"Existing partial {run_name} run uses a different protocol."
                )
    unverified_artifacts = any((output_dir / name).exists() for name in (
        "policy", "value", "checkpoints"
    )) or (result_dir / "train_metrics.jsonl").exists()
    if not run_config_path.exists() and unverified_artifacts:
        raise RuntimeError(
            f"Existing {run_name} artifacts have no protocol file. "
            "Inspect them or rerun with overwrite=True."
        )
    save_json(run_config_path, protocol)
    save_json(
        result_dir / "train_prompt_ids.json",
        [{"prompt_id": row["prompt_id"], "source_index": row["source_index"]} for row in schedule],
    )
    completed = _validate_completed_run(output_dir, result_dir)
    if completed is not None:
        print(f"Using completed {run_name} PPO continuation.")
        return completed

    checkpoint_dir, state = latest_update_checkpoint(output_dir)
    resume = checkpoint_dir is not None
    set_seed(int(config["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    (
        policy,
        value_model,
        reward_model,
        tokenizer,
        reward_tokenizer,
        policy_optimizer,
        value_optimizer,
    ) = _load_models_and_optimizers(config, checkpoint_dir)
    if resume:
        policy_optimizer.load_state_dict(state["policy_optimizer"])
        value_optimizer.load_state_dict(state["value_optimizer"])
        optimizer_to_parameter_devices(policy_optimizer)
        optimizer_to_parameter_devices(value_optimizer)
    start_update = int(state.get("next_update", 0))
    history_path = result_dir / "train_metrics.jsonl"
    saved_history = read_jsonl(history_path) if history_path.exists() else []
    if len(saved_history) < start_update:
        raise RuntimeError("The PPO history is shorter than its saved checkpoint state.")
    expected_updates = list(range(1, start_update + 1))
    actual_updates = [int(row["update"]) for row in saved_history[:start_update]]
    if actual_updates != expected_updates:
        raise RuntimeError("The saved PPO history has missing or reordered updates.")
    if len(saved_history) != start_update:
        # A log line may be written immediately before a checkpoint interruption.
        # Keep only records backed by the last complete optimizer state.
        write_jsonl(history_path, saved_history[:start_update])
    previous_seconds = float(state.get("wall_time_seconds", 0.0))
    generated_tokens = int(state.get("generated_tokens", 0))
    peak_vram_gib = float(state.get("peak_vram_gib", 0.0))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    session_start = time.perf_counter()
    for update_index in range(start_update, update_budget):
        set_seed(int(config["seed"]) + update_index)
        first = update_index * int(config["prompts_per_update"])
        last = first + int(config["prompts_per_update"])
        rollout = collect_rollout(
            policy,
            value_model,
            reward_model,
            tokenizer,
            reward_tokenizer,
            schedule[first:last],
            {**config, "clip_epsilon": epsilon, "kl_beta": beta},
        )
        update_metrics = update_from_rollout(
            policy,
            value_model,
            policy_optimizer,
            value_optimizer,
            rollout,
            {**config, "clip_epsilon": epsilon, "kl_beta": beta},
        )
        rollout_metrics = summarize_rollout(rollout)
        generated_tokens += rollout_metrics["generated_tokens"]
        elapsed = previous_seconds + time.perf_counter() - session_start
        if torch.cuda.is_available():
            peak_vram_gib = max(
                peak_vram_gib,
                torch.cuda.max_memory_allocated() / (1024**3),
            )
        record = {
            "run_name": run_name,
            "update": update_index + 1,
            "prompt_ids": [row["prompt_id"] for row in schedule[first:last]],
            **rollout_metrics,
            **update_metrics,
            "cumulative_generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed,
        }
        append_jsonl(history_path, record)
        save_update_checkpoint(output_dir, update_index + 1, policy, value_model, {
            "next_update": update_index + 1,
            "policy_optimizer": policy_optimizer.state_dict(),
            "value_optimizer": value_optimizer.state_dict(),
            "wall_time_seconds": elapsed,
            "generated_tokens": generated_tokens,
            "peak_vram_gib": peak_vram_gib,
        })
        print(
            f"{run_name}: update {update_index + 1}/{update_budget}, "
            f"reward={record['effective_reward']:.3f}, "
            f"KL={record['sampled_kl_token_mean']:.5f}, "
            f"policy_loss={record['policy_loss']:.4f}",
            flush=True,
        )
        del rollout
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_seconds = previous_seconds + time.perf_counter() - session_start
    policy.save_pretrained(output_dir / "policy")
    tokenizer.save_pretrained(output_dir / "policy")
    value_model.save_pretrained(output_dir / "value")
    summary = {
        "run_name": run_name,
        "updates": update_budget,
        "prompts_per_update": int(config["prompts_per_update"]),
        "ppo_epochs": int(config["ppo_epochs"]),
        "clip_epsilon": epsilon,
        "kl_beta": beta,
        "training_reward_device": "cpu",
        "training_reward_quantized": False,
        "seed": int(config["seed"]),
        "generated_tokens": generated_tokens,
        "maximum_generation_budget": (
            update_budget
            * int(config["prompts_per_update"])
            * int(config["max_response_length"])
        ),
        "wall_time_seconds": total_seconds,
        "peak_vram_gib": peak_vram_gib,
        "policy_directory": str(output_dir / "policy"),
        "value_directory": str(output_dir / "value"),
        "policy_sha256": directory_sha256(output_dir / "policy"),
        "value_sha256": directory_sha256(output_dir / "value"),
    }
    save_json(result_dir / "checkpoint_manifest.json", {
        "policy_sha256": summary["policy_sha256"],
        "value_sha256": summary["value_sha256"],
        "source_midpoint_policy_sha256": protocol["midpoint_policy_sha256"],
        "source_midpoint_value_sha256": protocol["midpoint_value_sha256"],
    })
    save_json(result_dir / "train_summary.json", summary)
    del policy_optimizer, value_optimizer, policy, value_model, reward_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
