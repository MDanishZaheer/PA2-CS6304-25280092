# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Run resumable standard, canonical, and Dr-GRPO continuations."""

from __future__ import annotations

import gc
import random
import shutil
import time

import torch
from torch.optim import AdamW

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import append_jsonl, load_json, save_json, set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, trainable_parameters
from task3_grpo.core.checkpoints import (
    directory_sha256,
    file_sha256,
    latest_update_checkpoint,
    optimizer_to_parameter_devices,
    save_update_checkpoint,
)
from task3_grpo.core.paths import output_directory, result_directory
from task3_grpo.core.protocol import validate_training_protocol
from task3_grpo.core.rollouts import (
    collect_rollout,
    completion_records,
    summarize_rollout,
    update_from_rollout,
)


def fixed_prompt_schedule(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Choose the deterministic schedule shared by every matched continuation."""
    if count > len(rows):
        raise ValueError("The Task 3 prompt pool is smaller than the requested budget.")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = [rows[index] for index in indices[:count]]
    if len({row["prompt_id"] for row in selected}) != len(selected):
        raise ValueError("The selected Task 3 prompt IDs must be unique.")
    return selected


def _protocol(
    config: dict,
    run_name: str,
    updates: int,
    loss_type: str,
    schedule: list[dict],
    output_dir,
) -> dict:
    """Record all choices needed to audit a controlled continuation."""
    return {
        "run_name": run_name,
        "loss_type": loss_type,
        "seed": int(config["seed"]),
        "base_model": config["base_model"],
        "reward_model": config["reward_model"],
        "reward_tokenizer": config.get("reward_tokenizer", config["base_model"]),
        "dtype": config["dtype"],
        "training_reward_device": "cpu",
        "training_reward_quantized": False,
        "midpoint_policy": str(repo_path(config["paths"]["grpo_midpoint_policy"])),
        "midpoint_policy_sha256": directory_sha256(
            config["paths"]["grpo_midpoint_policy"]
        ),
        "output_directory": str(output_dir),
        "updates": int(updates),
        "prompts_per_update": int(config["prompts_per_update"]),
        "policy_epochs": int(config["policy_epochs"]),
        "num_generations": int(config["num_generations"]),
        "prompt_ids": [row["prompt_id"] for row in schedule],
        "prompt_pool_sha256": file_sha256(config["paths"]["rl_prompt_train"]),
        "learning_rate": float(config["learning_rate"]),
        "clip_epsilon": float(config["clip_epsilon"]),
        "kl_beta": float(config["kl_beta"]),
        "max_grad_norm": float(config["max_grad_norm"]),
        "max_prompt_length": int(config["max_prompt_length"]),
        "max_completion_length": int(config["max_completion_length"]),
        "mask_truncated_completions": bool(config["mask_truncated_completions"]),
        "reward_max_length": int(config.get("reward_max_length", 1024)),
        "lora": config["lora"],
        "generation": config["generation"],
    }


def _safe_remove(directory) -> None:
    repository_root = repo_path(".").resolve()
    target = directory.resolve()
    if target == repository_root or repository_root not in target.parents:
        raise ValueError(f"Refusing to remove unsafe run path: {target}")
    shutil.rmtree(directory)


def _completed_run(
    output_dir,
    result_dir,
    updates: int,
    prompts_per_update: int,
    num_generations: int,
) -> dict | None:
    summary_path = result_dir / "train_summary.json"
    if not summary_path.exists():
        return None
    summary = load_json(summary_path)
    if int(summary.get("updates", -1)) != int(updates):
        raise RuntimeError("The completed GRPO summary has the wrong update budget.")
    if summary.get("policy_sha256") != directory_sha256(output_dir / "policy"):
        raise RuntimeError("The completed GRPO policy changed after training.")
    history = read_jsonl(result_dir / "train_metrics.jsonl")
    if [int(row["update"]) for row in history] != list(range(1, updates + 1)):
        raise RuntimeError("The completed GRPO history is incomplete or reordered.")
    completions = read_jsonl(result_dir / "completion_metrics.jsonl")
    expected_completions = updates * prompts_per_update * num_generations
    if len(completions) != expected_completions:
        raise RuntimeError("The completed GRPO run is missing completion diagnostics.")
    expected_update_counts = {
        update: prompts_per_update * num_generations
        for update in range(1, updates + 1)
    }
    actual_update_counts = {
        update: sum(int(row["update"]) == update for row in completions)
        for update in expected_update_counts
    }
    if actual_update_counts != expected_update_counts:
        raise RuntimeError("GRPO completion diagnostics are assigned to wrong updates.")
    return summary


def train_grpo_continuation(
    config_path: str = "configs/grpo.yaml",
    run_name: str = "standard",
    updates: int | None = None,
    loss_type: str = "grpo",
    overwrite: bool = False,
) -> dict:
    """Train, resume, or reuse one fixed Task 3 continuation."""
    config = load_yaml(config_path)
    update_budget = int(config["updates"] if updates is None else updates)
    validate_training_protocol(config, run_name, update_budget, loss_type)
    output_dir = output_directory(config, run_name)
    result_dir = result_directory(config, run_name)
    if overwrite:
        for directory in (output_dir, result_dir):
            if directory.exists():
                _safe_remove(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    prompt_pool = read_jsonl(config["paths"]["rl_prompt_train"])
    schedule = fixed_prompt_schedule(
        prompt_pool,
        update_budget * int(config["prompts_per_update"]),
        int(config["seed"]),
    )
    protocol = _protocol(
        config, run_name, update_budget, loss_type, schedule, output_dir
    )
    protocol_path = result_dir / "run_config.json"
    progress_exists = (
        (output_dir / "checkpoints").exists()
        or (output_dir / "policy").exists()
        or (result_dir / "train_metrics.jsonl").exists()
    )
    if protocol_path.exists() and load_json(protocol_path) != protocol:
        raise ValueError(f"Existing partial {run_name} run uses a different protocol.")
    if not protocol_path.exists() and progress_exists:
        raise RuntimeError(
            f"Existing {run_name} artifacts have no protocol file; inspect them "
            "or rerun with overwrite=True."
        )
    save_json(protocol_path, protocol)
    save_json(result_dir / "train_prompt_ids.json", [
        {"prompt_id": row["prompt_id"], "source_index": row.get("source_index")}
        for row in schedule
    ])
    completed = _completed_run(
        output_dir,
        result_dir,
        update_budget,
        int(config["prompts_per_update"]),
        int(config["num_generations"]),
    )
    if completed is not None:
        print(f"Using completed {run_name} GRPO continuation.")
        return completed

    checkpoint_dir, state = latest_update_checkpoint(output_dir)
    policy_source = (
        checkpoint_dir / "policy"
        if checkpoint_dir is not None
        else config["paths"]["grpo_midpoint_policy"]
    )
    set_seed(int(config["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    policy = load_policy(config, adapter_path=str(policy_source), trainable=True)
    tokenizer = load_tokenizer(config["base_model"])
    print("Loading the frozen reward model on CPU to preserve GPU memory.")
    reward_model, reward_tokenizer = load_reward_model(config, device="cpu")
    optimizer = AdamW(
        trainable_parameters(policy), lr=float(config["learning_rate"])
    )
    if checkpoint_dir is not None:
        optimizer.load_state_dict(state["optimizer"])
        optimizer_to_parameter_devices(optimizer)

    start_update = int(state.get("next_update", 0))
    history_path = result_dir / "train_metrics.jsonl"
    completion_path = result_dir / "completion_metrics.jsonl"
    history = read_jsonl(history_path) if history_path.exists() else []
    completions = read_jsonl(completion_path) if completion_path.exists() else []
    if len(history) < start_update:
        raise RuntimeError("GRPO history is shorter than its checkpoint state.")
    if [int(row["update"]) for row in history[:start_update]] != list(
        range(1, start_update + 1)
    ):
        raise RuntimeError("Saved GRPO history has missing or reordered updates.")
    if len(history) != start_update:
        write_jsonl(history_path, history[:start_update])
    backed_completions = [
        row for row in completions if int(row["update"]) <= start_update
    ]
    expected_completion_count = (
        start_update
        * int(config["prompts_per_update"])
        * int(config["num_generations"])
    )
    if len(backed_completions) != expected_completion_count:
        raise RuntimeError("Saved completion diagnostics do not match the checkpoint.")
    if len(backed_completions) != len(completions):
        write_jsonl(completion_path, backed_completions)

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
            reward_model,
            tokenizer,
            reward_tokenizer,
            schedule[first:last],
            config,
            loss_type,
        )
        update_metrics = update_from_rollout(
            policy, optimizer, rollout, config, loss_type
        )
        rollout_metrics = summarize_rollout(rollout)
        generated_tokens += int(rollout_metrics["generated_tokens"])
        elapsed = previous_seconds + time.perf_counter() - session_start
        if torch.cuda.is_available():
            peak_vram_gib = max(
                peak_vram_gib,
                torch.cuda.max_memory_allocated() / (1024**3),
            )
        record = {
            "run_name": run_name,
            "loss_type": loss_type,
            "update": update_index + 1,
            "prompt_ids": [row["prompt_id"] for row in schedule[first:last]],
            **rollout_metrics,
            **update_metrics,
            "cumulative_generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed,
        }
        append_jsonl(history_path, record)
        for completion in completion_records(rollout, update_index + 1, run_name):
            append_jsonl(completion_path, completion)
        save_update_checkpoint(output_dir, update_index + 1, policy, {
            "next_update": update_index + 1,
            "optimizer": optimizer.state_dict(),
            "wall_time_seconds": elapsed,
            "generated_tokens": generated_tokens,
            "peak_vram_gib": peak_vram_gib,
        })
        print(
            f"{run_name}: update {update_index + 1}/{update_budget}, "
            f"reward={record['reward_mean']:.3f}, "
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
    summary = {
        "run_name": run_name,
        "loss_type": loss_type,
        "updates": update_budget,
        "prompts_per_update": int(config["prompts_per_update"]),
        "num_generations": int(config["num_generations"]),
        "seed": int(config["seed"]),
        "generated_tokens": generated_tokens,
        "maximum_generation_budget": (
            update_budget
            * int(config["prompts_per_update"])
            * int(config["num_generations"])
            * int(config["max_completion_length"])
        ),
        "training_reward_device": "cpu",
        "wall_time_seconds": total_seconds,
        "peak_vram_gib": peak_vram_gib,
        "policy_directory": str(output_dir / "policy"),
        "policy_sha256": directory_sha256(output_dir / "policy"),
    }
    save_json(result_dir / "checkpoint_manifest.json", {
        "policy_sha256": summary["policy_sha256"],
        "source_midpoint_policy_sha256": protocol["midpoint_policy_sha256"],
    })
    save_json(result_dir / "train_summary.json", summary)
    del optimizer, policy, reward_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
