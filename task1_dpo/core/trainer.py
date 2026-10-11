# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Train a DPO LoRA adapter for the assignment's fixed epoch budget."""

from __future__ import annotations

import math
import time
import hashlib
import gc

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import append_jsonl, load_json, save_json, set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.core.batching import make_preference_collator
from task1_dpo.core.log_probs import response_sequence_log_probs
from task1_dpo.core.paths import result_directory
from task1_dpo.dpo import dpo_loss


def validate_training_config(config: dict, rows: list[dict], max_examples: int | None) -> None:
    """Reject settings or data that would make Task 1 comparisons invalid."""
    for key in ("batch_size", "grad_accum_steps", "epochs", "max_sequence_length"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if int(config["epochs"]) != 1:
        raise ValueError("Task 1 requires exactly one training epoch.")
    if max_examples is not None and not 0 < max_examples <= len(rows):
        raise ValueError("max_examples must be between 1 and the dataset size.")
    if not rows:
        raise ValueError("The DPO training dataset is empty.")
    required = {"prompt_id", "source_index", "chosen", "rejected"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"Every training row must contain {sorted(required)}.")
    if len({row["prompt_id"] for row in rows}) != len(rows):
        raise ValueError("Training prompt IDs must be unique.")


def _to_device(batch: dict, device: torch.device) -> dict:
    """Move a padded preference batch to the policy device."""
    return {key: value.to(device) for key, value in batch.items()}


def _one_example(batch: dict, index: int) -> dict:
    """Select one pair member and remove its left padding to reduce GPU memory."""
    first_token = int(batch["attention_mask"][index].nonzero()[0])
    return {
        key: value[index : index + 1, first_token:]
        for key, value in batch.items()
    }


def adapter_sha256(adapter_dir: str) -> str:
    """Hash the saved LoRA weight file without loading it into memory."""
    path = repo_path(adapter_dir) / "adapter_model.safetensors"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_saved_run(
    config: dict, run_name: str, adapter_dir: str, dataset_path: str,
    beta: float, training_examples: int,
) -> None:
    """Reject an existing adapter whose recorded protocol differs from this run."""
    results_dir = result_directory(config, run_name)
    saved = load_json(results_dir / "run_config.json")
    expected = {
        "seed": int(config["seed"]),
        "base_model": config["base_model"],
        "lora": config["lora"],
        "batch_size": int(config["batch_size"]),
        "grad_accum_steps": int(config["grad_accum_steps"]),
        "max_sequence_length": int(config["max_sequence_length"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "beta": float(beta),
        "training_examples": int(training_examples),
        "training_data": str(repo_path(dataset_path)),
        "adapter_dir": str(repo_path(adapter_dir)),
    }
    differences = [key for key, value in expected.items() if saved.get(key) != value]
    if differences:
        raise ValueError(f"Existing {run_name} adapter has mismatched settings: {differences}")
    summary = load_json(results_dir / "train_summary.json")
    if summary["adapter_sha256"] != adapter_sha256(adapter_dir):
        raise RuntimeError(f"Existing {run_name} adapter weights changed after training.")


def train_dpo(
    config_path: str = "configs/dpo.yaml",
    run_name: str = "standard",
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Train one fresh adapter and save its final state plus update logs."""
    config = load_yaml(config_path)
    training_data = dataset_path or config["paths"]["dpo_standard_train"]
    rows = read_jsonl(training_data)
    validate_training_config(config, rows, max_examples)
    if max_examples is not None:
        rows = rows[:max_examples]
    beta_value = float(config["beta"] if beta is None else beta)
    if beta_value <= 0:
        raise ValueError("DPO beta must be positive.")
    adapter_dir = repo_path(output_path or config["standard_output"])
    results_dir = result_directory(config, run_name)
    if (adapter_dir / "adapter_model.safetensors").exists() and not overwrite:
        raise FileExistsError(f"Adapter already exists: {adapter_dir}. Use --overwrite to replace it.")
    history_file = results_dir / "train_metrics.jsonl"
    if history_file.exists() and not overwrite:
        raise FileExistsError(f"Training log already exists: {history_file}. Use --overwrite to replace it.")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if history_file.exists():
        history_file.unlink()

    seed = int(config["seed"])
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tokenizer = load_tokenizer(config["base_model"])
    batch_size = int(config["batch_size"])
    accumulation = int(config["grad_accum_steps"])
    loader = DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        collate_fn=make_preference_collator(tokenizer, int(config["max_sequence_length"])),
    )
    policy = load_policy(config, trainable=True, fresh_lora=True)
    optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    device = next(policy.parameters()).device
    update_count = math.ceil(len(loader) / accumulation)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    seen_examples = 0
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)

    for update_index in range(update_count):
        group_size = min(accumulation, len(loader) - update_index * accumulation)
        group_examples = min(group_size * batch_size, len(rows) - seen_examples)
        update_loss = 0.0
        update_accuracy = 0.0
        for _ in range(group_size):
            chosen_cpu, rejected_cpu = next(iterator)
            for pair_index in range(chosen_cpu["input_ids"].shape[0]):
                chosen = _to_device(_one_example(chosen_cpu, pair_index), device)
                rejected = _to_device(_one_example(rejected_cpu, pair_index), device)
                with torch.no_grad(), reference_mode(policy):
                    ref_chosen = response_sequence_log_probs(policy, chosen)
                    ref_rejected = response_sequence_log_probs(policy, rejected)
                policy_chosen = response_sequence_log_probs(policy, chosen)
                policy_rejected = response_sequence_log_probs(policy, rejected)
                loss, details = dpo_loss(
                    policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta_value
                )
                (loss / group_examples).backward()
                update_loss += float(loss.detach())
                update_accuracy += float(details["preference_accuracy"])
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters(policy), float(config["max_grad_norm"])
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        seen_examples += group_examples
        record = {
            "run_name": run_name,
            "update": update_index + 1,
            "seen_examples": seen_examples,
            "train_loss": update_loss / group_examples,
            "train_preference_accuracy": update_accuracy / group_examples,
            "gradient_norm": float(grad_norm),
            "learning_rate": float(config["learning_rate"]),
            "elapsed_seconds": time.perf_counter() - start,
        }
        append_jsonl(history_file, record)
        print(
            f"{run_name}: update {update_index + 1}/{update_count}, "
            f"loss={record['train_loss']:.4f}, "
            f"accuracy={record['train_preference_accuracy']:.3f}",
            flush=True,
        )

    policy.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapter_hash = adapter_sha256(str(adapter_dir))
    run_config = dict(config)
    run_config.update(
        run_name=run_name,
        training_data=str(repo_path(training_data)),
        adapter_dir=str(adapter_dir),
        beta=beta_value,
        training_examples=len(rows),
    )
    save_json(results_dir / "run_config.json", run_config)
    save_json(
        results_dir / "train_indices.json",
        [{"prompt_id": row["prompt_id"], "source_index": row["source_index"]} for row in rows],
    )
    summary = {
        "run_name": run_name,
        "adapter_dir": str(adapter_dir),
        "training_examples": len(rows),
        "updates": update_count,
        "beta": beta_value,
        "seed": seed,
        "wall_time_seconds": time.perf_counter() - start,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
        ),
        "last_update_loss": record["train_loss"],
        "adapter_sha256": adapter_hash,
    }
    save_json(results_dir / "train_summary.json", summary)
    del optimizer, policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
