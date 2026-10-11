# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Save, verify, and resume Task 2 policy and critic checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

from common.data import repo_path
from common.logging_utils import load_json, save_json
from common.models import (
    enable_training_memory_savings,
    promote_value_head,
    resolve_dtype,
)


def directory_sha256(directory: str | Path) -> str:
    """Hash every file that defines a checkpoint or adapter directory."""
    root = repo_path(directory)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in {root}.")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one fixed cache or metadata file."""
    source = repo_path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_saved_value_model(config: dict, adapter_directory: str | Path):
    """Reload the supplied midpoint critic with a saved trainable adapter."""
    dtype = resolve_dtype(config.get("dtype", "float16"))
    base = AutoModelForSequenceClassification.from_pretrained(
        str(repo_path(config["paths"]["ppo_midpoint_value"])),
        num_labels=1,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base,
        str(repo_path(adapter_directory)),
        is_trainable=True,
    )
    promote_value_head(model)
    model.config.use_cache = False
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    enable_training_memory_savings(model)
    return model


def optimizer_to_parameter_devices(optimizer) -> None:
    """Move restored optimizer tensors to their parameter devices."""
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(parameter.device)


def save_training_state(path: str | Path, state: dict) -> None:
    """Atomically save resumable optimizer and progress state."""
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)


def load_training_state(path: str | Path) -> dict:
    """Load a trusted local Task 2 training-state file."""
    return torch.load(repo_path(path), map_location="cpu", weights_only=False)


def save_update_checkpoint(
    output_directory: str | Path,
    update: int,
    policy,
    value_model,
    state: dict,
) -> Path:
    """Atomically publish one complete, resumable PPO update checkpoint."""
    root = repo_path(output_directory) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"update_{int(update):03d}"
    if destination.exists():
        raise FileExistsError(f"Update checkpoint already exists: {destination}")
    temporary = root / f".update_{int(update):03d}_{uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    policy.save_pretrained(temporary / "policy")
    value_model.save_pretrained(temporary / "value")
    save_training_state(temporary / "training_state.pt", state)
    save_json(temporary / "manifest.json", {
        "update": int(update),
        "policy_sha256": directory_sha256(temporary / "policy"),
        "value_sha256": directory_sha256(temporary / "value"),
    })
    temporary.replace(destination)
    return destination


def latest_update_checkpoint(
    output_directory: str | Path,
) -> tuple[Path | None, dict]:
    """Return the highest complete update checkpoint and its optimizer state."""
    root = repo_path(output_directory) / "checkpoints"
    if not root.exists():
        return None, {}
    candidates = []
    for path in root.glob("update_[0-9][0-9][0-9]"):
        try:
            update = int(path.name.removeprefix("update_"))
        except ValueError:
            continue
        candidates.append((update, path))
    if not candidates:
        return None, {}
    update, checkpoint = max(candidates)
    manifest_path = checkpoint / "manifest.json"
    state_path = checkpoint / "training_state.pt"
    if not manifest_path.exists() or not state_path.exists():
        raise RuntimeError(f"Incomplete PPO checkpoint: {checkpoint}")
    manifest = load_json(manifest_path)
    if int(manifest.get("update", -1)) != update:
        raise RuntimeError(f"PPO checkpoint update mismatch: {checkpoint}")
    if manifest.get("policy_sha256") != directory_sha256(checkpoint / "policy"):
        raise RuntimeError(f"PPO policy checkpoint hash mismatch: {checkpoint}")
    if manifest.get("value_sha256") != directory_sha256(checkpoint / "value"):
        raise RuntimeError(f"PPO critic checkpoint hash mismatch: {checkpoint}")
    state = load_training_state(state_path)
    if int(state.get("next_update", -1)) != update:
        raise RuntimeError(f"PPO optimizer-state update mismatch: {checkpoint}")
    return checkpoint, state
