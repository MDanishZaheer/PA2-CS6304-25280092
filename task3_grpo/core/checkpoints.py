# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Atomically save, verify, and resume Task 3 policy checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import torch

from common.data import repo_path
from common.logging_utils import load_json, save_json


def directory_sha256(directory: str | Path) -> str:
    """Hash every file in a checkpoint directory in stable path order."""
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
    """Hash one fixed dataset, cache, or configuration file."""
    source = repo_path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optimizer_to_parameter_devices(optimizer) -> None:
    """Move restored optimizer tensors to their associated parameter device."""
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(parameter.device)


def _save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def save_update_checkpoint(
    output_directory: str | Path,
    update: int,
    policy,
    state: dict,
) -> Path:
    """Publish a complete adapter and optimizer state as one directory."""
    root = repo_path(output_directory) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"update_{int(update):03d}"
    if destination.exists():
        raise FileExistsError(f"Update checkpoint already exists: {destination}")
    temporary = root / f".update_{int(update):03d}_{uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    policy.save_pretrained(temporary / "policy")
    _save_state(temporary / "training_state.pt", state)
    save_json(temporary / "manifest.json", {
        "update": int(update),
        "policy_sha256": directory_sha256(temporary / "policy"),
    })
    temporary.replace(destination)
    return destination


def latest_update_checkpoint(
    output_directory: str | Path,
) -> tuple[Path | None, dict]:
    """Return the highest complete, hash-verified update checkpoint."""
    root = repo_path(output_directory) / "checkpoints"
    if not root.exists():
        return None, {}
    candidates = []
    for path in root.glob("update_[0-9][0-9][0-9]"):
        try:
            candidates.append((int(path.name.removeprefix("update_")), path))
        except ValueError:
            continue
    if not candidates:
        return None, {}
    update, checkpoint = max(candidates)
    manifest_path = checkpoint / "manifest.json"
    state_path = checkpoint / "training_state.pt"
    if not manifest_path.exists() or not state_path.exists():
        raise RuntimeError(f"Incomplete GRPO checkpoint: {checkpoint}")
    manifest = load_json(manifest_path)
    if int(manifest.get("update", -1)) != update:
        raise RuntimeError(f"GRPO checkpoint update mismatch: {checkpoint}")
    if manifest.get("policy_sha256") != directory_sha256(checkpoint / "policy"):
        raise RuntimeError(f"GRPO policy checkpoint hash mismatch: {checkpoint}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state.get("next_update", -1)) != update:
        raise RuntimeError(f"GRPO optimizer-state update mismatch: {checkpoint}")
    return checkpoint, state
