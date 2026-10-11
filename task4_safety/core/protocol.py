# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Protect the fixed models, data, labels, and decoding used by Task 4."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd

from common.data import repo_path
from common.logging_utils import load_json, save_json
from task4_safety.core.paths import POLICY_ORDER, protocol_directory


BEHAVIOR_LABELS = (
    "SAFE_ANSWER",
    "JUSTIFIED_REFUSAL",
    "UNSAFE_COMPLIANCE",
    "OVER_REFUSAL",
    "AMBIGUOUS",
)


def file_sha256(path: str | Path) -> str:
    """Hash one file in streaming chunks."""
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(directory: str | Path) -> str:
    """Hash a saved adapter directory deterministically."""
    root = repo_path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"Adapter directory is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def policy_adapter(config: dict, policy_name: str) -> str | None:
    """Return the assignment-fixed adapter, or None for the SFT baseline."""
    if policy_name not in POLICY_ORDER:
        raise ValueError(f"Unknown Task 4 policy: {policy_name!r}.")
    return config["policies"].get(policy_name)


def validate_config(config: dict) -> None:
    """Reject changes to the released Task 4 evaluation protocol."""
    expected = {
        "seed": 6304,
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "ai_judge_model": "Qwen/Qwen2.5-3B-Instruct",
        "safety_max_new_tokens": 256,
        "task4_max_prompt_length": 256,
        "judge_max_new_tokens": 64,
        "judge_temperature": 0.0,
        "manual_audit_per_class": 30,
    }
    for key, required in expected.items():
        actual = config.get(key)
        if isinstance(required, float):
            valid = actual is not None and math.isclose(float(actual), required)
        else:
            valid = actual == required
        if not valid:
            raise ValueError(
                f"Task 4 requires {key}={required!r}; found {actual!r}."
            )
    required_paths = {
        "sft": None,
        "dpo": "task1_dpo/outputs/standard",
        "ppo": "task2_ppo/outputs/standard/policy",
        "grpo": "task3_grpo/outputs/standard/policy",
    }
    configured_paths = config.get("policies", {})
    if any(configured_paths.get(name) != path for name, path in required_paths.items()):
        raise ValueError("Task 4 must use the four fixed standard policies.")
    if int(config.get("task4_generation_batch_size", 0)) < 1:
        raise ValueError("Task 4 generation batch size must be positive.")
    if int(config.get("judge_batch_size", 0)) < 1:
        raise ValueError("Task 4 judge batch size must be positive.")
    if int(config.get("judge_max_input_length", 0)) < 768:
        raise ValueError("Task 4 judge input length is too short for XSTest responses.")


def load_xstest(config: dict) -> pd.DataFrame:
    """Load and validate the unmodified released XSTest table."""
    validate_config(config)
    path = repo_path(config["paths"]["xstest"])
    table = pd.read_csv(path)
    required = {
        "id", "prompt", "type", "label", "focus", "note",
        "benchmark_class", "xstest_id",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"XSTest is missing columns: {sorted(missing)}")
    if len(table) != 450:
        raise ValueError(f"Task 4 requires all 450 XSTest prompts; found {len(table)}.")
    if table["xstest_id"].duplicated().any():
        raise ValueError("XSTest prompt identifiers must be unique.")
    if table["xstest_id"].astype(int).tolist() != list(range(450)):
        raise ValueError("XSTest prompt order or identifiers changed.")
    counts = table["benchmark_class"].value_counts().to_dict()
    if counts != {"SAFE": 250, "UNSAFE": 200}:
        raise ValueError(f"Unexpected XSTest class counts: {counts}")
    if table["prompt"].isna().any() or (table["prompt"].astype(str).str.len() == 0).any():
        raise ValueError("XSTest contains an empty prompt.")
    return table


def policy_fingerprint(config: dict, policy_name: str) -> dict:
    """Record the base policy or the exact adapter used by one condition."""
    adapter = policy_adapter(config, policy_name)
    if adapter is None:
        return {"adapter": None, "adapter_sha256": None}
    resolved = repo_path(adapter)
    required = {"adapter_config.json", "adapter_model.safetensors"}
    missing = [name for name in required if not (resolved / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete {policy_name} adapter at {resolved}: missing {missing}"
        )
    return {
        "adapter": str(adapter),
        "adapter_sha256": directory_sha256(resolved),
    }


def generation_protocol(config: dict, policy_name: str) -> dict:
    """Describe everything that can change generated safety responses."""
    xstest_path = repo_path(config["paths"]["xstest"])
    return {
        "policy": policy_name,
        "base_model": config["base_model"],
        **policy_fingerprint(config, policy_name),
        "xstest": str(config["paths"]["xstest"]),
        "xstest_sha256": file_sha256(xstest_path),
        "prompt_count": 450,
        "prompt_order": "released xstest_id order 0..449",
        "max_prompt_length": int(config["task4_max_prompt_length"]),
        "max_new_tokens": int(config["safety_max_new_tokens"]),
        "decoding": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }


def protect_protocol(path: str | Path, expected: dict, overwrite: bool = False) -> None:
    """Save a new protocol or reject reuse under different settings."""
    resolved = repo_path(path)
    if resolved.exists() and not overwrite:
        actual = load_json(resolved)
        if actual != expected:
            raise RuntimeError(
                f"Saved Task 4 protocol does not match this run: {resolved}"
            )
        return
    save_json(resolved, expected)


def generation_protocol_path(config: dict, policy_name: str):
    return protocol_directory(config) / f"generation_{policy_name}.json"


def judge_prompt_sha256(prompt_template: str) -> str:
    return hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()


def judge_protocol(config: dict, prompt_template: str) -> dict:
    """Describe the fixed categorical judge and its deterministic decoding."""
    return {
        "judge_model": config["ai_judge_model"],
        "prompt_template_sha256": judge_prompt_sha256(prompt_template),
        "labels": list(BEHAVIOR_LABELS),
        "max_input_length": int(config["judge_max_input_length"]),
        "max_new_tokens": int(config["judge_max_new_tokens"]),
        "decoding": {"do_sample": False, "temperature": 0.0},
        "raw_output_saved": True,
    }


def save_task4_manifest(config: dict) -> dict:
    """Save the fixed data and policy hashes used by the complete evaluation."""
    validate_config(config)
    table = load_xstest(config)
    manifest = {
        "seed": int(config["seed"]),
        "xstest_rows": len(table),
        "safe_prompts": int((table["benchmark_class"] == "SAFE").sum()),
        "unsafe_prompts": int((table["benchmark_class"] == "UNSAFE").sum()),
        "xstest_sha256": file_sha256(config["paths"]["xstest"]),
        "policies": {
            name: policy_fingerprint(config, name) for name in POLICY_ORDER
        },
    }
    save_json(protocol_directory(config) / "task4_manifest.json", manifest)
    return manifest
