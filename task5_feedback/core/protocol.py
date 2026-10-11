# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Validate and fingerprint every fixed component of the Task 5 comparison."""

from __future__ import annotations

import hashlib
from pathlib import Path

from common.data import read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task5_feedback.core.paths import POLICY_ORDER, protocols_directory


EXPECTED_VARIANTS = (
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(directory: str | Path) -> str:
    root = repo_path(directory)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No adapter files found in {root}.")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    """Reject changes to assignment-fixed models, budgets, or policies."""
    expected = {
        "seed": 6304,
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "ai_judge_model": "Qwen/Qwen2.5-3B-Instruct",
        "math_max_new_tokens": 512,
        "task5_max_prompt_length": 256,
        "task5_judge_max_input_length": 2048,
        "task5_judge_max_new_tokens": 4,
    }
    for key, required in expected.items():
        if config.get(key) != required:
            raise ValueError(
                f"Task 5 requires {key}={required!r}; found {config.get(key)!r}."
            )
    required_policies = {
        "sft": None,
        "rlvr": "checkpoints/rlvr_policy",
        "rlaif": "checkpoints/rlaif_policy",
    }
    if any(
        config.get("policies", {}).get(name) != path
        for name, path in required_policies.items()
    ):
        raise ValueError("Task 5 must use the supplied SFT, RLVR, and RLAIF policies.")
    if int(config.get("task5_generation_batch_size", 0)) < 1:
        raise ValueError("Task 5 generation batch size must be positive.")


def policy_adapter(config: dict, policy_name: str) -> str | None:
    if policy_name not in POLICY_ORDER:
        raise ValueError(f"Unknown Task 5 policy: {policy_name!r}.")
    return config["policies"][policy_name]


def policy_fingerprint(config: dict, policy_name: str) -> dict:
    adapter = policy_adapter(config, policy_name)
    if adapter is None:
        return {"adapter": None, "adapter_sha256": None}
    root = repo_path(adapter)
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete {policy_name} adapter at {root}: missing {missing}"
        )
    return {
        "adapter": str(adapter),
        "adapter_sha256": directory_sha256(root),
    }


def dataset_path(config: dict, dataset: str) -> str:
    if dataset == "gsm":
        return config["paths"]["gsm_eval"]
    if dataset == "transfer":
        return config["paths"]["math_transfer_eval"]
    raise ValueError(f"Unknown Task 5 dataset: {dataset!r}.")


def row_id(row: dict, dataset: str) -> str:
    if row.get("prompt_id"):
        return str(row["prompt_id"])
    prefix = "gsm" if dataset == "gsm" else "svamp"
    return f"{prefix}:{row['source_index']}"


def load_math_rows(config: dict, dataset: str) -> list[dict]:
    validate_config(config)
    rows = read_jsonl(dataset_path(config, dataset))
    expected_count = 300 if dataset == "gsm" else 100
    if len(rows) != expected_count:
        raise ValueError(
            f"Task 5 {dataset} requires {expected_count} rows; found {len(rows)}."
        )
    required = {"source_index", "question", "messages", "gold_final"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"Task 5 {dataset} rows must contain {sorted(required)}.")
    identifiers = [row_id(row, dataset) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Task 5 {dataset} identifiers must be unique.")
    if any(not row["messages"] for row in rows):
        raise ValueError(f"Task 5 {dataset} contains an empty message list.")
    return rows


def load_diagnostic_rows(config: dict) -> list[dict]:
    validate_config(config)
    rows = read_jsonl(config["paths"]["task5_diagnostics"])
    if len(rows) != 100:
        raise ValueError(f"Task 5 requires 100 diagnostic rows; found {len(rows)}.")
    required = {
        "problem_id", "question", "gold_final", "variant_type", "response",
        "expected_exact_reward", "reasoning_quality", "style",
        "manual_validation",
    }
    if any(not required.issubset(row) for row in rows):
        raise ValueError("Task 5 diagnostic schema is incomplete.")
    if any(not bool(row["manual_validation"]) for row in rows):
        raise ValueError("Every Task 5 diagnostic row must be manually validated.")
    problem_order = list(dict.fromkeys(str(row["problem_id"]) for row in rows))
    if len(problem_order) != 20:
        raise ValueError(f"Expected 20 diagnostic problems; found {len(problem_order)}.")
    for problem_id in problem_order:
        variants = [
            row["variant_type"] for row in rows
            if str(row["problem_id"]) == problem_id
        ]
        if variants != list(EXPECTED_VARIANTS):
            raise ValueError(
                f"Diagnostic variants changed for problem {problem_id}: {variants}"
            )
    return rows


def generation_protocol(config: dict, dataset: str, policy_name: str) -> dict:
    rows = load_math_rows(config, dataset)
    path = dataset_path(config, dataset)
    return {
        "dataset": dataset,
        "dataset_path": path,
        "dataset_sha256": file_sha256(path),
        "prompt_ids": [row_id(row, dataset) for row in rows],
        "policy": policy_name,
        "base_model": config["base_model"],
        **policy_fingerprint(config, policy_name),
        "max_prompt_length": int(config["task5_max_prompt_length"]),
        "max_new_tokens": int(config["math_max_new_tokens"]),
        "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0},
    }


def protect_protocol(path: str | Path, expected: dict, overwrite: bool = False) -> None:
    destination = repo_path(path)
    if destination.exists() and not overwrite:
        if load_json(destination) != expected:
            raise RuntimeError(f"Saved Task 5 protocol does not match: {destination}")
        return
    save_json(destination, expected)


def save_task5_manifest(config: dict) -> dict:
    validate_config(config)
    gsm = load_math_rows(config, "gsm")
    transfer = load_math_rows(config, "transfer")
    diagnostics = load_diagnostic_rows(config)
    manifest = {
        "seed": int(config["seed"]),
        "base_model": config["base_model"],
        "ai_judge_model": config["ai_judge_model"],
        "datasets": {
            "gsm": {
                "rows": len(gsm),
                "sha256": file_sha256(config["paths"]["gsm_eval"]),
            },
            "transfer": {
                "rows": len(transfer),
                "sha256": file_sha256(config["paths"]["math_transfer_eval"]),
            },
            "diagnostics": {
                "rows": len(diagnostics),
                "problems": 20,
                "sha256": file_sha256(config["paths"]["task5_diagnostics"]),
            },
        },
        "policies": {
            name: policy_fingerprint(config, name) for name in POLICY_ORDER
        },
    }
    save_json(protocols_directory(config) / "task5_manifest.json", manifest)
    return manifest
