# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Evaluate one GRPO adapter on the fixed 200-prompt held-out pool."""

from __future__ import annotations

import argparse
import gc

import torch

from common.data import load_yaml, read_jsonl, write_jsonl
from common.logging_utils import load_json, save_json
from common.models import load_policy, load_tokenizer
from task3_grpo.core.checkpoints import directory_sha256, file_sha256
from task3_grpo.core.paths import output_directory, result_directory
from task3_grpo.core.protocol import validate_shared_protocol
from task3_grpo.evaluation.generation import generate_evaluation_records
from task3_grpo.evaluation.policy_metrics import (
    add_reward_scores,
    reward_scoring_protocol,
    summarize_policy,
)
from task3_grpo.evaluation.qualitative import review_candidates


def _refresh_available_evidence(config: dict) -> None:
    """Update tables/figures only when their prerequisite artifacts exist."""
    from task3_grpo.analysis.aggregate import build_available_evidence
    from task3_grpo.analysis.plots import save_available_figures

    build_available_evidence(config)
    save_available_figures(config)


def _evaluation_protocol(config: dict, adapter, rows: list[dict]) -> dict:
    return {
        "adapter_directory": str(adapter),
        "adapter_sha256": directory_sha256(adapter),
        "base_model": config["base_model"],
        "dtype": config["dtype"],
        "prompt_ids": [row["prompt_id"] for row in rows],
        "prompt_pool_sha256": file_sha256(config["paths"]["rl_prompt_eval"]),
        "seed": int(config["seed"]),
        "max_prompt_length": int(config["max_prompt_length"]),
        "prompt_truncation_side": "left",
        "max_completion_length": int(config["max_completion_length"]),
        "generation": config["generation"],
        "reference_policy": "base model with policy adapter disabled",
    }


def _clear_evaluation_files(paths: tuple) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def run_evaluation(
    config_path: str = "configs/grpo.yaml",
    adapter: str | None = None,
    name: str = "standard",
    overwrite: bool = False,
) -> dict:
    """Generate, reward-score, summarize, or safely reuse one evaluation."""
    config = load_yaml(config_path)
    validate_shared_protocol(config)
    adapter_path = adapter or str(output_directory(config, name) / "policy")
    result_dir = result_directory(config, name)
    result_dir.mkdir(parents=True, exist_ok=True)
    raw_path = result_dir / "generations_raw.jsonl"
    scored_path = result_dir / "generations.jsonl"
    metrics_path = result_dir / "eval_metrics.json"
    prompt_ids_path = result_dir / "eval_prompt_ids.json"
    candidates_path = result_dir / "qualitative_candidates.jsonl"
    protocol_path = result_dir / "evaluation_protocol.json"
    artifact_paths = (
        raw_path, scored_path, metrics_path, prompt_ids_path, candidates_path
    )

    rows = read_jsonl(config["paths"]["rl_prompt_eval"])
    if len(rows) != 200:
        raise ValueError("Task 3 requires exactly 200 held-out prompts.")
    protocol = _evaluation_protocol(config, adapter_path, rows)
    saved_protocol = load_json(protocol_path) if protocol_path.exists() else None
    if overwrite:
        _clear_evaluation_files(artifact_paths)
        if protocol_path.exists():
            protocol_path.unlink()
        saved_protocol = None
    elif saved_protocol is not None and saved_protocol != protocol:
        raise ValueError(
            f"Existing {name} evaluation uses a different adapter or protocol. "
            "Inspect it or pass overwrite=True."
        )
    elif saved_protocol is None and any(path.exists() for path in artifact_paths):
        raise RuntimeError(
            f"Existing {name} evaluation artifacts have no protocol file. "
            "Inspect them or pass overwrite=True."
        )
    if saved_protocol is None:
        save_json(protocol_path, protocol)

    if metrics_path.exists() and scored_path.exists():
        records = read_jsonl(scored_path)
        metrics = load_json(metrics_path)
        if (
            [row.get("prompt_id") for row in records]
            == [row["prompt_id"] for row in rows]
            and metrics.get("adapter_sha256") == protocol["adapter_sha256"]
            and metrics.get("reward_scoring_protocol")
            == reward_scoring_protocol(config)
        ):
            if not prompt_ids_path.exists():
                save_json(prompt_ids_path, [
                    {
                        "prompt_id": row["prompt_id"],
                        "source_index": row.get("source_index"),
                    }
                    for row in rows
                ])
            if not candidates_path.exists():
                write_jsonl(candidates_path, review_candidates(records))
            print(f"Using completed {name} held-out evaluation.")
            _refresh_available_evidence(config)
            return metrics
        raise RuntimeError(
            f"Completed {name} evaluation files are inconsistent; "
            "inspect them or pass overwrite=True."
        )

    existing = read_jsonl(raw_path) if raw_path.exists() else []
    if len(existing) > len(rows):
        raise ValueError("Saved evaluation contains more than 200 prompts.")
    if [row.get("prompt_id") for row in existing] != [
        row["prompt_id"] for row in rows[: len(existing)]
    ]:
        raise RuntimeError("Saved generations do not match the fixed prompt order.")
    if len(existing) < len(rows):
        tokenizer = load_tokenizer(config["base_model"])
        policy = load_policy(config, adapter_path=adapter_path, trainable=False)
        records = generate_evaluation_records(
            policy, tokenizer, rows, config, raw_path, existing
        )
        del policy, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        records = existing

    # The policy is unloaded before the reward model is placed automatically.
    add_reward_scores(records, config)
    write_jsonl(scored_path, records)
    metrics = {
        "run_name": name,
        "adapter_sha256": protocol["adapter_sha256"],
        "evaluation_protocol": protocol,
        "reward_scoring_protocol": reward_scoring_protocol(config),
        **summarize_policy(records),
    }
    save_json(metrics_path, metrics)
    save_json(prompt_ids_path, [
        {"prompt_id": row["prompt_id"], "source_index": row.get("source_index")}
        for row in rows
    ])
    write_jsonl(candidates_path, review_candidates(records))
    _refresh_available_evidence(config)
    print(f"Saved {name} held-out evaluation to {result_dir}.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--adapter")
    parser.add_argument("--name", default="standard")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(run_evaluation(args.config, args.adapter, args.name, args.overwrite))


if __name__ == "__main__":
    main()
