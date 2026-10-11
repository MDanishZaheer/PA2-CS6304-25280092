# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Evaluate a PPO policy on the fixed 200-prompt held-out set."""

from __future__ import annotations

import argparse
import gc

import torch

from common.data import load_yaml, read_jsonl, write_jsonl
from common.logging_utils import load_json, save_json
from common.models import load_policy, load_tokenizer
from task2_ppo.core.checkpoints import directory_sha256, file_sha256
from task2_ppo.core.paths import output_directory, result_directory
from task2_ppo.core.protocol import validate_shared_protocol
from task2_ppo.evaluation.generation import generate_evaluation_records
from task2_ppo.evaluation.policy_metrics import (
    add_reward_scores,
    reward_scoring_protocol,
    summarize_policy,
)
from task2_ppo.evaluation.qualitative import review_candidates


def _save_standard_evidence(config: dict, name: str) -> None:
    """Create the standard report table and figure when both stages exist."""
    if name != "standard":
        return
    from task2_ppo.analysis.aggregate import standard_summary_table
    from task2_ppo.analysis.plots import save_task2_figures

    table = standard_summary_table(config["results_dir"])
    table.to_csv(
        result_directory(config, "standard") / "summary.csv", index=False
    )
    save_task2_figures(config["results_dir"])


def _evaluation_protocol(config: dict, adapter, rows: list[dict]) -> dict:
    """Record the checkpoint, prompts, and decoding settings used for evaluation."""
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
        "max_response_length": int(config["eval_max_response_length"]),
        "generation": config["generation"],
        "reference_policy": "base model with the policy adapter disabled",
    }


def run_evaluation(
    config_path: str = "configs/ppo.yaml",
    adapter: str | None = None,
    name: str = "standard",
    overwrite: bool = False,
) -> dict:
    """Generate, independently reward-score, and summarize one saved policy."""
    config = load_yaml(config_path)
    validate_shared_protocol(config)
    adapter_path = adapter or str(output_directory(config, name) / "policy")
    result_dir = result_directory(config, name)
    result_dir.mkdir(parents=True, exist_ok=True)
    raw_path = result_dir / "generations_raw.jsonl"
    scored_path = result_dir / "generations.jsonl"
    metrics_path = result_dir / "eval_metrics.json"
    protocol_path = result_dir / "evaluation_protocol.json"

    rows = read_jsonl(config["paths"]["rl_prompt_eval"])
    if len(rows) != 200:
        raise ValueError("Task 2 requires exactly 200 held-out prompts.")
    protocol = _evaluation_protocol(config, adapter_path, rows)
    saved_protocol = load_json(protocol_path) if protocol_path.exists() else None
    generated_paths = (
        raw_path,
        scored_path,
        metrics_path,
        result_dir / "eval_prompt_ids.json",
        result_dir / "qualitative_candidates.jsonl",
    )
    unverified_artifacts = saved_protocol is None and any(
        path.exists() for path in generated_paths
    )
    if (
        overwrite
        or unverified_artifacts
        or (saved_protocol is not None and saved_protocol != protocol)
    ):
        write_jsonl(raw_path, [])
        for path in generated_paths[1:]:
            if path.exists():
                path.unlink()
        saved_protocol = None
    if saved_protocol is None:
        save_json(protocol_path, protocol)

    if metrics_path.exists() and scored_path.exists():
        records = read_jsonl(scored_path)
        metrics = load_json(metrics_path)
        record_ids = [record.get("prompt_id") for record in records]
        expected_ids = [row["prompt_id"] for row in rows]
        if (
            len(records) == len(rows)
            and record_ids == expected_ids
            and metrics.get("adapter_sha256") == protocol["adapter_sha256"]
            and metrics.get("reward_scoring_protocol")
            == reward_scoring_protocol(config)
        ):
            print(f"Using completed {name} held-out evaluation.")
            _save_standard_evidence(config, name)
            return metrics
        raise RuntimeError(
            f"Completed {name} evaluation files are inconsistent. "
            "Inspect them or rerun with overwrite=True."
        )

    existing = read_jsonl(raw_path) if raw_path.exists() else []
    if len(existing) > len(rows):
        raise ValueError("Saved evaluation contains more than 200 prompts.")
    existing_ids = [record.get("prompt_id") for record in existing]
    expected_ids = [row["prompt_id"] for row in rows[: len(existing)]]
    if existing_ids != expected_ids:
        raise RuntimeError("Saved held-out generations do not match the fixed prompt order.")
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
    save_json(
        result_dir / "eval_prompt_ids.json",
        [
            {"prompt_id": row["prompt_id"], "source_index": row.get("source_index")}
            for row in rows
        ],
    )
    write_jsonl(result_dir / "qualitative_candidates.jsonl", review_candidates(records))
    _save_standard_evidence(config, name)
    print(f"Saved {name} held-out evaluation to {result_dir}.")
    return metrics


def main() -> None:
    """Parse command-line arguments and run the fixed evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--adapter")
    parser.add_argument("--name", default="standard")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(run_evaluation(args.config, args.adapter, args.name, args.overwrite))


if __name__ == "__main__":
    main()
