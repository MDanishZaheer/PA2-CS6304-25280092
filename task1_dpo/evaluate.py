# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Evaluate one DPO adapter on fixed held-out pairs and generated prompts."""

from __future__ import annotations

import argparse
import gc
import json

import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import load_json, save_json, set_seed
from common.models import load_policy, load_reward_model, load_tokenizer
from task1_dpo.core.paths import result_directory
from task1_dpo.core.trainer import adapter_sha256
from task1_dpo.evaluation.generation import generate_comparison_records
from task1_dpo.evaluation.policy_metrics import (
    add_reward_scores,
    reward_scoring_protocol,
    summarize_generation,
)
from task1_dpo.evaluation.preference import evaluate_preference_pairs
from task1_dpo.evaluation.qualitative import review_candidates


def run_evaluation(
    config_path: str = "configs/dpo.yaml",
    adapter: str | None = None,
    name: str = "standard",
    dataset_path: str | None = None,
    beta: float | None = None,
    max_eval_examples: int | None = None,
) -> dict:
    """Save held-out DPO diagnostics and reward-scored generated responses."""
    config = load_yaml(config_path)
    adapter_path = repo_path(adapter or config["standard_output"])
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"DPO adapter is missing: {adapter_path}")
    evaluation_data = dataset_path or config["paths"]["dpo_standard_eval"]
    rows = read_jsonl(evaluation_data)
    if max_eval_examples is not None:
        if max_eval_examples <= 0:
            raise ValueError("max_eval_examples must be positive.")
        rows = rows[:max_eval_examples]
    if not rows:
        raise ValueError("The held-out DPO dataset is empty.")
    output_dir = result_directory(config, name)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_hash = adapter_sha256(str(adapter_path))
    generation_protocol = {
        "adapter_sha256": adapter_hash,
        "eval_data": str(repo_path(evaluation_data)),
        "prompt_ids": [row["prompt_id"] for row in rows],
        "seed": int(config["seed"]),
        "max_sequence_length": int(config["max_sequence_length"]),
        "max_generation_tokens": int(config["max_generation_tokens"]),
        "generation": config["generation"],
    }
    raw_file = output_dir / "generations_raw.jsonl"
    protocol_file = output_dir / "generation_protocol.json"
    set_seed(int(config["seed"]))
    tokenizer = load_tokenizer(config["base_model"])
    policy = load_policy(config, adapter_path=str(adapter_path), trainable=False)
    beta_value = float(config["beta"] if beta is None else beta)
    preference_metrics, pair_records = evaluate_preference_pairs(
        policy, tokenizer, rows, config, beta_value
    )
    write_jsonl(output_dir / "preference_pairs.jsonl", pair_records)
    if raw_file.exists() and protocol_file.exists() and load_json(protocol_file) == generation_protocol:
        generation_records = read_jsonl(raw_file)
        sampled_kl = sum(row["sampled_kl_sequence_sum"] for row in generation_records) / max(
            sum(row["response_length"] for row in generation_records), 1
        )
    else:
        generation_records, sampled_kl = generate_comparison_records(
            policy, tokenizer, rows, config
        )
        write_jsonl(raw_file, generation_records)
        save_json(protocol_file, generation_protocol)
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reward_model, reward_tokenizer = load_reward_model(config)
    add_reward_scores(generation_records, reward_model, reward_tokenizer)
    del reward_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_jsonl(output_dir / "generations.jsonl", generation_records)
    write_jsonl(output_dir / "qualitative_candidates.jsonl", review_candidates(generation_records))
    metrics = {
        "run_name": name,
        "adapter": str(adapter_path),
        "adapter_sha256": adapter_hash,
        "eval_data": str(repo_path(evaluation_data)),
        "beta": beta_value,
        "seed": int(config["seed"]),
        "generation": config["generation"],
        "max_generation_tokens": int(config["max_generation_tokens"]),
        "reward_scoring": reward_scoring_protocol(config),
        **preference_metrics,
        **summarize_generation(generation_records, sampled_kl),
    }
    save_json(output_dir / "eval_metrics.json", metrics)
    save_json(
        output_dir / "eval_indices.json",
        [{"prompt_id": row["prompt_id"], "source_index": row.get("source_index")} for row in rows],
    )
    return metrics


def main() -> None:
    """Parse the fixed evaluation settings for one adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--adapter")
    parser.add_argument("--name", default="standard")
    parser.add_argument("--dataset")
    parser.add_argument("--beta", type=float)
    parser.add_argument("--max-eval-examples", type=int)
    arguments = parser.parse_args()
    metrics = run_evaluation(
        config_path=arguments.config,
        adapter=arguments.adapter,
        name=arguments.name,
        dataset_path=arguments.dataset,
        beta=arguments.beta,
        max_eval_examples=arguments.max_eval_examples,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
