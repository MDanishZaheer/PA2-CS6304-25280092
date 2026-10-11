# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Train the length-control adapter and compare it with standard DPO."""

from __future__ import annotations

import argparse
import csv
import gc
import json

import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import load_json, save_json, set_seed
from common.models import load_policy, load_tokenizer
from task1_dpo.core.trainer import adapter_sha256, train_dpo, validate_saved_run
from task1_dpo.evaluation.generation import generate_comparison_records
from task1_dpo.evaluation.length_metrics import (
    dataset_length_summary,
    preference_by_stratum,
    word_limit_summary,
)
from task1_dpo.evaluation.policy_metrics import add_word_limit_fields
from task1_dpo.evaluation.preference import evaluate_preference_pairs


def _save_csv(path, rows: list[dict]) -> None:
    """Save a short result table with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


LENGTH_RESULT_FILES = (
    "dataset_length_metrics.csv",
    "stratum_metrics.csv",
    "word_limit_metrics.csv",
    "standard_preference_pairs.jsonl",
    "standard_word_generations.jsonl",
    "length_balanced_preference_pairs.jsonl",
    "length_balanced_word_generations.jsonl",
    "qualitative_examples.jsonl",
    "eval_indices.json",
)


def _length_protocol(config, standard_adapter, balanced_adapter, pair_rows, word_rows):
    """Record the inputs that must match before saved results can be reused."""
    return {
        "standard_adapter_sha256": adapter_sha256(str(standard_adapter)),
        "balanced_adapter_sha256": adapter_sha256(str(balanced_adapter)),
        "length_eval_data": str(repo_path(config["paths"]["dpo_length_eval"])),
        "length_eval_prompt_ids": [row["prompt_id"] for row in pair_rows],
        "word_limit_data": str(repo_path(config["paths"]["word_limit_prompts"])),
        "word_limit_prompt_ids": [row["prompt_id"] for row in word_rows],
        "seed": int(config["seed"]),
        "base_model": config["base_model"],
        "beta": float(config["beta"]),
        "max_sequence_length": int(config["max_sequence_length"]),
        "max_generation_tokens": int(config["max_generation_tokens"]),
        "generation": config["generation"],
    }


def _load_completed_length_results(result_dir, protocol):
    """Return results only when the final summary and every output are present."""
    summary_path = result_dir / "study_summary.json"
    if not summary_path.exists():
        return None
    saved = load_json(summary_path)
    if saved.get("protocol") != protocol:
        return None
    if any(not (result_dir / name).exists() for name in LENGTH_RESULT_FILES):
        return None
    return saved["results"]


def run_length_study(config_path: str = "configs/dpo.yaml", overwrite: bool = False) -> dict:
    """Save per-stratum pair results and common-prompt word-limit results."""
    config = load_yaml(config_path)
    standard_adapter = repo_path(config["standard_output"])
    balanced_adapter = repo_path(config["length_output"])
    if not (standard_adapter / "adapter_config.json").exists():
        raise FileNotFoundError("Train the standard DPO adapter before the length study.")
    validate_saved_run(
        config, "standard", str(standard_adapter), config["paths"]["dpo_standard_train"],
        float(config["beta"]), len(read_jsonl(config["paths"]["dpo_standard_train"])),
    )
    if overwrite or not (balanced_adapter / "adapter_config.json").exists():
        train_dpo(
            config_path=config_path,
            run_name="length_balanced",
            dataset_path=config["paths"]["dpo_length_train"],
            output_path=str(balanced_adapter),
            overwrite=overwrite,
        )
    else:
        validate_saved_run(
            config, "length_balanced", str(balanced_adapter),
            config["paths"]["dpo_length_train"], float(config["beta"]),
            len(read_jsonl(config["paths"]["dpo_length_train"])),
        )
    pair_rows = read_jsonl(config["paths"]["dpo_length_eval"])
    word_rows = read_jsonl(config["paths"]["word_limit_prompts"])
    if len({row["prompt_id"] for row in pair_rows}) != len(pair_rows):
        raise ValueError("Length-evaluation prompt IDs must be unique.")
    result_dir = repo_path(config["results_dir"]) / "length"
    result_dir.mkdir(parents=True, exist_ok=True)
    protocol = _length_protocol(
        config, standard_adapter, balanced_adapter, pair_rows, word_rows
    )
    if not overwrite:
        cached = _load_completed_length_results(result_dir, protocol)
        if cached is not None:
            print("Using completed length-study results; skipping evaluation.")
            return cached
    set_seed(int(config["seed"]))
    tokenizer = load_tokenizer(config["base_model"])
    dataset_lengths = [
        dataset_length_summary(read_jsonl(config["paths"]["dpo_standard_train"]), tokenizer, "standard"),
        dataset_length_summary(read_jsonl(config["paths"]["dpo_length_train"]), tokenizer, "length_balanced"),
    ]
    _save_csv(result_dir / "dataset_length_metrics.csv", dataset_lengths)
    stratum_results = []
    word_results = []
    word_records_by_policy = {}
    for name, adapter in (("standard", standard_adapter), ("length_balanced", balanced_adapter)):
        model = load_policy(config, adapter_path=str(adapter), trainable=False)
        _, pair_records = evaluate_preference_pairs(
            model, tokenizer, pair_rows, config, float(config["beta"])
        )
        word_records, _ = generate_comparison_records(model, tokenizer, word_rows, config)
        add_word_limit_fields(word_records)
        word_records_by_policy[name] = word_records
        stratum_results.extend(preference_by_stratum(pair_records, name))
        word_results.append(word_limit_summary(word_records, name))
        write_jsonl(result_dir / f"{name}_preference_pairs.jsonl", pair_records)
        write_jsonl(result_dir / f"{name}_word_generations.jsonl", word_records)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _save_csv(result_dir / "stratum_metrics.csv", stratum_results)
    _save_csv(result_dir / "word_limit_metrics.csv", word_results)
    standard_words = {row["prompt_id"]: row for row in word_records_by_policy["standard"]}
    balanced_words = {row["prompt_id"]: row for row in word_records_by_policy["length_balanced"]}
    qualitative = []
    for prompt_id in standard_words:
        standard = standard_words[prompt_id]
        balanced = balanced_words[prompt_id]
        qualitative.append({
            "prompt_id": prompt_id,
            "prompt": standard["prompt"],
            "standard_response": standard["policy_response"],
            "balanced_response": balanced["policy_response"],
            "standard_length": standard["response_length"],
            "balanced_length": balanced["response_length"],
            "standard_compliance": standard["policy_word_limit_compliance"],
            "balanced_compliance": balanced["policy_word_limit_compliance"],
            "manual_quality_assessment": None,
        })
    write_jsonl(result_dir / "qualitative_examples.jsonl", qualitative)
    save_json(
        result_dir / "eval_indices.json",
        {
            "length_pairs": [
                {"prompt_id": row["prompt_id"], "source_index": row["source_index"]}
                for row in pair_rows
            ],
            "word_limit_prompts": [row["prompt_id"] for row in word_rows],
        },
    )
    results = {
        "dataset_lengths": dataset_lengths,
        "strata": stratum_results,
        "word_limits": word_results,
    }
    # Save this last: its presence marks a complete evaluation, not a partial run.
    save_json(result_dir / "study_summary.json", {"protocol": protocol, "results": results})
    return results


def main() -> None:
    """Run Task 1's fixed length intervention from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run_length_study(arguments.config, arguments.overwrite), indent=2))


if __name__ == "__main__":
    main()
