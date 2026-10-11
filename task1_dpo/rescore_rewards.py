# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Correct reward scores using saved Task 1 generations only."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import append_jsonl, load_json, save_json
from common.models import load_reward_model
from task1_dpo.analysis.plots import plot_beta_comparison
from task1_dpo.evaluation.policy_metrics import (
    reward_scoring_protocol,
    score_one_response,
)
from task1_dpo.evaluation.qualitative import review_candidates


RUN_FOLDERS = {
    "standard": "standard",
    "beta_0.03": "beta/beta_0.03",
    "beta_0.10": "beta/beta_0.10",
    "beta_0.30": "beta/beta_0.30",
}


def _score_key(protocol: dict, messages: list[dict], response: str) -> str:
    """Hash the complete input so identical responses can safely reuse a score."""
    payload = {
        "protocol": protocol,
        "messages": messages,
        "response": response,
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_score_cache(cache_path, protocol_hash: str) -> dict[str, float]:
    """Load completed scores from an earlier or interrupted rescoring run."""
    if not cache_path.exists():
        return {}
    scores = {}
    for row in read_jsonl(cache_path):
        if row.get("protocol_sha256") == protocol_hash:
            scores[row["response_sha256"]] = float(row["reward_score"])
    return scores


def _collect_records(results_root, protocol: dict):
    """Load every saved response and collect unique reward-model inputs."""
    records_by_run = {}
    unique_inputs = {}
    expected_prompt_ids = None
    for run_name, relative_folder in RUN_FOLDERS.items():
        folder = results_root / relative_folder
        raw_path = folder / "generations_raw.jsonl"
        metrics_path = folder / "eval_metrics.json"
        if not raw_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(
                f"Complete {run_name} evaluation before rescoring: {raw_path}"
            )
        records = read_jsonl(raw_path)
        prompt_ids = [row["prompt_id"] for row in records]
        if expected_prompt_ids is None:
            expected_prompt_ids = prompt_ids
        elif prompt_ids != expected_prompt_ids:
            raise ValueError("All Task 1 conditions must use identical held-out prompt IDs.")
        records_by_run[run_name] = records
        for record in records:
            for response_role in ("policy", "reference"):
                response = record[f"{response_role}_response"]
                key = _score_key(protocol, record["messages"], response)
                unique_inputs.setdefault(key, {
                    "messages": record["messages"],
                    "response": response,
                    "prompt_id": record["prompt_id"],
                    "response_role": response_role,
                    "first_seen_run": run_name,
                })
    return records_by_run, unique_inputs


def _results_are_current(results_root, protocol: dict) -> bool:
    """Return true only when every run already uses independent scoring."""
    for relative_folder in RUN_FOLDERS.values():
        folder = results_root / relative_folder
        metrics_path = folder / "eval_metrics.json"
        generations_path = folder / "generations.jsonl"
        if not metrics_path.exists() or not generations_path.exists():
            return False
        if load_json(metrics_path).get("reward_scoring") != protocol:
            return False
        records = read_jsonl(generations_path)
        if not records or any(
            "policy_reward_score" not in row or "reference_reward_score" not in row
            for row in records
        ):
            return False
    return True


def _save_beta_summaries(results_root, metrics_by_run: dict) -> None:
    """Refresh the beta JSON files and combined CSV after rescoring."""
    beta_metrics = []
    for run_name in ("beta_0.03", "beta_0.10", "beta_0.30"):
        metrics = metrics_by_run[run_name]
        save_json(results_root / "beta" / f"{run_name}.json", metrics)
        beta_metrics.append(metrics)
    summary_path = results_root / "beta" / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(beta_metrics[0]))
        writer.writeheader()
        writer.writerows(beta_metrics)


def _apply_scores(results_root, records_by_run, scores, protocol: dict) -> dict:
    """Write corrected records, summaries, candidates, and the beta figure."""
    metrics_by_run = {}
    for run_name, relative_folder in RUN_FOLDERS.items():
        folder = results_root / relative_folder
        records = records_by_run[run_name]
        for record in records:
            policy_key = _score_key(
                protocol, record["messages"], record["policy_response"]
            )
            reference_key = _score_key(
                protocol, record["messages"], record["reference_response"]
            )
            record["policy_reward_score"] = scores[policy_key]
            record["reference_reward_score"] = scores[reference_key]
        write_jsonl(folder / "generations.jsonl", records)
        write_jsonl(
            folder / "qualitative_candidates.jsonl", review_candidates(records)
        )
        policy_scores = np.asarray(
            [row["policy_reward_score"] for row in records], dtype=float
        )
        reference_scores = np.asarray(
            [row["reference_reward_score"] for row in records], dtype=float
        )
        metrics = load_json(folder / "eval_metrics.json")
        metrics.update({
            "reward_scoring": protocol,
            "reward_model_score_mean": float(policy_scores.mean()),
            "reward_model_score_std": float(policy_scores.std(ddof=0)),
            "reference_reward_score_mean": float(reference_scores.mean()),
            "reference_reward_score_std": float(reference_scores.std(ddof=0)),
        })
        save_json(folder / "eval_metrics.json", metrics)
        metrics_by_run[run_name] = metrics
    _save_beta_summaries(results_root, metrics_by_run)
    figure_dir = results_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure = plot_beta_comparison(str(results_root))
    figure.savefig(figure_dir / "beta_comparison.png", dpi=180, bbox_inches="tight")
    return metrics_by_run


def rescore_saved_rewards(config_path: str = "configs/dpo.yaml") -> dict:
    """Rescore saved generations independently without training or generation."""
    config = load_yaml(config_path)
    results_root = repo_path(config["results_dir"])
    protocol = reward_scoring_protocol(config)
    if _results_are_current(results_root, protocol):
        print("Independent reward scores already exist; skipping rescoring.")
        return {
            run_name: load_json(results_root / folder / "eval_metrics.json")
            for run_name, folder in RUN_FOLDERS.items()
        }

    records_by_run, unique_inputs = _collect_records(results_root, protocol)
    protocol_text = json.dumps(protocol, sort_keys=True, ensure_ascii=False)
    protocol_hash = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
    cache_path = results_root / "reward_score_cache.jsonl"
    scores = _load_score_cache(cache_path, protocol_hash)
    pending = [(key, item) for key, item in unique_inputs.items() if key not in scores]
    print(
        f"Reward inputs: {len(unique_inputs)} unique, "
        f"{len(scores)} cached, {len(pending)} remaining."
    )

    if pending:
        reward_model, reward_tokenizer = load_reward_model(config)
        for number, (key, item) in enumerate(pending, start=1):
            score = score_one_response(
                reward_model,
                reward_tokenizer,
                item["messages"],
                item["response"],
            )
            if not math.isfinite(score):
                raise ValueError(f"Non-finite reward score for {item['prompt_id']}.")
            scores[key] = score
            append_jsonl(cache_path, {
                "protocol_sha256": protocol_hash,
                "response_sha256": key,
                "reward_score": score,
                "prompt_id": item["prompt_id"],
                "response_role": item["response_role"],
                "first_seen_run": item["first_seen_run"],
            })
            if number % 25 == 0 or number == len(pending):
                print(f"Scored {number}/{len(pending)} remaining inputs.", flush=True)
        del reward_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = _apply_scores(
        results_root, records_by_run, scores, protocol
    )
    save_json(results_root / "reward_scoring_protocol.json", {
        **protocol,
        "protocol_sha256": protocol_hash,
        "unique_scored_inputs": len(unique_inputs),
        "cache_file": str(cache_path),
    })
    print("Corrected reward summaries and beta figure were saved.")
    return metrics


def main() -> None:
    """Run the independent reward-only correction from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dpo.yaml")
    arguments = parser.parse_args()
    metrics = rescore_saved_rewards(arguments.config)
    print(json.dumps({
        name: {
            "reward_model_score_mean": row["reward_model_score_mean"],
            "reference_reward_score_mean": row["reference_reward_score_mean"],
        }
        for name, row in metrics.items()
    }, indent=2))


if __name__ == "__main__":
    main()
