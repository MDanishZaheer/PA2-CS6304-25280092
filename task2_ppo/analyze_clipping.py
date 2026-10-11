# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Analyze cached PPO clipping geometry and run matched epsilon forks."""

from __future__ import annotations

import argparse
import gc

import pandas as pd
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import response_token_logprobs
from common.logging_utils import append_jsonl, load_json, save_json
from common.models import load_policy, load_tokenizer
from task2_ppo.analysis.plots import save_task2_figures
from task2_ppo.core.batching import encode_cached_response, pad_token_vectors
from task2_ppo.core.checkpoints import directory_sha256, file_sha256
from task2_ppo.core.paths import result_directory
from task2_ppo.core.protocol import validate_shared_protocol
from task2_ppo.core.trainer import train_ppo_continuation
from task2_ppo.evaluate import run_evaluation
from task2_ppo.evaluation.qualitative import compare_condition_responses
from task2_ppo.evaluation.stability import stability_summary
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
)


def load_cached_rollouts(path) -> list[dict]:
    """Load and validate the fixed course PPO rollout cache."""
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache.")

    normalized = []
    required = {
        "source_index",
        "prompt_id",
        "response",
        "response_tokens",
        "terminated_with_eos",
        "raw_terminal_reward",
        "effective_terminal_reward",
        "old_logprobs",
        "ref_logprobs",
        "values",
    }
    for index, cached_row in enumerate(rows):
        row = dict(cached_row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Cached row {index} is missing {sorted(missing)}.")
        token_count = int(row["response_tokens"])
        for field in ("old_logprobs", "ref_logprobs", "values"):
            tensor = torch.as_tensor(row[field], dtype=torch.float32).flatten()
            if tensor.numel() != token_count:
                raise ValueError(
                    f"Cached row {index} has {tensor.numel()} {field} values, "
                    f"but response_tokens is {token_count}."
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Cached row {index} contains non-finite {field} values.")
            row[field] = tensor
        normalized.append(row)
    return normalized


def compute_cached_clipping_metrics(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    epsilon_values: list[float],
) -> list[dict]:
    """Measure clipping on one fixed batch for every requested epsilon."""
    records = []
    valid_tokens = int(response_mask.sum().item())
    for epsilon in epsilon_values:
        loss, ratio, clip_fraction, affected_fraction = ppo_policy_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            response_mask,
            eps=float(epsilon),
        )
        valid_ratios = ratio[response_mask.bool()]
        values = (loss, clip_fraction, affected_fraction, valid_ratios)
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("Cached clipping analysis produced a non-finite value.")
        records.append({
            "clip_epsilon": float(epsilon),
            "cached_responses": int(response_mask.shape[0]),
            "cached_response_tokens": valid_tokens,
            "clipped_surrogate": float(-loss),
            "clip_fraction": float(clip_fraction),
            "affected_token_fraction": float(affected_fraction),
            "ratio_mean": float(valid_ratios.mean()),
            "ratio_std": float(valid_ratios.std(unbiased=False)),
            "ratio_min": float(valid_ratios.min()),
            "ratio_max": float(valid_ratios.max()),
        })
    return records


def _cached_protocol(config: dict, rows: list[dict]) -> dict:
    """Describe the immutable inputs to cached-rollout reconstruction."""
    return {
        "cache_sha256": file_sha256(config["cached_rollouts"]),
        "midpoint_policy_sha256": directory_sha256(
            config["paths"]["ppo_midpoint_policy"]
        ),
        "base_model": config["base_model"],
        "dtype": config["dtype"],
        "prompt_ids": [row["prompt_id"] for row in rows],
        "prompt_pool_sha256": file_sha256(config["paths"]["rl_prompt_eval"]),
        "max_prompt_length": int(config["max_prompt_length"]),
        "prompt_truncation_side": "left",
        "gamma": float(config["gamma"]),
        "gae_lambda": float(config["gae_lambda"]),
        "kl_beta": float(config["kl_beta"]),
        "clip_values": [float(value) for value in config["clip_values"]],
        "advantage_normalization": "global over all valid cached response tokens",
    }


def _prompt_lookup(config: dict) -> dict[str, dict]:
    """Index the released cached-batch prompts by their stable prompt ID."""
    prompts = read_jsonl(config["paths"]["rl_prompt_eval"])
    lookup = {row["prompt_id"]: row for row in prompts}
    if len(lookup) != len(prompts):
        raise ValueError("The PPO cached-batch prompt IDs are not unique.")
    return lookup


def _reconstruct_midpoint_logprobs(
    config: dict,
    cached_rows: list[dict],
    result_dir,
    overwrite: bool,
) -> list[dict]:
    """Run cached responses through the supplied midpoint, with resume support."""
    logprob_path = result_dir / "cached_midpoint_logprobs.jsonl"
    protocol_path = result_dir / "cached_rollout_protocol.json"
    protocol = _cached_protocol(config, cached_rows)
    saved_protocol = load_json(protocol_path) if protocol_path.exists() else None
    unverified_logprobs = saved_protocol is None and logprob_path.exists()
    if (
        overwrite
        or unverified_logprobs
        or (saved_protocol is not None and saved_protocol != protocol)
    ):
        write_jsonl(logprob_path, [])
        saved_protocol = None
    if saved_protocol is None:
        save_json(protocol_path, protocol)

    records = read_jsonl(logprob_path) if logprob_path.exists() else []
    expected_prefix = [row["prompt_id"] for row in cached_rows[: len(records)]]
    if [row["prompt_id"] for row in records] != expected_prefix:
        raise ValueError("Partial cached log-probabilities have a different prompt order.")
    if len(records) == len(cached_rows):
        return records

    prompts = _prompt_lookup(config)
    tokenizer = load_tokenizer(config["base_model"])
    policy = load_policy(
        config,
        adapter_path=config["paths"]["ppo_midpoint_policy"],
        trainable=False,
    )
    device = next(policy.parameters()).device
    for index in range(len(records), len(cached_rows)):
        row = cached_rows[index]
        if row["prompt_id"] not in prompts:
            raise KeyError(f"Cached prompt {row['prompt_id']} is not in the prompt pool.")
        prompt_row = prompts[row["prompt_id"]]
        if int(prompt_row["source_index"]) != int(row["source_index"]):
            raise ValueError("Cached source index does not match the prompt pool.")
        batch = encode_cached_response(
            tokenizer,
            prompt_messages(prompt_row),
            row["response"],
            int(config["max_prompt_length"]),
            int(row["response_tokens"]),
            bool(row["terminated_with_eos"]),
        )
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.inference_mode():
            new_logprobs, logits = response_token_logprobs(
                policy,
                batch["input_ids"],
                batch["attention_mask"],
                batch["prompt_width"],
                batch["response_ids"],
            )
            del logits
        record = {
            "prompt_id": row["prompt_id"],
            "source_index": int(row["source_index"]),
            "response_tokens": int(row["response_tokens"]),
            "new_logprobs": new_logprobs[0].float().cpu().tolist(),
        }
        append_jsonl(logprob_path, record)
        records.append(record)
        print(f"Cached rollout reconstruction {index + 1}/{len(cached_rows)}", flush=True)
    del policy, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def analyze_cached_rollouts(
    config_path: str = "configs/ppo.yaml", overwrite: bool = False
) -> list[dict]:
    """Calculate cached advantages and clipping metrics for all epsilons."""
    config = load_yaml(config_path)
    validate_shared_protocol(config)
    cached_rows = load_cached_rollouts(config["cached_rollouts"])
    result_dir = repo_path(config["results_dir"]) / "clipping"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "cached_rollout_metrics.json"
    protocol = _cached_protocol(config, cached_rows)
    protocol_path = result_dir / "cached_rollout_protocol.json"
    csv_path = result_dir / "cached_rollout_metrics.csv"
    if output_path.exists() and csv_path.exists() and protocol_path.exists() and not overwrite:
        if load_json(protocol_path) == protocol:
            print("Using completed cached-rollout clipping analysis.")
            return load_json(output_path)

    reconstructed = _reconstruct_midpoint_logprobs(
        config, cached_rows, result_dir, overwrite
    )
    old_vectors = []
    new_vectors = []
    advantage_vectors = []
    for cached, current in zip(cached_rows, reconstructed):
        old_logprobs = cached["old_logprobs"].reshape(1, -1)
        reference_logprobs = cached["ref_logprobs"].reshape(1, -1)
        values = cached["values"].reshape(1, -1)
        mask = torch.ones_like(old_logprobs)
        terminal_reward = torch.tensor(
            [float(cached["effective_terminal_reward"])], dtype=torch.float32
        )
        rewards = shaped_rewards(
            terminal_reward,
            old_logprobs,
            reference_logprobs,
            mask,
            float(config["kl_beta"]),
        )
        advantages, _ = compute_gae(
            rewards,
            values,
            mask,
            gamma=float(config["gamma"]),
            lam=float(config["gae_lambda"]),
        )
        new_vector = torch.tensor(current["new_logprobs"], dtype=torch.float32)
        if new_vector.numel() != old_logprobs.numel():
            raise ValueError("Reconstructed and cached response lengths do not match.")
        old_vectors.append(old_logprobs[0])
        new_vectors.append(new_vector)
        advantage_vectors.append(advantages[0])

    old_batch, response_mask = pad_token_vectors(old_vectors)
    new_batch, new_mask = pad_token_vectors(new_vectors)
    advantage_batch, advantage_mask = pad_token_vectors(advantage_vectors)
    if not torch.equal(response_mask, new_mask) or not torch.equal(
        response_mask, advantage_mask
    ):
        raise RuntimeError("Cached PPO tensors have inconsistent response masks.")
    normalized_advantages = normalize_advantages(advantage_batch, response_mask)
    metrics = compute_cached_clipping_metrics(
        new_batch,
        old_batch,
        normalized_advantages,
        response_mask,
        [float(value) for value in config["clip_values"]],
    )
    save_json(output_path, metrics)
    pd.DataFrame(metrics).to_csv(csv_path, index=False)
    return metrics


def run_clipping_study(
    config_path: str = "configs/ppo.yaml",
    overwrite: bool = False,
    cached_only: bool = False,
) -> pd.DataFrame:
    """Run cached diagnostics and all matched short epsilon continuations."""
    config = load_yaml(config_path)
    cached_metrics = analyze_cached_rollouts(config_path, overwrite=overwrite)
    if cached_only:
        return pd.DataFrame(cached_metrics)

    cached_by_epsilon = {
        float(row["clip_epsilon"]): row for row in cached_metrics
    }
    summaries = []
    records_by_run = {}
    for epsilon in [float(value) for value in config["clip_values"]]:
        run_name = f"clip_eps_{epsilon:.2f}"
        training = train_ppo_continuation(
            config_path,
            run_name=run_name,
            updates=int(config["fork_updates"]),
            clip_epsilon=epsilon,
            kl_beta=float(config["kl_beta"]),
            overwrite=overwrite,
        )
        evaluation = run_evaluation(
            config_path, name=run_name, overwrite=overwrite
        )
        run_directory = result_directory(config, run_name)
        history = read_jsonl(run_directory / "train_metrics.jsonl")
        stability = stability_summary(history)
        cached = cached_by_epsilon[epsilon]
        summaries.append({
            "run": run_name,
            "clip_epsilon": epsilon,
            "updates": training["updates"],
            "maximum_generation_budget": (
                training["updates"]
                * training["prompts_per_update"]
                * int(config["max_response_length"])
            ),
            "generated_tokens": training["generated_tokens"],
            "cached_clipped_surrogate": cached["clipped_surrogate"],
            "cached_clip_fraction": cached["clip_fraction"],
            "cached_affected_token_fraction": cached["affected_token_fraction"],
            "heldout_reward": evaluation["reward_model_score_mean"],
            "heldout_kl": evaluation["sampled_kl_token_mean"],
            "heldout_entropy": evaluation["entropy_token_mean"],
            "heldout_response_length": evaluation["response_length_mean"],
            **stability,
        })
        records_by_run[run_name] = read_jsonl(run_directory / "generations.jsonl")

    frame = pd.DataFrame(summaries).sort_values("clip_epsilon")
    study_dir = repo_path(config["results_dir"]) / "clipping"
    frame.to_csv(study_dir / "summary.csv", index=False)
    save_json(study_dir / "summary.json", frame.to_dict(orient="records"))
    comparisons = compare_condition_responses(records_by_run, "clip_eps_0.20")
    write_jsonl(study_dir / "qualitative_comparison.jsonl", comparisons)
    save_task2_figures(config["results_dir"])
    return frame


def main() -> None:
    """Parse command-line arguments and run the clipping study."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(run_clipping_study(args.config, args.overwrite, args.cached_only))


if __name__ == "__main__":
    main()
