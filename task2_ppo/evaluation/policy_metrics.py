# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Attach independent reward scores and summarize PPO policy behavior."""

from __future__ import annotations

import gc
import hashlib
import json
import math

import numpy as np
import torch

from common.data import read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, save_json
from common.models import load_reward_model


def reward_scoring_protocol(config: dict) -> dict:
    """Describe the padding-independent Task 2 reward evaluation."""
    return {
        "reward_model": config["reward_model"],
        "reward_tokenizer": config.get("reward_tokenizer", config["base_model"]),
        "max_length": int(config["reward_max_length"]),
        "batch_size": 1,
        "truncation_side": "left",
        "device_placement": "auto",
        "quantized_on_cuda": bool(config["quantize_frozen_models"]),
        "method": "each prompt-response pair is scored independently",
    }


def _score_key(protocol: dict, messages: list[dict], response: str) -> str:
    payload = {"protocol": protocol, "messages": messages, "response": response}
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def add_reward_scores(records: list[dict], config: dict) -> None:
    """Use a resumable shared cache to attach one-item-batch reward scores."""
    protocol = reward_scoring_protocol(config)
    protocol_text = json.dumps(protocol, sort_keys=True, ensure_ascii=False)
    protocol_hash = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
    results_root = repo_path(config["results_dir"])
    cache_path = results_root / "reward_score_cache.jsonl"
    scores = {}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            if row.get("protocol_sha256") == protocol_hash:
                scores[row["response_sha256"]] = float(row["reward_score"])
    pending = {}
    for record in records:
        key = _score_key(protocol, record["messages"], record["response"])
        if key not in scores:
            pending.setdefault(key, record)
    if pending:
        reward_model, reward_tokenizer = load_reward_model(config)
        for number, (key, record) in enumerate(pending.items(), start=1):
            score = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                [record["messages"]],
                [record["response"]],
                max_length=int(config["reward_max_length"]),
            )
            value = float(score[0])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite reward score for {record['prompt_id']}.")
            scores[key] = value
            append_jsonl(cache_path, {
                "protocol_sha256": protocol_hash,
                "response_sha256": key,
                "prompt_id": record["prompt_id"],
                "reward_score": value,
            })
            if number % 25 == 0 or number == len(pending):
                print(f"Reward scoring {number}/{len(pending)}", flush=True)
        del reward_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for record in records:
        key = _score_key(protocol, record["messages"], record["response"])
        record["reward_model_score"] = scores[key]
    save_json(results_root / "reward_scoring_protocol.json", {
        **protocol,
        "protocol_sha256": protocol_hash,
        "cache_file": str(cache_path),
    })


def summarize_policy(records: list[dict]) -> dict:
    """Aggregate all held-out metrics using response-token weighting."""
    if not records:
        raise ValueError("No held-out PPO records were supplied.")
    lengths = np.asarray([row["response_length"] for row in records], dtype=float)
    token_count = max(int(lengths.sum()), 1)
    rewards = np.asarray([row["reward_model_score"] for row in records], dtype=float)
    return {
        "evaluation_prompts": len(records),
        "response_tokens": int(lengths.sum()),
        "reward_model_score_mean": float(rewards.mean()),
        "reward_model_score_std": float(rewards.std(ddof=0)),
        "sampled_kl_token_mean": float(
            sum(row["sampled_kl_sequence_sum"] for row in records) / token_count
        ),
        "entropy_token_mean": float(
            sum(row["entropy_sequence_sum"] for row in records) / token_count
        ),
        "response_length_mean": float(lengths.mean()),
        "response_length_std": float(lengths.std(ddof=0)),
        "response_length_iqr": float(
            np.percentile(lengths, 75) - np.percentile(lengths, 25)
        ),
        "terminated_fraction": float(
            np.mean([row["terminated_with_eos"] for row in records])
        ),
        "truncated_fraction": float(np.mean([row["truncated"] for row in records])),
    }
