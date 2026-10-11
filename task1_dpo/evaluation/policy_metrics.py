# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Score generated responses and summarize Task 1 policy behavior."""

from __future__ import annotations

import numpy as np

from common.generation import score_reward_pairs
from common.metrics import word_limit_compliance


REWARD_MAX_LENGTH = 1024


def reward_scoring_protocol(config: dict) -> dict:
    """Describe the fixed, padding-independent reward evaluation."""
    return {
        "reward_model": config["reward_model"],
        "reward_tokenizer": config.get("reward_tokenizer", config["base_model"]),
        "max_length": REWARD_MAX_LENGTH,
        "batch_size": 1,
        "method": "each prompt-response pair is scored independently",
    }


def score_one_response(
    reward_model, reward_tokenizer, messages: list[dict], response: str
) -> float:
    """Score one response without padding it against an unrelated response."""
    scores = score_reward_pairs(
        reward_model,
        reward_tokenizer,
        [messages],
        [response],
        max_length=REWARD_MAX_LENGTH,
    )
    return float(scores[0])


def add_reward_scores(records: list[dict], reward_model, reward_tokenizer) -> None:
    """Attach independent fixed reward-model scores to both responses."""
    for record in records:
        record["policy_reward_score"] = score_one_response(
            reward_model,
            reward_tokenizer,
            record["messages"],
            record["policy_response"],
        )
        record["reference_reward_score"] = score_one_response(
            reward_model,
            reward_tokenizer,
            record["messages"],
            record["reference_response"],
        )


def summarize_generation(records: list[dict], sampled_kl: float) -> dict:
    """Summarize reward, reference drift, and response-length distribution."""
    if not records:
        raise ValueError("No generated responses were supplied.")
    lengths = np.asarray([row["response_length"] for row in records], dtype=float)
    result = {
        "generated_prompts": len(records),
        "sampled_kl_token_mean": float(sampled_kl),
        "response_length_mean": float(lengths.mean()),
        "response_length_std": float(lengths.std(ddof=0)),
        "response_length_iqr": float(np.percentile(lengths, 75) - np.percentile(lengths, 25)),
        "truncated_fraction": sum(row["policy_truncated"] for row in records) / len(records),
    }
    if "policy_reward_score" in records[0]:
        result["reward_model_score_mean"] = float(
            np.mean([row["policy_reward_score"] for row in records])
        )
        result["reference_reward_score_mean"] = float(
            np.mean([row["reference_reward_score"] for row in records])
        )
    return result


def add_word_limit_fields(records: list[dict]) -> None:
    """Attach exact word-limit compliance for explicit word-limit prompts."""
    for record in records:
        prompt = record["prompt"]
        record["policy_word_limit_compliance"] = word_limit_compliance(
            prompt, record["policy_response"]
        )
        record["reference_word_limit_compliance"] = word_limit_compliance(
            prompt, record["reference_response"]
        )
