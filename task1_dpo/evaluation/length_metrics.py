# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Summarize preference fit by length stratum and word-limit compliance."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from common.data import preference_responses


EXPECTED_STRATA = {"preferred_longer", "length_matched", "rejected_longer"}


def preference_by_stratum(pair_records: list[dict], policy_name: str) -> list[dict]:
    """Calculate accuracy separately for the course's three length strata."""
    groups = defaultdict(list)
    for record in pair_records:
        groups[record["length_stratum"]].append(record)
    if set(groups) != EXPECTED_STRATA:
        raise ValueError(f"Unexpected length strata: {sorted(groups)}")
    return [
        {
            "policy": policy_name,
            "length_stratum": name,
            "pairs": len(groups[name]),
            "preference_accuracy": sum(row["preference_correct"] for row in groups[name])
            / len(groups[name]),
            "mean_preference_margin": float(
                np.mean([row["preference_margin"] for row in groups[name]])
            ),
        }
        for name in ("preferred_longer", "length_matched", "rejected_longer")
    ]


def word_limit_summary(records: list[dict], policy_name: str) -> dict:
    """Return generated length and explicit limit compliance on common prompts."""
    if not records or any(row["policy_word_limit_compliance"] is None for row in records):
        raise ValueError("Every word-limit prompt must contain a parseable limit.")
    lengths = np.asarray([row["response_length"] for row in records], dtype=float)
    return {
        "policy": policy_name,
        "prompts": len(records),
        "response_length_mean": float(lengths.mean()),
        "response_length_std": float(lengths.std(ddof=0)),
        "word_limit_compliance": float(
            np.mean([row["policy_word_limit_compliance"] for row in records])
        ),
    }


def dataset_length_summary(rows: list[dict], tokenizer, name: str) -> dict:
    """Describe response-token length correlation in a training-pair file."""
    differences = []
    chosen_lengths = []
    rejected_lengths = []
    for row in rows:
        chosen, rejected = preference_responses(row)
        chosen_length = len(tokenizer(chosen, add_special_tokens=False)["input_ids"])
        rejected_length = len(tokenizer(rejected, add_special_tokens=False)["input_ids"])
        chosen_lengths.append(chosen_length)
        rejected_lengths.append(rejected_length)
        differences.append(chosen_length - rejected_length)
    values = np.asarray(differences)
    return {
        "dataset": name,
        "pairs": len(rows),
        "chosen_tokens_mean": float(np.mean(chosen_lengths)),
        "rejected_tokens_mean": float(np.mean(rejected_lengths)),
        "token_difference_mean": float(values.mean()),
        "chosen_longer_fraction": float(np.mean(values > 0)),
        "rejected_longer_fraction": float(np.mean(values < 0)),
        "exact_length_tie_fraction": float(np.mean(values == 0)),
    }
