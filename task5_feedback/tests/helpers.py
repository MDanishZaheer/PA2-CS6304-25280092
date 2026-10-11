# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Synthetic Task 5 artifacts that avoid model inference during tests."""

from __future__ import annotations

import copy
import hashlib

from common.data import load_yaml, write_jsonl
from task5_feedback.core.paths import (
    POLICY_ORDER,
    diagnostic_pairwise_path,
    pairwise_path,
    response_path,
)
from task5_feedback.core.protocol import load_math_rows, row_id
from task5_feedback.evaluation.diagnostics import expected_diagnostic_pairs
from task5_feedback.evaluation.pairwise import (
    expected_policy_pairs,
    preference_from_rewards,
    winner_name,
)


def temporary_config(root):
    config = copy.deepcopy(load_yaml("configs/feedback.yaml"))
    config["task5_output_dir"] = str(root / "outputs")
    config["task5_results_dir"] = str(root / "results")
    return config


def write_synthetic_responses(config, dataset):
    rows = load_math_rows(config, dataset)
    for policy_name in POLICY_ORDER:
        records = []
        for source in rows:
            response = f"A short correct solution from {policy_name}.\n#### {source['gold_final']}"
            records.append({
                "dataset": dataset,
                "prompt_id": row_id(source, dataset),
                "source_index": source["source_index"],
                "policy": policy_name,
                "question": str(source["question"]),
                "messages": source["messages"],
                "gold_final": str(source["gold_final"]),
                "response": response,
                "response_tokens": 12,
                "terminated_with_eos": True,
                "truncated": False,
                "predicted_final": str(source["gold_final"]),
                "format_compliant": True,
                "exact_reward": 1.0,
                "category": (
                    str(source.get("svamp_type", ""))
                    if dataset == "transfer"
                    else "GSM8K"
                ),
            })
        write_jsonl(response_path(config, dataset, policy_name), records)


def _judge_fields(preference, verifier, identity):
    key = hashlib.sha256(identity.encode()).hexdigest()
    return {
        "judge_preference": preference,
        "verifier_preference": verifier,
        "verifier_judge_agree": preference == verifier,
        "raw_judge_output": preference,
        "judge_swapped": False,
        "judge_parse_failure": False,
        "judge_input_tokens": 100,
        "judge_inference_seconds": 0.01,
        "judge_cache_hit": False,
        "judge_cache_key": key,
    }


def write_synthetic_policy_pairs(config, dataset):
    records = []
    for index, pair in enumerate(expected_policy_pairs(config, dataset)):
        verifier = preference_from_rewards(
            pair["exact_reward_a"], pair["exact_reward_b"]
        )
        preference = "A" if pair["candidate_a"] != "sft" else "TIE"
        records.append({
            **pair,
            "response_a": None,
            "response_b": None,
            "judge_winner": winner_name(
                preference, pair["candidate_a"], pair["candidate_b"]
            ),
            "verifier_winner": winner_name(
                verifier, pair["candidate_a"], pair["candidate_b"]
            ),
            **_judge_fields(preference, verifier, f"{dataset}:{index}"),
        })
    write_jsonl(pairwise_path(config, dataset), records)


def write_synthetic_diagnostic_pairs(config):
    records = []
    for index, pair in enumerate(expected_diagnostic_pairs(config)):
        verifier = preference_from_rewards(
            pair["exact_reward_a"], pair["exact_reward_b"]
        )
        preference = "A" if pair["diagnostic_preference"] == "A" else "TIE"
        records.append({
            **pair,
            "response_a": None,
            "response_b": None,
            **_judge_fields(preference, verifier, f"diagnostic:{index}"),
        })
    write_jsonl(diagnostic_pairwise_path(config), records)
