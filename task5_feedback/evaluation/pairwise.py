# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Apply the fixed pairwise judge to frozen-policy math responses."""

from __future__ import annotations

import hashlib

from common.data import read_jsonl
from common.logging_utils import append_jsonl, load_json, save_json
from task5_feedback.core.paths import (
    POLICY_ORDER,
    judge_cache_path,
    pairwise_path,
    protocols_directory,
    response_path,
)
from task5_feedback.core.protocol import (
    file_sha256,
    load_math_rows,
    protect_protocol,
    validate_config,
)
from task5_feedback.core.records import (
    validate_complete_responses,
    validate_pairwise_prefix,
)
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge


PAIR_SPECS = (
    ("rlvr_vs_sft", "rlvr", "sft"),
    ("rlaif_vs_sft", "rlaif", "sft"),
    ("rlaif_vs_rlvr", "rlaif", "rlvr"),
)


def text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def preference_from_rewards(a: float, b: float) -> str:
    if a > b:
        return "A"
    if b > a:
        return "B"
    return "TIE"


def winner_name(preference: str, candidate_a: str, candidate_b: str):
    if preference == "A":
        return candidate_a
    if preference == "B":
        return candidate_b
    return "tie"


def load_dataset_responses(config: dict, dataset: str) -> dict[str, list[dict]]:
    rows = load_math_rows(config, dataset)
    policies = {}
    for policy_name in POLICY_ORDER:
        path = response_path(config, dataset, policy_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Generate {dataset}/{policy_name} responses first: {path}"
            )
        records = read_jsonl(path)
        validate_complete_responses(records, rows, dataset, policy_name)
        policies[policy_name] = records
    return policies


def expected_policy_pairs(config: dict, dataset: str) -> list[dict]:
    policies = load_dataset_responses(config, dataset)
    expected = []
    for index in range(len(policies["sft"])):
        for pair_name, candidate_a, candidate_b in PAIR_SPECS:
            a = policies[candidate_a][index]
            b = policies[candidate_b][index]
            expected.append({
                "dataset": dataset,
                "prompt_id": a["prompt_id"],
                "source_index": a["source_index"],
                "question": a["question"],
                "gold_final": a["gold_final"],
                "pair_name": pair_name,
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "response_a": a["response"],
                "response_b": b["response"],
                "response_a_sha256": text_sha256(a["response"]),
                "response_b_sha256": text_sha256(b["response"]),
                "exact_reward_a": float(a["exact_reward"]),
                "exact_reward_b": float(b["exact_reward"]),
            })
    return expected


def pairwise_protocol(config: dict, dataset: str, expected: list[dict]) -> dict:
    return {
        "dataset": dataset,
        "comparison_count": len(expected),
        "pair_order": [name for name, _, _ in PAIR_SPECS],
        "judge_model": config["ai_judge_model"],
        "rubric_sha256": hashlib.sha256(PAIRWISE_RUBRIC.encode()).hexdigest(),
        "orientation": "released deterministic SHA-256 A/B balancing",
        "max_input_length": int(config["task5_judge_max_input_length"]),
        "max_new_tokens": int(config["task5_judge_max_new_tokens"]),
        "response_files": {
            name: file_sha256(response_path(config, dataset, name))
            for name in POLICY_ORDER
        },
    }


def judge_dataset_pairs(
    config: dict, dataset: str, overwrite: bool = False
) -> list[dict]:
    """Judge all missing policy pairs and safely resume at comparison level."""
    validate_config(config)
    expected = expected_policy_pairs(config, dataset)
    destination = pairwise_path(config, dataset)
    protocol_path = protocols_directory(config) / f"pairwise_{dataset}.json"
    protocol = pairwise_protocol(config, dataset, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if destination.exists():
            destination.unlink()
        protect_protocol(protocol_path, protocol, overwrite=True)
    else:
        protect_protocol(protocol_path, protocol)
    existing = read_jsonl(destination) if destination.exists() else []
    validate_pairwise_prefix(existing, expected)
    if len(existing) == len(expected):
        print(f"Using completed {dataset} pairwise judgments.")
        return existing

    judge = PairwiseAIJudge(config, judge_cache_path(config))
    try:
        for index in range(len(existing), len(expected)):
            pair = expected[index]
            detail = judge.compare_detail(
                pair["question"], pair["response_a"], pair["response_b"]
            )
            verifier_preference = preference_from_rewards(
                pair["exact_reward_a"], pair["exact_reward_b"]
            )
            append_jsonl(destination, {
                **pair,
                "response_a": None,
                "response_b": None,
                "judge_preference": detail["preference"],
                "judge_winner": winner_name(
                    detail["preference"],
                    pair["candidate_a"],
                    pair["candidate_b"],
                ),
                "verifier_preference": verifier_preference,
                "verifier_winner": winner_name(
                    verifier_preference,
                    pair["candidate_a"],
                    pair["candidate_b"],
                ),
                "verifier_judge_agree": (
                    detail["preference"] == verifier_preference
                ),
                "raw_judge_output": detail["raw_output"],
                "judge_swapped": detail["swapped"],
                "judge_parse_failure": detail["parse_failure"],
                "judge_input_tokens": detail["input_tokens"],
                "judge_inference_seconds": detail["inference_seconds"],
                "judge_cache_hit": detail["cache_hit"],
                "judge_cache_key": detail["cache_key"],
            })
            if (index + 1) % 20 == 0 or index + 1 == len(expected):
                print(
                    f"{dataset}: judged {index + 1}/{len(expected)} pairs",
                    flush=True,
                )
    finally:
        usage = judge.usage()
        judge.close()
        usage_path = destination.parent / "judge_usage.json"
        previous = load_json(usage_path) if usage_path.exists() else {}
        cumulative = {
            key: float(previous.get(key, 0.0)) + float(value)
            for key, value in usage.items()
        }
        save_json(usage_path, cumulative)

    records = read_jsonl(destination)
    validate_pairwise_prefix(records, expected)
    if len(records) != len(expected):
        raise RuntimeError(f"Incomplete {dataset} pairwise evaluation.")
    return records
