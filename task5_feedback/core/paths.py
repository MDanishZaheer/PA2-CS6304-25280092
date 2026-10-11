# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Keep Task 5 working caches and committed evidence in predictable places."""

from __future__ import annotations

from common.data import repo_path


POLICY_ORDER = ("sft", "rlvr", "rlaif")
DATASET_ORDER = ("gsm", "transfer")


def validate_policy_name(policy_name: str) -> None:
    if policy_name not in POLICY_ORDER:
        raise ValueError(f"Unknown Task 5 policy: {policy_name!r}.")


def validate_dataset_name(dataset: str) -> None:
    if dataset not in DATASET_ORDER:
        raise ValueError(f"Unknown Task 5 dataset: {dataset!r}.")


def output_root(config: dict):
    return repo_path(config["task5_output_dir"])


def response_path(config: dict, dataset: str, policy_name: str):
    validate_dataset_name(dataset)
    validate_policy_name(policy_name)
    return output_root(config) / dataset / "responses" / f"{policy_name}.jsonl"


def pairwise_path(config: dict, dataset: str):
    validate_dataset_name(dataset)
    return output_root(config) / dataset / "pairwise_judgments.jsonl"


def diagnostic_pairwise_path(config: dict):
    return output_root(config) / "diagnostics" / "pairwise_judgments.jsonl"


def judge_cache_path(config: dict):
    return output_root(config) / "ai_pairwise_cache.json"


def results_root(config: dict):
    return repo_path(config["task5_results_dir"])


def domain_directory(config: dict, dataset: str):
    validate_dataset_name(dataset)
    name = "in_domain" if dataset == "gsm" else "transfer"
    return results_root(config) / name


def diagnostics_directory(config: dict):
    return results_root(config) / "diagnostics"


def qualitative_directory(config: dict):
    return results_root(config) / "qualitative"


def figures_directory(config: dict):
    return results_root(config) / "figures"


def protocols_directory(config: dict):
    return results_root(config) / "protocols"


def raw_directory(config: dict):
    return results_root(config) / "raw"
