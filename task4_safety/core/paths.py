# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Resolve Task 4 paths without scattering directory names through the code."""

from __future__ import annotations

from common.data import repo_path


POLICY_ORDER = ("sft", "dpo", "ppo", "grpo")


def validate_policy_name(policy_name: str) -> None:
    """Reject policy names outside the four assignment conditions."""
    if policy_name not in POLICY_ORDER:
        raise ValueError(f"Unknown Task 4 policy: {policy_name!r}.")


def output_root(config: dict):
    """Return the ignored directory containing expensive raw inference."""
    return repo_path(config["task4_output_dir"])


def response_path(config: dict, policy_name: str):
    """Return the resumable response file for one frozen policy."""
    validate_policy_name(policy_name)
    return output_root(config) / "responses" / f"{policy_name}.jsonl"


def judgment_path(config: dict, policy_name: str):
    """Return the resumable AI-judge file for one frozen policy."""
    validate_policy_name(policy_name)
    return output_root(config) / "ai_judgments" / f"{policy_name}.jsonl"


def results_root(config: dict):
    """Return the committed Task 4 evidence directory."""
    return repo_path(config["task4_results_dir"])


def summary_directory(config: dict):
    return results_root(config) / "summary"


def category_directory(config: dict):
    return results_root(config) / "categories"


def manual_directory(config: dict):
    return results_root(config) / "manual"


def qualitative_directory(config: dict):
    return results_root(config) / "qualitative"


def protocol_directory(config: dict):
    return results_root(config) / "protocols"


def figures_directory(config: dict):
    return results_root(config) / "figures"


def raw_results_directory(config: dict):
    """Return the committed response-level evidence directory."""
    return results_root(config) / "raw"
