# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Resolve Task 3 output and result paths from a small set of run names."""

from __future__ import annotations

from common.data import repo_path


ALLOWED_RUNS = {"standard", "norm_canonical", "norm_dr_grpo"}


def _validate_run_name(run_name: str) -> None:
    if run_name not in ALLOWED_RUNS:
        raise ValueError(f"Unknown Task 3 run name: {run_name!r}.")


def output_directory(config: dict, run_name: str):
    """Return the ignored adapter/checkpoint directory for one run."""
    _validate_run_name(run_name)
    standard = repo_path(config["output"])
    return standard if run_name == "standard" else standard.parent / run_name


def result_directory(config: dict, run_name: str):
    """Return the committed evidence directory for one run."""
    _validate_run_name(run_name)
    root = repo_path(config["results_dir"])
    if run_name == "standard":
        return root / "standard"
    return root / "normalization" / run_name


def group_size_directory(config: dict):
    """Return the result directory for the fixed-cache K study."""
    return repo_path(config["results_dir"]) / "group_size"


def figures_directory(config: dict):
    """Return the common Task 3 figure directory."""
    return repo_path(config["results_dir"]) / "figures"
