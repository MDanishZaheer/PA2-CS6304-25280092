# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Resolve Task 2 run directories without hiding experiment boundaries."""

from __future__ import annotations

from common.data import repo_path


def run_group(run_name: str) -> str | None:
    """Map a run name to its controlled-study result group."""
    if run_name == "standard":
        return None
    if run_name.startswith("clip_eps_"):
        return "clipping"
    if run_name.startswith("kl_beta_"):
        return "kl"
    raise ValueError(f"Unknown Task 2 run name: {run_name}")


def result_directory(config: dict, run_name: str):
    """Return the committed result directory for one PPO run."""
    root = repo_path(config["results_dir"])
    group = run_group(run_name)
    return root / run_name if group is None else root / group / run_name


def output_directory(config: dict, run_name: str):
    """Return the ignored policy/value output directory for one PPO run."""
    standard = repo_path(config["output"])
    return standard if run_name == "standard" else standard.parent / run_name


def figures_directory(config: dict):
    """Return the common Task 2 figure directory."""
    return repo_path(config["results_dir"]) / "figures"
