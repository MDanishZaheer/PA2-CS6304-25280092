# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Keep Task 1 run artifacts in the agreed folder layout."""

from __future__ import annotations

from pathlib import Path

from common.data import repo_path


def result_directory(config: dict, run_name: str) -> Path:
    """Resolve a Task 1 result folder from a short run name."""
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one directory name.")
    root = repo_path(config["results_dir"])
    if run_name.startswith("beta_"):
        return root / "beta" / run_name
    if run_name.startswith("length_"):
        return root / "length" / run_name
    return root / run_name
