# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Command-line entry point for resumable Task 3 GRPO training."""

from __future__ import annotations

import argparse

from task3_grpo.core.trainer import train_grpo_continuation


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
    overwrite: bool = False,
) -> dict:
    """Run a validated continuation; custom output paths are intentionally blocked."""
    if output is not None:
        raise ValueError(
            "Task 3 output paths are fixed by configs/grpo.yaml for safe resume."
        )
    return train_grpo_continuation(
        config_path=config_path,
        run_name=run_name,
        updates=updates,
        loss_type=loss_type,
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    parser.add_argument("--run-name", default="standard")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_grpo(
        args.config,
        updates=args.updates,
        loss_type=args.loss_type,
        run_name=args.run_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
