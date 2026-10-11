# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Command-line entry point for the standard PPO continuation."""

from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from task2_ppo.core.paths import output_directory
from task2_ppo.core.trainer import train_ppo_continuation


def run_ppo(
    config_path: str = "configs/ppo.yaml",
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
    overwrite: bool = False,
) -> dict:
    """Train or resume one PPO continuation using the shared pipeline."""
    config = load_yaml(config_path)
    expected_output = output_directory(config, run_name).resolve()
    if output is not None and repo_path(output).resolve() != expected_output:
        raise ValueError(
            "Task 2 output locations are fixed by the run name. "
            f"Expected {expected_output}."
        )
    return train_ppo_continuation(
        config_path=config_path,
        run_name=run_name,
        updates=updates,
        clip_epsilon=clip_epsilon,
        kl_beta=kl_beta,
        overwrite=overwrite,
    )


def main() -> None:
    """Parse command-line arguments and run PPO training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--output")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--clip-epsilon", type=float)
    parser.add_argument("--kl-beta", type=float)
    parser.add_argument("--run-name", default="standard")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_ppo(
        args.config,
        args.output,
        args.updates,
        args.clip_epsilon,
        args.kl_beta,
        args.run_name,
        args.overwrite,
    )
    print(summary)


if __name__ == "__main__":
    main()
