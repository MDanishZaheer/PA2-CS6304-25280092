# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Command-line entry point for one Task 1 DPO training run."""

from __future__ import annotations

import argparse
import json

from task1_dpo.core.trainer import train_dpo


def main() -> None:
    """Parse a run configuration and train one fresh LoRA adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--run-name", default="standard")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument("--beta", type=float)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    summary = train_dpo(
        config_path=arguments.config,
        run_name=arguments.run_name,
        dataset_path=arguments.dataset,
        output_path=arguments.output,
        beta=arguments.beta,
        max_examples=arguments.max_examples,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
