# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""CLI for resumable deterministic generation from four frozen policies."""

from __future__ import annotations

import argparse

from common.data import load_yaml
from task4_safety.core.paths import POLICY_ORDER
from task4_safety.evaluation.generation import generate_all_policies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    parser.add_argument("--policy", choices=["all", *POLICY_ORDER], default="all")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    policy_names = POLICY_ORDER if args.policy == "all" else (args.policy,)
    generate_all_policies(
        config,
        policy_names=policy_names,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
