# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Run one complete frozen-policy GSM8K or SVAMP evaluation."""

from __future__ import annotations

import argparse

from common.data import load_yaml
from task5_feedback.core.paths import POLICY_ORDER
from task5_feedback.core.protocol import load_math_rows
from task5_feedback.evaluation.generation import generate_dataset_responses
from task5_feedback.evaluation.math_metrics import build_math_tables
from task5_feedback.evaluation.pairwise import judge_dataset_pairs


def policy_specs(config: dict):
    return {name: config["policies"][name] for name in POLICY_ORDER}


def load_math_evaluation(config_path: str, dataset: str):
    config = load_yaml(config_path)
    return config, load_math_rows(config, dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    parser.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    parser.add_argument(
        "--stage",
        choices=["all", "generate", "judge", "summarize"],
        default="all",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)

    if args.stage in {"all", "generate"}:
        generate_dataset_responses(
            config,
            args.dataset,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    if args.stage in {"all", "judge"}:
        judge_dataset_pairs(config, args.dataset, overwrite=args.overwrite)
    if args.stage in {"all", "summarize"}:
        tables = build_math_tables(config, args.dataset)
        print(f"Built {args.dataset} evidence:", ", ".join(sorted(tables)))


if __name__ == "__main__":
    main()
