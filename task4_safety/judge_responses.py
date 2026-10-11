# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""CLI and compatibility exports for the fixed Task 4 AI judge."""

from __future__ import annotations

import argparse

from common.data import load_yaml
from task4_safety.core.paths import POLICY_ORDER
from task4_safety.evaluation.ai_judge import (
    JUDGE_PROMPT,
    LABELS,
    judge_all_responses,
    judge_batch,
    judge_one,
    load_judge,
    parse_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    parser.add_argument("--policy", choices=["all", *POLICY_ORDER], default="all")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    policy_names = POLICY_ORDER if args.policy == "all" else (args.policy,)
    judge_all_responses(
        config,
        policy_names=policy_names,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
