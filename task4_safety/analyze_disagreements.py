# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Rebuild manual agreement and qualitative disagreement evidence."""

from __future__ import annotations

import argparse

from common.data import load_yaml
from task4_safety.evaluation.agreement import build_manual_agreement
from task4_safety.evaluation.qualitative import select_qualitative_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    agreement = build_manual_agreement(
        config, require_complete=args.require_complete
    )
    comparison = agreement["comparison"] if agreement else None
    examples = select_qualitative_examples(config, comparison)
    print(f"Selected {len(examples)} qualitative examples.")


if __name__ == "__main__":
    main()
