# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Create or validate the fixed 60-row, four-policy manual audit sheet."""

from __future__ import annotations

import argparse

from common.data import load_yaml
from task4_safety.evaluation.manual_audit import (
    create_manual_audit_sheet,
    fixed_audit_ids,
    validate_manual_sheet,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.check:
        print(validate_manual_sheet(
            config, require_complete=args.require_complete
        ))
    else:
        create_manual_audit_sheet(config)


if __name__ == "__main__":
    main()
