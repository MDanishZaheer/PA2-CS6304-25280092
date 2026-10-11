# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build Task 4 tables, figures, agreement, and qualitative evidence."""

from __future__ import annotations

import argparse

from common.data import load_yaml
from task4_safety.analysis.aggregate import build_available_evidence
from task4_safety.analysis.plots import save_available_figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    parser.add_argument("--require-manual", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    evidence = build_available_evidence(
        config, require_manual=args.require_manual
    )
    figures = save_available_figures(config)
    print("Built Task 4 evidence:", sorted(evidence))
    print("Figures:", [str(path) for path in figures])


if __name__ == "__main__":
    main()
