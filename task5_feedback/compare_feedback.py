# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build the final Task 5 comparison, figures, and RQ evidence."""

from __future__ import annotations

import argparse

from common.data import load_yaml
from task5_feedback.analysis.aggregate import build_all_evidence
from task5_feedback.analysis.plots import save_all_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    evidence = build_all_evidence(config)
    figures = save_all_figures(config)
    print("Built Task 5 evidence:", ", ".join(sorted(evidence)))
    print("Figures:", [str(path) for path in figures])


if __name__ == "__main__":
    main()
