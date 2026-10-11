# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Run the matched eight-update canonical GRPO versus Dr-GRPO study."""

from __future__ import annotations

import argparse

from common.data import load_yaml
from task3_grpo.analysis.aggregate import normalization_evidence
from task3_grpo.analysis.plots import save_available_figures
from task3_grpo.core.protocol import validate_shared_protocol
from task3_grpo.core.trainer import train_grpo_continuation
from task3_grpo.evaluate import run_evaluation


def run_normalization_study(
    config_path: str = "configs/grpo.yaml",
    overwrite: bool = False,
) -> dict:
    """Train/evaluate both matched forks, reusing verified completed work."""
    config = load_yaml(config_path)
    validate_shared_protocol(config)
    budget = int(config["fork_updates"])
    conditions = [
        ("norm_canonical", "grpo"),
        ("norm_dr_grpo", "dr_grpo"),
    ]
    output = {}
    for run_name, loss_type in conditions:
        output[f"{run_name}_training"] = train_grpo_continuation(
            config_path=config_path,
            run_name=run_name,
            updates=budget,
            loss_type=loss_type,
            overwrite=overwrite,
        )
        output[f"{run_name}_evaluation"] = run_evaluation(
            config_path=config_path,
            name=run_name,
            overwrite=overwrite,
        )
    summary, length = normalization_evidence(config)
    save_available_figures(config)
    output["summary"] = summary.to_dict(orient="records")
    output["length_conditioned"] = length.to_dict(orient="records")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_normalization_study(args.config, args.overwrite)
    print("Completed normalization conditions:", [
        row["run_name"] for row in result["summary"]
    ])


if __name__ == "__main__":
    main()
