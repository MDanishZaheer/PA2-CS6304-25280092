# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Run the three matched short-budget DPO beta conditions."""

from __future__ import annotations

import argparse
import csv
import json

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task1_dpo.core.trainer import train_dpo, validate_saved_run
from task1_dpo.evaluate import run_evaluation


def run_beta_study(config_path: str = "configs/dpo.yaml", overwrite: bool = False):
    """Train each beta from the same base and evaluate the fixed held-out set."""
    config = load_yaml(config_path)
    values = [float(value) for value in config["betas"]]
    if values != [0.03, 0.10, 0.30]:
        raise ValueError("Task 1 requires beta values 0.03, 0.10, and 0.30.")
    results = []
    for beta in values:
        name = f"beta_{beta:.2f}"
        adapter = repo_path("task1_dpo/outputs") / name
        result_folder = repo_path(config["results_dir"]) / "beta" / name
        if overwrite or not (adapter / "adapter_config.json").exists():
            train_dpo(
                config_path=config_path,
                run_name=name,
                output_path=str(adapter),
                beta=beta,
                max_examples=int(config["short_ablation_examples"]),
                overwrite=overwrite,
            )
        else:
            validate_saved_run(
                config, name, str(adapter), config["paths"]["dpo_standard_train"],
                beta, int(config["short_ablation_examples"]),
            )
        if overwrite or not (result_folder / "eval_metrics.json").exists():
            metrics = run_evaluation(
                config_path=config_path,
                adapter=str(adapter),
                name=name,
                beta=beta,
            )
        else:
            metrics = load_json(result_folder / "eval_metrics.json")
        save_json(repo_path(config["results_dir"]) / "beta" / f"{name}.json", metrics)
        results.append(metrics)
    summary = repo_path(config["results_dir"]) / "beta" / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


def main() -> None:
    """Run the fixed beta study from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run_beta_study(arguments.config, arguments.overwrite), indent=2))


if __name__ == "__main__":
    main()
