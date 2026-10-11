# Task 1: Direct Preference Optimization

Run these commands from the repository root in the `ATML` Conda environment. The course
datasets must pass `python -m scripts.validate_assets` first. The reference policy is the
same frozen Qwen base used to initialize each fresh LoRA adapter; the adapter is disabled
when reference probabilities are calculated.

## Validate the objective

```powershell
python -m unittest discover -s task1_dpo/tests -v
```

## Required experiments

```powershell
python -m task1_dpo.train --run-name standard
python -m task1_dpo.evaluate --name standard
python -m task1_dpo.ablate_beta
python -m task1_dpo.analyze_length
```

If the responses were generated before independent reward scoring was added, correct
only their saved reward scores with:

```powershell
python -m task1_dpo.rescore_rewards
```

This command does not train adapters, generate text, recompute preference accuracy, or
recompute KL. It loads the reward model once, scores each unique prompt-response pair
in a one-item batch, and refreshes reward-dependent JSON/CSV files and the beta figure.
Progress is appended to `results/reward_score_cache.jsonl`, so an interrupted run resumes
from its completed scores. Identical reference responses are scored once and reused.

The standard and length-balanced adapters each train for one epoch on 1,500 fixed pairs.
Each beta fork starts again from Qwen with a fresh LoRA adapter and trains on the same
first 600 standard pairs. Evaluation uses all 300 standard held-out pairs. The length
study uses all 246 supplied stratified pairs and the 10 fixed word-limit prompts.
Within each configured batch of two pairs, training processes one pair at a time and
averages its gradient over the same accumulation group to fit smaller GPUs.
`--overwrite` on training/study commands permits an intentional rerun. Do not compare
the standard run and beta forks as equal training budgets.
The length study saves `results/length/study_summary.json` last. On later runs it
reuses a complete evaluation when both adapter hashes and evaluation settings match.

The notebook at `task1_dpo/notebooks/task1_dpo_workbench.ipynb` runs these stages
interactively and displays tables, plots, and responses for manual review. Its first
setup block checks the Hugging Face cache and downloads the public policy and reward
models if they are missing. Its final evidence section runs the reward-only correction
when needed and displays the complete Task 1 summary, length diagnostics, figures, and
qualitative review worksheets. The Python
commands above are the reproducible experiment pipeline.

Adapters are saved under `task1_dpo/outputs/` and ignored by Git. Machine-readable
training logs, prompt IDs, configuration snapshots, per-example outputs, summaries,
and figures belong under `task1_dpo/results/` and should be committed. A higher reward
or preference margin is not a manual quality label; inspect candidate responses before
making qualitative claims.
