# Task 3 - GRPO

This folder implements all three Task 3 parts from the released midpoint and fixed caches. The assignment constants in `configs/grpo.yaml` are validated before any expensive work begins.

## Recommended workflow

Use `notebooks/task3_grpo_workbench.ipynb`. Run **Shared Setup** once, then run only the cells inside the part you need.

The equivalent command-line workflow is:

```powershell
# Part 1: 20-update standard continuation and fixed held-out evaluation
python -m task3_grpo.continue_train --run-name standard --loss-type grpo
python -m task3_grpo.evaluate --name standard

# Part 2: K={2,4,8} fixed-cache study; no model or training is needed
python -m task3_grpo.analyze_group_size

# Part 3: matched 8-update canonical and Dr-GRPO forks plus evaluation
python -m task3_grpo.compare_normalization
```

Run validation before training:

```powershell
python -m unittest discover -s task3_grpo/tests -v
```

## Resume and reuse behavior

- Training saves an adapter and optimizer state after every completed update.
- A stopped run resumes from its latest hash-verified checkpoint.
- A completed run is validated and reused instead of retrained.
- Evaluation appends one prompt at a time and independently caches reward scores.
- `OVERWRITE=False` is the safe notebook default. Set it to `True` only when you intentionally want to delete and replace one run's artifacts.

The reward model stays on CPU during training so the policy has the laptop GPU to itself. Evaluation unloads the policy before loading the quantized reward model automatically.

## Evidence produced

- `results/standard/`: training history, completion diagnostics, held-out generations/metrics, qualitative candidates, runtime/VRAM, and stability diagnostics.
- `results/group_size/`: equal-budget overall and difficulty-bin tables, per-group records, bin definition, and qualitative examples.
- `results/normalization/`: matched summary, fixed length-bin gradient-allocation statistics, paired held-out differences, and qualitative comparisons.
- `results/figures/`: standard trajectory, group-size, difficulty, normalization, and length-allocation figures.

The qualitative files deliberately leave manual assessment fields blank. Fill those yourself after reading the responses, because the assignment requires your own report interpretation.
