# Task 2 — Proximal Policy Optimization

This folder implements the three Task 2 experiments from the supplied PPO
midpoint. The Python modules contain the experiment logic; the notebook only
runs stages and displays saved evidence.

## Folder layout

```text
task2_ppo/
├── ppo.py                    # corrected token-level PPO objective
├── continue_train.py         # Part 1: standard 20-update continuation
├── evaluate.py               # common fixed 200-prompt evaluation
├── analyze_clipping.py       # Part 2: cached analysis + epsilon forks
├── ablate_kl.py              # Part 3: KL-pressure forks
├── core/                     # rollout, batching, checkpoint, and training logic
├── evaluation/               # held-out metrics, stability, and examples
├── analysis/                 # saved-result tables and report figures
├── notebooks/
│   └── task2_ppo_workbench.ipynb
├── tests/
├── outputs/                  # adapters/optimizer states; ignored by Git
└── results/                  # small report evidence; committed to Git
```

The trained policies and critic adapters are stored under `outputs/`. Each run
keeps a versioned, complete checkpoint after every update, so rerunning the
same command resumes an incomplete continuation or reuses a completed one.
Partially written checkpoint directories are never selected. Held-out
generations and independent reward-model scores are also resumable and reused.

## Run order

Activate the `ATML` Conda environment and start commands from the repository
root.

```powershell
conda activate ATML
python -m unittest discover -s task2_ppo/tests -v

# Part 1: standard continuation and evaluation
python -m task2_ppo.continue_train
python -m task2_ppo.evaluate --name standard

# Part 2: cached clipping diagnostic and three matched 8-update forks
python -m task2_ppo.analyze_clipping

# Part 3: three matched 8-update KL-pressure forks
python -m task2_ppo.ablate_kl
```

The same stages can be run interactively from
`notebooks/task2_ppo_workbench.ipynb`. Its first section downloads/caches the
two public Hugging Face models and checks the released local assets. Parts 1,
2, and 3 are separated, so completing a later part never causes an earlier
part to retrain.

Use `--overwrite` only when an experiment must intentionally be restarted.
Without that flag, protocol files and checkpoint hashes prevent incompatible
artifacts from being silently mixed.

## Fixed comparisons

- Standard PPO: 20 updates, epsilon `0.20`, KL beta `0.10`.
- Clipping forks: epsilon in `{0.05, 0.20, 0.50}`, 8 updates each.
- KL forks: beta in `{0.00, 0.10, 0.20}`, 8 updates each.
- Every fork starts from the same released policy and critic midpoint.
- All held-out comparisons use the same 200 prompts, seeds, response cap, and
  decoding configuration.
- Assignment-fixed models, budgets, LoRA settings, caps, and decoding values
  are validated before expensive work starts. Over-budget causal prompts are
  left-truncated so the final user instruction and generation marker survive.
- Learned rewards are scored one response at a time. This avoids changing a
  score because an unrelated response changes dynamic batch padding.
- During PPO updates the frozen reward model stays on CPU so the trainable
  policy and critic fit on a 6 GB GPU. Held-out evaluation unloads the policy
  first, then uses automatic placement for the quantized reward model.

The cached clipping table reports two different quantities: `clip_fraction`
is the fraction of valid tokens whose ratio is outside the interval, while
`affected_token_fraction` is the fraction for which the clipped surrogate is
the conservative objective selected by PPO.

## Saved evidence

- `results/standard/train_metrics.jsonl`: all required per-update trajectories.
- `results/standard/train_summary.json`: update/token budgets, wall time, and
  peak VRAM.
- `results/*/eval_metrics.json`: held-out reward, KL, entropy, and length.
- `results/clipping/cached_rollout_metrics.csv`: immediate clipping geometry.
- `results/clipping/summary.csv`: matched epsilon fork outcomes and stability.
- `results/kl/summary.csv`: matched KL-pressure fork outcomes.
- `results/*/qualitative_*.jsonl`: aligned responses for manual
  reward-versus-quality analysis.
- `results/figures/`: report-ready plots regenerated only from saved results.
  This includes both final-condition comparisons and per-update clipping/KL
  trajectories needed to establish which behavior changed first.

The code selects qualitative candidates using numeric signals but deliberately
leaves the quality judgment to the report author. A reward score is evidence
about the fixed reward model, not a complete measure of correctness or quality.
