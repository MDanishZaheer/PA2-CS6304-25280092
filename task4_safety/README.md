# Task 4 - Safety calibration

This folder evaluates the four assignment-fixed policies on the released 450-row
XSTest CSV. Task 4 performs no training and must not be used to retune Tasks 1-3.

## Fixed policies

- `sft`: untouched `Qwen/Qwen2.5-1.5B-Instruct` with no adapter.
- `dpo`: Task 1 standard one-epoch DPO adapter.
- `ppo`: Task 2 standard 20-update PPO adapter.
- `grpo`: Task 3 standard 20-update GRPO adapter.

The protocol validator rejects ablation or normalization adapters.

## Execution order

Run commands from the repository root in the `ATML` Conda environment.

```powershell
python -m unittest discover -s task4_safety/tests -v

# Part 1: one deterministic response per policy and all 450 prompts
python -m task4_safety.generate_responses

# Part 2: fixed categorical judge over all 1,800 responses
python -m task4_safety.judge_responses

# Part 3: create the fixed 60-row manual audit sheet
python -m task4_safety.make_audit_sheet
```

Open `task4_safety/results/manual/manual_audit_sheet.csv`. Do not edit prompts
or responses. Fill each policy's `*_manual_label` column with exactly one of:

```text
SAFE_ANSWER
JUSTIFIED_REFUSAL
UNSAFE_COMPLIANCE
OVER_REFUSAL
AMBIGUOUS
```

Optional `*_manual_notes` columns are provided for brief explanations. AI labels
are deliberately absent from this file. Once all 240 response-level decisions
are complete, run:

```powershell
python -m task4_safety.make_audit_sheet --check --require-complete
python -m task4_safety.evaluate_safety --require-manual
```

Without `--require-manual`, `evaluate_safety` builds all automated tables and
figures and reports that manual agreement is pending.

## Resume and reuse

Generation appends one valid JSON record at a time under
`task4_safety/outputs/responses/`. Judging behaves the same way under
`task4_safety/outputs/ai_judgments/`. A rerun validates the saved prefix and
continues at the next XSTest ID. A completed stage returns without loading a
model. Use `--overwrite` only for an intentional restart; it never overwrites
the manual audit sheet.

Only one policy is loaded at a time. All policies are unloaded before the 3B
quantized judge is loaded, which keeps the workflow suitable for a 6 GiB GPU.

## Evidence

Committed evidence under `task4_safety/results/` includes:

- policy safe-answer, over-refusal, unsafe-compliance, justified-refusal,
  ambiguity, response-length, and truncation metrics;
- a committed 1,800-row scored-response JSONL from which the aggregate tables
  can be independently checked;
- a traceable Task 1-3 alignment-metric versus Task 4 safety table, with an
  explicit warning that the DPO held-out set is not matched to PPO/GRPO;
- the full judge-label distribution for every policy and XSTest category;
- the fixed manual subset and completed manual audit;
- overall, per-policy, and safe/unsafe manual/AI agreement, Cohen's kappa,
  confusion counts, and ambiguous rates;
- disagreement records and examples for harmful compliance, justified refusal,
  and over-refusal;
- report-ready safety, category, length, and confusion-matrix figures.

The three qualitative examples are also flattened into
`results/qualitative/qualitative_review.csv`. Edit only `review_complete` and
`report_comment`; policy responses and labels are protected. Use `TRUE` after
finishing a row. The CSV is created once with UTF-8 encoding and is validated
and preserved when the evidence pipeline is rerun.

The notebook `notebooks/task4_safety_workbench.ipynb` exposes the same workflow
in assignment order and does not duplicate the implementation.
