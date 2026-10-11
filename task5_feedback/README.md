# Task 5 - Feedback source: RLVR versus RLAIF

Task 5 performs frozen-policy evaluation only. It does not train or modify the
SFT, supplied RLVR, or supplied RLAIF policy. The two aligned adapters remain
under `checkpoints/` and are loaded with `trainable=False`.

## Fixed comparison

- Policies: untouched SFT, supplied RLVR adapter, supplied RLAIF adapter.
- In-domain set: all 300 released GSM8K evaluation prompts.
- Transfer set: the fixed first 100 official SVAMP challenge examples supplied
  in `data/math_transfer_eval.jsonl`.
- Diagnostics: 20 problems with the five manually validated response variants.
- Generation: greedy decoding, a common 256-token prompt cap, and a common
  512-token completion cap.
- AI feedback: the supplied deterministic Qwen2.5-3B pairwise judge and rubric.
- Verifiable feedback: the supplied designated-final exact numeric verifier.

The numerical scale of binary exact reward and normalized AI pairwise reward is
not treated as a calibrated utility comparison.

## Run order

Run from the repository root in the `ATML` Conda environment.

```powershell
python -m unittest discover -s task5_feedback/tests -v

# Part 1 - in-domain GSM8K
python -m task5_feedback.evaluate_math --dataset gsm

# Part 2 - controlled feedback diagnostics
python -m task5_feedback.score_perturbations

# Part 3 - out-of-domain SVAMP
python -m task5_feedback.evaluate_math --dataset transfer

# Evidence for all required comparisons and research questions
python -m task5_feedback.compare_feedback
```

Each math evaluation can also be separated safely:

```powershell
python -m task5_feedback.evaluate_math --dataset gsm --stage generate
python -m task5_feedback.evaluate_math --dataset gsm --stage judge
python -m task5_feedback.evaluate_math --dataset gsm --stage summarize
```

## Resume behavior

Policy responses are appended under `task5_feedback/outputs/<dataset>/responses/`.
Pairwise judgments and the shared judge cache are appended under
`task5_feedback/outputs/`. Existing records must match the fixed prompt,
policy, response hashes, rubric, and comparison order. A completed stage returns
without loading its model. Keep `--overwrite` off during normal execution.

An interrupted generation may repeat only its current small batch. It never
restarts training because Task 5 contains no training.

## Controlled diagnostic interpretation

The release contains five response variants but four unambiguous
clean-versus-perturbation comparisons. Therefore:

- `variant_reward_scores.csv` reports both mechanisms for all five variants;
- `five_variant_mechanism_metrics.csv` reports all five categories, with the
  clean category aggregated over the four comparisons;
- `controlled_pair_metrics.csv` reports better/tie/wrong rates for reasoning
  corruption, wrong final answer, persuasive filler, and gold distractor;
- `sensitivity_summary.csv` reports S_reason, S_outcome, filler susceptibility,
  and gold-distractor robustness.

This avoids inventing a total quality order between two differently flawed
responses that the assignment does not define.

## Evidence produced

Committed files under `task5_feedback/results/` include:

- GSM8K and SVAMP policy metrics: exact accuracy, AI pairwise score, format
  compliance, mean and dispersion of response length, and verifier-judge
  agreement;
- all three policy-pair comparisons, including RLVR versus RLAIF;
- SVAMP category behavior and in-domain-to-transfer drops;
- all five diagnostic variant scores and four controlled-pair outcome tables;
- qualitative examples for reasoning corruption, persuasive filler, wrong
  final answers, and gold distractors;
- feedback coverage, noise, exploitability, and recorded inference cost;
- machine-readable and Markdown evidence for all four research questions;
- report-ready figures and complete response-level records.

The notebook `notebooks/task5_feedback_workbench.ipynb` exposes the same stages
without duplicating their implementation.
