# Task 4 results

This committed directory is populated from cached Task 4 inference. The raw
policy responses and AI-judge outputs live under `task4_safety/outputs/` and
are ignored by Git. Do not edit the generated tables directly.

The only user-edited evidence file is
`manual/manual_audit_sheet.csv`. Keep all prompts and responses unchanged;
fill only the four `*_manual_label` columns and optional notes columns.

`summary/cross_task_alignment_safety.csv` joins the saved standard Task 1-3
alignment metrics to Task 4 safety rates for Research Question 1. Its scope
column records that DPO used a different held-out set from PPO/GRPO.

After evidence aggregation, `raw/scored_responses.jsonl` contains all 1,800
prompt-policy responses, judge labels, confidence fields, and raw judge text.
The ignored `outputs/` copies are working caches used to resume inference.

For the three qualitative examples, edit only the `review_complete` and
`report_comment` columns in `qualitative/qualitative_review.csv`. Set the flag
to `TRUE` only after adding a short interpretation. Rerunning Task 4 preserves
these two columns and rejects changes to the prompts, responses, or labels.
