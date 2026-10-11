# Task 1 saved evidence

The experiment scripts populate these directories with JSON, JSONL, CSV, and plots.
Keep the small machine-readable files in Git; `../outputs/` contains ignored LoRA
weights. A run's `train_indices.json`, `run_config.json`, and adapter SHA-256 in
`train_summary.json` identify the exact training source and checkpoint. Inspect
`qualitative_candidates.jsonl` manually before using any response as a quality example.

`reward_scoring_protocol.json` records the fixed reward model, tokenizer, maximum input
length, and one-item scoring batch. `reward_score_cache.jsonl` stores resumable scalar
scores keyed by a hash of the complete prompt-response input. These files support the
corrected reward means without repeating training or response generation.
