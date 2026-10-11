# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Generate fixed held-out GRPO responses with resumable per-prompt logging."""

from __future__ import annotations

import torch

from common.data import prompt_messages
from common.generation import batch_generate, response_token_logprobs
from common.logging_utils import append_jsonl, set_seed
from common.models import reference_mode
from task3_grpo.core.batching import token_entropy


def generate_evaluation_records(
    policy,
    tokenizer,
    rows: list[dict],
    config: dict,
    raw_path,
    existing_records: list[dict] | None = None,
) -> list[dict]:
    """Generate only missing prompts while preserving the fixed prompt order."""
    records = list(existing_records or [])
    expected_prefix = [row["prompt_id"] for row in rows[: len(records)]]
    if [row.get("prompt_id") for row in records] != expected_prefix:
        raise ValueError("Partial Task 3 evaluation does not match the prompt order.")
    for index in range(len(records), len(rows)):
        row = rows[index]
        messages = prompt_messages(row)
        set_seed(int(config["seed"]) + index)
        generated = batch_generate(
            policy,
            tokenizer,
            [messages],
            max_prompt_length=int(config["max_prompt_length"]),
            max_new_tokens=int(config["max_completion_length"]),
            temperature=float(config["generation"]["temperature"]),
            top_p=float(config["generation"]["top_p"]),
            do_sample=bool(config["generation"]["do_sample"]),
        )
        policy.eval()
        with torch.no_grad():
            policy_logp, logits = response_token_logprobs(
                policy,
                generated["sequences"],
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
            entropy = token_entropy(logits, generated["response_mask"])
            del logits
            with reference_mode(policy):
                reference_logp, reference_logits = response_token_logprobs(
                    policy,
                    generated["sequences"],
                    generated["attention_mask"],
                    generated["prompt_width"],
                    generated["response_ids"],
                )
                del reference_logits
        mask = generated["response_mask"]
        length = int(mask.sum())
        sampled_kl_sum = float(((policy_logp - reference_logp) * mask).sum())
        record = {
            "prompt_id": row["prompt_id"],
            "source_index": row.get("source_index"),
            "messages": messages,
            "prompt": messages[-1].get("content", row.get("prompt", "")),
            "response": generated["responses"][0],
            "response_length": length,
            "terminated_with_eos": bool(generated["terminated_with_eos"][0]),
            "truncated": bool(generated["truncated"][0]),
            "policy_logp_sequence_sum": float((policy_logp * mask).sum()),
            "reference_logp_sequence_sum": float((reference_logp * mask).sum()),
            "sampled_kl_sequence_sum": sampled_kl_sum,
            "sampled_kl_token_mean": sampled_kl_sum / max(length, 1),
            "entropy_sequence_sum": float(entropy) * length,
            "entropy_token_mean": float(entropy),
        }
        append_jsonl(raw_path, record)
        records.append(record)
        print(f"Held-out generation {index + 1}/{len(rows)}", flush=True)
    return records
