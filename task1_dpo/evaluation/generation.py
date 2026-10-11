# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Generate paired policy and reference responses with fixed prompt seeds."""

from __future__ import annotations

import torch

from common.data import prompt_messages, prompt_messages_from_preference
from common.generation import batch_generate, response_token_logprobs
from common.metrics import sampled_kl
from common.models import reference_mode


def _messages(row: dict) -> list[dict]:
    """Select the course prompt representation for a data row."""
    return (
        prompt_messages_from_preference(row)
        if "chosen" in row and "rejected" in row
        else prompt_messages(row)
    )


@torch.inference_mode()
def generate_comparison_records(policy, tokenizer, rows: list[dict], config: dict):
    """Generate policy/reference pairs and compute sampled-response KL."""
    records = []
    total_log_ratio = 0.0
    total_response_tokens = 0.0
    generation = config["generation"]
    for index, row in enumerate(rows):
        messages = _messages(row)
        seed = int(config["seed"]) + index
        arguments = dict(
            max_prompt_length=int(config["max_sequence_length"]),
            max_new_tokens=int(config["max_generation_tokens"]),
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            do_sample=bool(generation["do_sample"]),
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        policy_output = batch_generate(policy, tokenizer, [messages], **arguments)
        policy_logp, _ = response_token_logprobs(
            policy,
            policy_output["sequences"],
            policy_output["attention_mask"],
            policy_output["prompt_width"],
            policy_output["response_ids"],
        )
        with reference_mode(policy):
            reference_logp, _ = response_token_logprobs(
                policy,
                policy_output["sequences"],
                policy_output["attention_mask"],
                policy_output["prompt_width"],
                policy_output["response_ids"],
            )
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            reference_output = batch_generate(policy, tokenizer, [messages], **arguments)
        response_mask = policy_output["response_mask"]
        token_sum = float(((policy_logp - reference_logp) * response_mask).sum())
        token_count = float(response_mask.sum())
        total_log_ratio += token_sum
        total_response_tokens += token_count
        record = {
            "prompt_id": row["prompt_id"],
            "source_index": row.get("source_index"),
            "prompt": row.get("prompt", messages[-1].get("content", "")),
            "messages": messages,
            "policy_response": policy_output["responses"][0],
            "reference_response": reference_output["responses"][0],
            "response_length": policy_output["response_lengths"][0],
            "reference_length": reference_output["response_lengths"][0],
            "policy_truncated": policy_output["truncated"][0],
            "reference_truncated": reference_output["truncated"][0],
            "sampled_kl_sequence_sum": token_sum,
            "sampled_kl_token_mean": float(sampled_kl(policy_logp, reference_logp, response_mask)),
        }
        records.append(record)
    return records, total_log_ratio / max(total_response_tokens, 1.0)
