# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Generate resumable deterministic GSM8K and SVAMP policy responses."""

from __future__ import annotations

from common.data import read_jsonl
from common.generation import batch_generate
from common.logging_utils import append_jsonl, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.core.paths import POLICY_ORDER, response_path
from task5_feedback.core.protocol import (
    generation_protocol,
    load_math_rows,
    policy_adapter,
    protect_protocol,
    row_id,
    save_task5_manifest,
    validate_config,
)
from task5_feedback.core.records import (
    validate_complete_responses,
    validate_response_prefix,
)
from task5_feedback.core.paths import protocols_directory
from task5_feedback.rlvr import exact_reward, extract_designated_final


def generate_policy_responses(
    config: dict,
    dataset: str,
    policy_name: str,
    batch_size: int | None = None,
    overwrite: bool = False,
) -> list[dict]:
    """Generate, resume, or reuse one frozen policy/dataset condition."""
    validate_config(config)
    rows = load_math_rows(config, dataset)
    destination = response_path(config, dataset, policy_name)
    protocol_path = (
        protocols_directory(config) / f"generation_{dataset}_{policy_name}.json"
    )
    expected_protocol = generation_protocol(config, dataset, policy_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if destination.exists():
            destination.unlink()
        protect_protocol(protocol_path, expected_protocol, overwrite=True)
    else:
        protect_protocol(protocol_path, expected_protocol)

    existing = read_jsonl(destination) if destination.exists() else []
    validate_response_prefix(existing, rows, dataset, policy_name)
    if len(existing) == len(rows):
        print(f"Using completed {dataset}/{policy_name} responses.")
        return existing

    selected_batch_size = int(
        batch_size or config["task5_generation_batch_size"]
    )
    if selected_batch_size < 1:
        raise ValueError("Task 5 generation batch size must be positive.")
    set_seed(int(config["seed"]))
    tokenizer = load_tokenizer(config["base_model"], padding_side="left")
    model = load_policy(
        config,
        adapter_path=policy_adapter(config, policy_name),
        trainable=False,
    )
    try:
        for start in range(len(existing), len(rows), selected_batch_size):
            chunk = rows[start:start + selected_batch_size]
            generated = batch_generate(
                model,
                tokenizer,
                [row["messages"] for row in chunk],
                max_prompt_length=int(config["task5_max_prompt_length"]),
                max_new_tokens=int(config["math_max_new_tokens"]),
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            values = zip(
                chunk,
                generated["responses"],
                generated["response_lengths"],
                generated["terminated_with_eos"],
                generated["truncated"],
            )
            for source, response, length, terminated, truncated in values:
                predicted = extract_designated_final(response)
                append_jsonl(destination, {
                    "dataset": dataset,
                    "prompt_id": row_id(source, dataset),
                    "source_index": source["source_index"],
                    "policy": policy_name,
                    "question": str(source["question"]),
                    "messages": source["messages"],
                    "gold_final": str(source["gold_final"]),
                    "response": response,
                    "response_tokens": int(length),
                    "terminated_with_eos": bool(terminated),
                    "truncated": bool(truncated),
                    "predicted_final": predicted,
                    "format_compliant": predicted is not None,
                    "exact_reward": exact_reward(
                        response, str(source["gold_final"])
                    ),
                    "category": (
                        str(source.get("svamp_type", ""))
                        if dataset == "transfer"
                        else "GSM8K"
                    ),
                })
            completed = min(start + len(chunk), len(rows))
            print(
                f"{dataset}/{policy_name}: generated {completed}/{len(rows)}",
                flush=True,
            )
    finally:
        clear_gpu(model, tokenizer)
        del model, tokenizer

    records = read_jsonl(destination)
    validate_complete_responses(records, rows, dataset, policy_name)
    return records


def generate_dataset_responses(
    config: dict,
    dataset: str,
    policy_names=POLICY_ORDER,
    batch_size: int | None = None,
    overwrite: bool = False,
) -> dict[str, list[dict]]:
    """Generate all policies sequentially so only one occupies the GPU."""
    save_task5_manifest(config)
    return {
        policy_name: generate_policy_responses(
            config,
            dataset,
            policy_name,
            batch_size=batch_size,
            overwrite=overwrite,
        )
        for policy_name in policy_names
    }
