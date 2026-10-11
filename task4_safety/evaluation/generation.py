# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Generate deterministic XSTest responses with safe per-batch resume."""

from __future__ import annotations

import pandas as pd

from common.data import read_jsonl
from common.generation import batch_generate
from common.logging_utils import append_jsonl, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from task4_safety.core.paths import POLICY_ORDER, response_path
from task4_safety.core.protocol import (
    generation_protocol,
    generation_protocol_path,
    load_xstest,
    policy_adapter,
    protect_protocol,
    save_task4_manifest,
    validate_config,
)
from task4_safety.core.records import (
    validate_complete_responses,
    validate_response_prefix,
)


def _optional_text(value) -> str:
    return "" if pd.isna(value) else str(value)


def generate_policy_responses(
    config: dict,
    policy_name: str,
    batch_size: int | None = None,
    overwrite: bool = False,
) -> list[dict]:
    """Generate, resume, or reuse all 450 responses for one policy."""
    validate_config(config)
    xstest = load_xstest(config)
    destination = response_path(config, policy_name)
    protocol_path = generation_protocol_path(config, policy_name)
    expected_protocol = generation_protocol(config, policy_name)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if overwrite:
        if destination.exists():
            destination.unlink()
        protect_protocol(protocol_path, expected_protocol, overwrite=True)
    else:
        protect_protocol(protocol_path, expected_protocol)

    existing = read_jsonl(destination) if destination.exists() else []
    validate_response_prefix(existing, xstest, policy_name)
    if len(existing) == len(xstest):
        print(f"Using completed Task 4 responses for {policy_name}.")
        return existing

    batch_size = int(batch_size or config["task4_generation_batch_size"])
    if batch_size < 1:
        raise ValueError("Generation batch size must be positive.")
    set_seed(int(config["seed"]))
    tokenizer = load_tokenizer(config["base_model"], padding_side="left")
    model = load_policy(
        config,
        adapter_path=policy_adapter(config, policy_name),
        trainable=False,
    )
    try:
        for start in range(len(existing), len(xstest), batch_size):
            chunk = xstest.iloc[start:start + batch_size]
            prompts = [
                [{"role": "user", "content": str(prompt)}]
                for prompt in chunk["prompt"].tolist()
            ]
            generated = batch_generate(
                model,
                tokenizer,
                prompts,
                max_prompt_length=int(config["task4_max_prompt_length"]),
                max_new_tokens=int(config["safety_max_new_tokens"]),
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            for (_, source), response, length, terminated, truncated in zip(
                chunk.iterrows(),
                generated["responses"],
                generated["response_lengths"],
                generated["terminated_with_eos"],
                generated["truncated"],
            ):
                record = {
                    "xstest_id": int(source["xstest_id"]),
                    "policy": policy_name,
                    "prompt": str(source["prompt"]),
                    "benchmark_class": str(source["benchmark_class"]),
                    "type": str(source["type"]),
                    "focus": _optional_text(source["focus"]),
                    "note": _optional_text(source["note"]),
                    "response": str(response),
                    "response_tokens": int(length),
                    "terminated_with_eos": bool(terminated),
                    "truncated": bool(truncated),
                }
                append_jsonl(destination, record)
            completed = min(start + len(chunk), len(xstest))
            print(f"{policy_name}: generated {completed}/{len(xstest)} responses")
    finally:
        del model
        del tokenizer
        clear_gpu()

    rows = read_jsonl(destination)
    validate_complete_responses(rows, xstest, policy_name)
    return rows


def generate_all_policies(
    config: dict,
    policy_names=POLICY_ORDER,
    batch_size: int | None = None,
    overwrite: bool = False,
) -> dict[str, list[dict]]:
    """Generate policies sequentially so only one 1.5B model occupies the GPU."""
    save_task4_manifest(config)
    return {
        name: generate_policy_responses(
            config,
            name,
            batch_size=batch_size,
            overwrite=overwrite,
        )
        for name in policy_names
    }
