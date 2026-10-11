# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Evaluate reference-relative DPO preference fit on fixed held-out pairs."""

from __future__ import annotations

import torch

from common.models import reference_mode
from task1_dpo.core.batching import make_preference_collator
from task1_dpo.core.log_probs import response_sequence_log_probs
from task1_dpo.dpo import dpo_loss


@torch.inference_mode()
def evaluate_preference_pairs(policy, tokenizer, rows: list[dict], config: dict, beta: float):
    """Return aggregate DPO loss, accuracy, and per-pair margins."""
    collate = make_preference_collator(tokenizer, int(config["max_sequence_length"]))
    device = next(policy.parameters()).device
    records = []
    total_loss = 0.0
    for start in range(0, len(rows), int(config["batch_size"])):
        selected = rows[start : start + int(config["batch_size"])]
        chosen, rejected = collate(selected)
        chosen = {key: value.to(device) for key, value in chosen.items()}
        rejected = {key: value.to(device) for key, value in rejected.items()}
        with reference_mode(policy):
            ref_chosen = response_sequence_log_probs(policy, chosen)
            ref_rejected = response_sequence_log_probs(policy, rejected)
        policy_chosen = response_sequence_log_probs(policy, chosen)
        policy_rejected = response_sequence_log_probs(policy, rejected)
        loss, _ = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta
        )
        total_loss += float(loss) * len(selected)
        margins = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
        for row, margin, pc, pr, rc, rr in zip(
            selected, margins, policy_chosen, policy_rejected, ref_chosen, ref_rejected
        ):
            records.append({
                "prompt_id": row["prompt_id"],
                "source_index": row.get("source_index"),
                "length_stratum": row.get("length_stratum"),
                "preference_margin": float(margin),
                "preference_correct": bool(margin > 0),
                "policy_chosen_logp": float(pc),
                "policy_rejected_logp": float(pr),
                "reference_chosen_logp": float(rc),
                "reference_rejected_logp": float(rr),
            })
    if not records:
        raise ValueError("The preference evaluation dataset is empty.")
    summary = {
        "heldout_dpo_loss": total_loss / len(records),
        "preference_accuracy": sum(row["preference_correct"] for row in records) / len(records),
        "preference_pairs": len(records),
    }
    return summary, records
