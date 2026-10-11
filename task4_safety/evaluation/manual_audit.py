# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Create and validate the fixed 60-row manual audit without exposing AI labels."""

from __future__ import annotations

import pandas as pd
import numpy as np

from common.data import read_jsonl
from common.logging_utils import load_json, save_json
from task4_safety.core.paths import POLICY_ORDER, manual_directory, response_path
from task4_safety.core.protocol import (
    BEHAVIOR_LABELS,
    file_sha256,
    load_xstest,
    validate_config,
)
from task4_safety.core.records import validate_complete_responses


def fixed_audit_ids(xstest: pd.DataFrame, per_class: int, seed: int) -> list[int]:
    """Select one deterministic set of prompt IDs, balanced SAFE/UNSAFE."""
    rng = np.random.default_rng(seed)
    selected = []
    for benchmark_class in ("SAFE", "UNSAFE"):
        pool = xstest.loc[
            xstest["benchmark_class"] == benchmark_class, "xstest_id"
        ].to_numpy(dtype=int)
        if len(pool) < per_class:
            raise ValueError(f"Not enough {benchmark_class} rows for manual audit.")
        selected.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(value) for value in selected)


def _response_maps(config: dict) -> tuple[dict, dict]:
    xstest = load_xstest(config)
    maps, hashes = {}, {}
    for policy_name in POLICY_ORDER:
        path = response_path(config, policy_name)
        if not path.exists():
            raise FileNotFoundError(f"Generate {policy_name} responses first: {path}")
        rows = read_jsonl(path)
        validate_complete_responses(rows, xstest, policy_name)
        maps[policy_name] = {int(row["xstest_id"]): row for row in rows}
        hashes[policy_name] = file_sha256(path)
    return maps, hashes


def audit_sheet_path(config: dict):
    return manual_directory(config) / "manual_audit_sheet.csv"


def _expected_sheet(config: dict) -> tuple[pd.DataFrame, dict]:
    xstest = load_xstest(config)
    selected = fixed_audit_ids(
        xstest,
        int(config["manual_audit_per_class"]),
        int(config["seed"]),
    )
    response_maps, response_hashes = _response_maps(config)
    metadata = xstest.set_index("xstest_id")
    records = []
    for xstest_id in selected:
        source = metadata.loc[xstest_id]
        record = {
            "xstest_id": xstest_id,
            "benchmark_class": source["benchmark_class"],
            "type": source["type"],
            "focus": "" if pd.isna(source["focus"]) else source["focus"],
            "prompt": source["prompt"],
        }
        for policy_name in POLICY_ORDER:
            response = response_maps[policy_name][xstest_id]
            record[f"{policy_name}_response"] = response["response"]
            record[f"{policy_name}_response_tokens"] = response["response_tokens"]
            record[f"{policy_name}_manual_label"] = ""
            record[f"{policy_name}_manual_notes"] = ""
        records.append(record)
    protocol = {
        "seed": int(config["seed"]),
        "per_benchmark_class": int(config["manual_audit_per_class"]),
        "prompt_count": len(selected),
        "safe_prompt_count": 30,
        "unsafe_prompt_count": 30,
        "xstest_ids": selected,
        "policies_per_prompt": list(POLICY_ORDER),
        "manual_decisions_required": len(selected) * len(POLICY_ORDER),
        "response_file_sha256": response_hashes,
        "ai_labels_excluded_from_sheet": True,
    }
    return pd.DataFrame(records), protocol


def _normalized_label(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def validate_manual_sheet(config: dict, require_complete: bool = False) -> dict:
    """Validate prompt/response immutability and return completion counts."""
    expected, expected_protocol = _expected_sheet(config)
    path = audit_sheet_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Create the manual audit sheet first: {path}")
    actual = pd.read_csv(path, keep_default_na=False)
    if list(actual.columns) != list(expected.columns):
        raise ValueError("Manual audit columns were changed.")
    if len(actual) != 60:
        raise ValueError(f"Manual audit must contain exactly 60 rows; found {len(actual)}.")
    immutable = [
        "xstest_id", "benchmark_class", "type", "focus", "prompt",
        *[
            field
            for policy_name in POLICY_ORDER
            for field in (
                f"{policy_name}_response",
                f"{policy_name}_response_tokens",
            )
        ],
    ]
    for column in immutable:
        if actual[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError(f"Manual audit immutable column changed: {column}")
    filled = 0
    invalid = []
    for row_index, row in actual.iterrows():
        for policy_name in POLICY_ORDER:
            column = f"{policy_name}_manual_label"
            label = _normalized_label(row[column])
            if label:
                filled += 1
                if label not in BEHAVIOR_LABELS:
                    invalid.append((int(row_index), policy_name, label))
    if invalid:
        raise ValueError(f"Invalid manual behavior labels: {invalid[:8]}")
    required = len(actual) * len(POLICY_ORDER)
    if require_complete and filled != required:
        raise ValueError(
            f"Manual audit is incomplete: {filled}/{required} response labels filled."
        )
    protocol_path = manual_directory(config) / "manual_audit_protocol.json"
    if protocol_path.exists() and load_json(protocol_path) != expected_protocol:
        raise RuntimeError("Saved manual-audit protocol no longer matches responses.")
    return {
        "prompt_rows": len(actual),
        "required_response_labels": required,
        "filled_response_labels": filled,
        "remaining_response_labels": required - filled,
        "complete": filled == required,
    }


def create_manual_audit_sheet(config: dict) -> pd.DataFrame:
    """Create the sheet once; never overwrite existing manual work."""
    validate_config(config)
    expected, protocol = _expected_sheet(config)
    path = audit_sheet_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        status = validate_manual_sheet(config, require_complete=False)
        print(
            "Using existing manual audit sheet: "
            f"{status['filled_response_labels']}/{status['required_response_labels']} "
            "labels completed."
        )
        return pd.read_csv(path, keep_default_na=False)
    expected.to_csv(path, index=False)
    pd.DataFrame({"xstest_id": protocol["xstest_ids"]}).to_csv(
        path.parent / "audit_subset_ids.csv", index=False
    )
    save_json(path.parent / "manual_audit_protocol.json", protocol)
    save_json(path.parent / "manual_completion.json", {
        "prompt_rows": 60,
        "required_response_labels": 240,
        "filled_response_labels": 0,
        "remaining_response_labels": 240,
        "complete": False,
    })
    print(f"Created manual audit sheet: {path}")
    print("Fill only the four *_manual_label columns and optional notes columns.")
    return expected


def manual_sheet_long(config: dict, require_complete: bool = True) -> pd.DataFrame:
    """Reshape the 60-row sheet into response-level records for comparison."""
    validate_manual_sheet(config, require_complete=require_complete)
    wide = pd.read_csv(audit_sheet_path(config), keep_default_na=False)
    records = []
    for _, row in wide.iterrows():
        for policy_name in POLICY_ORDER:
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "benchmark_class": row["benchmark_class"],
                "type": row["type"],
                "prompt": row["prompt"],
                "policy": policy_name,
                "response": row[f"{policy_name}_response"],
                "response_tokens": int(row[f"{policy_name}_response_tokens"]),
                "manual_label": _normalized_label(
                    row[f"{policy_name}_manual_label"]
                ),
                "manual_notes": row[f"{policy_name}_manual_notes"],
            })
    return pd.DataFrame(records)
