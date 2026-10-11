# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Select brief Task 4 examples without replacing human interpretation."""

from __future__ import annotations

import pandas as pd

from common.data import read_jsonl, write_jsonl
from task4_safety.core.paths import POLICY_ORDER, qualitative_directory
from task4_safety.evaluation.safety_metrics import load_all_judgments


TARGET_LABELS = (
    "UNSAFE_COMPLIANCE",
    "JUSTIFIED_REFUSAL",
    "OVER_REFUSAL",
)

EDITABLE_REVIEW_COLUMNS = ("review_complete", "report_comment")


def qualitative_review_path(config: dict):
    """Return the small, user-editable qualitative-review CSV."""
    return qualitative_directory(config) / "qualitative_review.csv"


def _review_table(examples: list[dict]) -> pd.DataFrame:
    """Flatten selected examples while keeping responses visibly immutable."""
    rows = []
    for example in examples:
        row = {
            "selection_target": example["selection_target"],
            "selection_basis": example["selection_basis"],
            "xstest_id": int(example["xstest_id"]),
            "benchmark_class": example["benchmark_class"],
            "type": example["type"],
            "prompt": example["prompt"],
            "observed_disagreement_source": example[
                "observed_disagreement_source"
            ],
            "covered_by_fixed_manual_audit": bool(
                example["covered_by_fixed_manual_audit"]
            ),
        }
        for policy_name in POLICY_ORDER:
            policy = example["policies"][policy_name]
            row.update({
                f"{policy_name}_response": policy["response"],
                f"{policy_name}_response_tokens": int(
                    policy["response_tokens"]
                ),
                f"{policy_name}_ai_label": policy["ai_label"],
                f"{policy_name}_manual_label": (
                    "" if policy["manual_label"] is None
                    else policy["manual_label"]
                ),
                f"{policy_name}_manual_notes": (
                    "" if policy["manual_notes"] is None
                    else policy["manual_notes"]
                ),
            })
        row["review_complete"] = False
        row["report_comment"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def _review_flag(value) -> bool:
    """Accept the ordinary TRUE/FALSE values written by spreadsheet tools."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().upper()
    if text == "TRUE":
        return True
    if text == "FALSE":
        return False
    raise ValueError(
        "qualitative_review.csv review_complete values must be TRUE or FALSE."
    )


def _migrate_json_comments(config: dict, expected: pd.DataFrame) -> None:
    """Copy any earlier JSONL comments into a newly created review sheet."""
    path = qualitative_directory(config) / "selected_examples.jsonl"
    if not path.exists():
        return
    previous = read_jsonl(path)
    comments = {
        (str(row.get("selection_target")), int(row.get("xstest_id"))): str(
            row.get("report_comment", "")
        ).strip()
        for row in previous
    }
    for index, row in expected.iterrows():
        key = (str(row["selection_target"]), int(row["xstest_id"]))
        comment = comments.get(key, "")
        if comment:
            expected.loc[index, "report_comment"] = comment
            expected.loc[index, "review_complete"] = True


def _read_review_csv(path) -> tuple[pd.DataFrame, bool]:
    """Read UTF-8 normally and safely accept an Excel Windows-1252 save."""
    try:
        table = pd.read_csv(
            path, encoding="utf-8-sig", keep_default_na=False
        )
        return table, False
    except UnicodeDecodeError:
        table = pd.read_csv(
            path, encoding="cp1252", keep_default_na=False
        )
        return table, True


def preserve_qualitative_review(
    config: dict, examples: list[dict]
) -> list[dict]:
    """Preserve editable review fields and restore fixed content on reruns."""
    expected = _review_table(examples)
    path = qualitative_review_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_encoding = False
    repaired_columns = []
    if path.exists():
        actual, normalize_encoding = _read_review_csv(path)
        if list(actual.columns) != list(expected.columns):
            raise ValueError("Qualitative review CSV columns were changed.")
        if len(actual) != len(expected):
            raise ValueError("Qualitative review CSV row count was changed.")
        immutable = [
            column for column in expected.columns
            if column not in EDITABLE_REVIEW_COLUMNS
        ]
        for column in immutable:
            if (
                actual[column].astype(str).tolist()
                != expected[column].astype(str).tolist()
            ):
                repaired_columns.append(column)
        expected["review_complete"] = [
            _review_flag(value) for value in actual["review_complete"]
        ]
        expected["report_comment"] = actual["report_comment"].astype(str)
    else:
        _migrate_json_comments(config, expected)
        expected.to_csv(path, index=False, encoding="utf-8-sig")

    if normalize_encoding or repaired_columns:
        expected.to_csv(path, index=False, encoding="utf-8-sig")
        if normalize_encoding:
            print(
                "Normalized qualitative_review.csv from Windows-1252 to UTF-8."
            )
        if repaired_columns:
            print(
                "Restored protected qualitative-review columns: "
                + ", ".join(repaired_columns)
            )

    for index, row in expected.iterrows():
        complete = bool(row["review_complete"])
        comment = str(row["report_comment"]).strip()
        if complete and not comment:
            raise ValueError(
                "A completed qualitative review requires a report_comment."
            )
        examples[index]["review_complete"] = complete
        examples[index]["report_comment"] = comment
    return examples


def select_qualitative_examples(config: dict, manual_comparison=None) -> list[dict]:
    """Prefer prompts with policy disagreement for each required behavior."""
    judgments = load_all_judgments(config)
    manual_lookup = {}
    audited_ids = set()
    if manual_comparison is not None:
        manual_lookup = {
            (int(row.xstest_id), row.policy): row
            for row in manual_comparison.itertuples()
        }
        audited_ids = {int(value) for value in manual_comparison["xstest_id"]}
    selected = []
    used_ids = set()
    for target in TARGET_LABELS:
        candidates = []
        for xstest_id, group in judgments.groupby("xstest_id", sort=True):
            labels = set(group["ai_label"])
            manual_labels_for_prompt = {
                manual_lookup[(int(xstest_id), policy_name)].manual_label
                for policy_name in POLICY_ORDER
                if (int(xstest_id), policy_name) in manual_lookup
            }
            if target not in labels and target not in manual_labels_for_prompt:
                continue
            manual_policy_difference = len(manual_labels_for_prompt) > 1
            manual_judge_error = any(
                manual_lookup[(int(xstest_id), row.policy)].manual_label
                != row.ai_label
                for row in group.itertuples()
                if (int(xstest_id), row.policy) in manual_lookup
            )
            if target in labels and target in manual_labels_for_prompt:
                selection_basis = "ai_and_manual_label"
            elif target in manual_labels_for_prompt:
                selection_basis = "manual_label_only"
            else:
                selection_basis = "ai_label_only"
            candidates.append((
                int(xstest_id) not in audited_ids,
                not (
                    manual_policy_difference
                    or manual_judge_error
                    or len(labels) > 1
                ),
                int(xstest_id),
                group,
                selection_basis,
            ))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        for _, _, xstest_id, group, selection_basis in candidates:
            if xstest_id in used_ids:
                continue
            used_ids.add(xstest_id)
            policy_records = {}
            manual_labels = []
            ai_errors = False
            for policy_name in POLICY_ORDER:
                row = group[group["policy"] == policy_name].iloc[0]
                record = {
                    "response": row["response"],
                    "response_tokens": int(row["response_tokens"]),
                    "ai_label": row["ai_label"],
                    "ai_confidence": float(row["ai_confidence"]),
                    "ai_rationale_tag": row["ai_rationale_tag"],
                    "manual_label": None,
                    "manual_notes": None,
                }
                manual = manual_lookup.get((xstest_id, policy_name))
                if manual is not None:
                    record["manual_label"] = manual.manual_label
                    record["manual_notes"] = manual.manual_notes
                    manual_labels.append(manual.manual_label)
                    ai_errors = ai_errors or manual.manual_label != row["ai_label"]
                policy_records[policy_name] = record
            if manual_labels:
                policy_difference = len(set(manual_labels)) > 1
                if policy_difference and ai_errors:
                    source = "both_policy_difference_and_judge_error"
                elif policy_difference:
                    source = "policy_difference"
                elif ai_errors:
                    source = "judge_error"
                else:
                    source = "no_manual_ai_disagreement"
            else:
                source = "manual_review_required"
            first = group.iloc[0]
            selected.append({
                "selection_target": target,
                "selection_basis": selection_basis,
                "xstest_id": xstest_id,
                "benchmark_class": first["benchmark_class"],
                "type": first["type"],
                "prompt": first["prompt"],
                "observed_disagreement_source": source,
                "covered_by_fixed_manual_audit": xstest_id in audited_ids,
                "policies": policy_records,
            })
            break
    selected = preserve_qualitative_review(config, selected)
    destination = qualitative_directory(config) / "selected_examples.jsonl"
    write_jsonl(destination, selected)
    return selected
