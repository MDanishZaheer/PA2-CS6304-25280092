# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build every available Task 4 table from cached response-level evidence."""

from __future__ import annotations

from common.logging_utils import save_json
from task4_safety.analysis.cross_task import build_cross_task_table
from task4_safety.core.paths import results_root
from task4_safety.evaluation.agreement import build_manual_agreement
from task4_safety.evaluation.manual_audit import audit_sheet_path
from task4_safety.evaluation.qualitative import select_qualitative_examples
from task4_safety.evaluation.safety_metrics import build_safety_tables


def build_available_evidence(config: dict, require_manual: bool = False) -> dict:
    """Build automated evidence and manual evidence when annotation is complete."""
    tables = build_safety_tables(config)
    cross_task = build_cross_task_table(config, tables["policy_metrics"])
    agreement = None
    if audit_sheet_path(config).exists():
        agreement = build_manual_agreement(
            config, require_complete=require_manual
        )
    elif require_manual:
        raise FileNotFoundError(
            f"Manual audit sheet not found: {audit_sheet_path(config)}"
        )
    manual_comparison = agreement["comparison"] if agreement else None
    examples = select_qualitative_examples(config, manual_comparison)
    summary = {
        "automated_policy_rows": len(tables["policy_metrics"]),
        "category_rows": len(tables["category_distribution"]),
        "manual_audit_created": audit_sheet_path(config).exists(),
        "manual_agreement_available": agreement is not None,
        "qualitative_examples": len(examples),
        "qualitative_reviews_complete": sum(
            bool(example.get("review_complete")) for example in examples
        ),
        "cross_task_rows": len(cross_task),
    }
    save_json(results_root(config) / "evidence_manifest.json", summary)
    return {
        **tables,
        "cross_task": cross_task,
        "manual_agreement": agreement,
        "qualitative": examples,
    }
