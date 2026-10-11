# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Small synthetic Task 4 artifacts shared by tests."""

from __future__ import annotations

from common.data import load_yaml, read_jsonl, write_jsonl
from task4_safety.core.paths import POLICY_ORDER, judgment_path, response_path
from task4_safety.core.protocol import load_xstest


def temporary_config(root):
    config = load_yaml("configs/feedback.yaml")
    config["task4_output_dir"] = str(root / "outputs")
    config["task4_results_dir"] = str(root / "results")
    return config


def write_synthetic_responses(config):
    xstest = load_xstest(config)
    for policy_name in POLICY_ORDER:
        rows = []
        for _, source in xstest.iterrows():
            rows.append({
                "xstest_id": int(source["xstest_id"]),
                "policy": policy_name,
                "prompt": str(source["prompt"]),
                "benchmark_class": str(source["benchmark_class"]),
                "type": str(source["type"]),
                "focus": "",
                "note": "",
                "response": f"{policy_name} response {int(source['xstest_id'])}",
                "response_tokens": 8,
                "terminated_with_eos": True,
                "truncated": False,
            })
        write_jsonl(response_path(config, policy_name), rows)


def write_synthetic_judgments(config):
    for policy_name in POLICY_ORDER:
        responses = read_jsonl(response_path(config, policy_name))
        judgments = []
        for source in responses:
            label = (
                "SAFE_ANSWER"
                if source["benchmark_class"] == "SAFE"
                else "JUSTIFIED_REFUSAL"
            )
            judgments.append({
                **source,
                "ai_label": label,
                "ai_confidence": 0.9,
                "ai_rationale_tag": "synthetic",
                "raw_judge_output": "{}",
            })
        write_jsonl(judgment_path(config, policy_name), judgments)
