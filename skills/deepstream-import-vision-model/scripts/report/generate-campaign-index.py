#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Index completed per-case Step 8 reports without converting missing cases to PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
FUNCTIONAL = {"PASS", "FAIL", "BLOCKED", "UNTESTED", "ADAPTER_REQUIRED"}
CAPACITY = {"PASS", "FAIL", "BLOCKED", "UNTESTED"}
PIPELINE_VISUAL = {"PASS", "FAIL", "BLOCKED", "UNTESTED"}
VISION13_CASES = Path(__file__).resolve().parents[2] / "assets" / "vision13-cases.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_index(campaign_path: Path) -> dict:
    campaign_path = campaign_path.resolve()
    data = json.loads(campaign_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("campaign must contain a JSON object")
    cases = data.get("cases")
    if (data.get("schema_version") != 1 or not isinstance(cases, list) or not cases
            or any(not isinstance(case, str) or not SAFE_NAME.fullmatch(case) for case in cases)
            or len(cases) != len(set(cases))):
        raise ValueError("campaign must have schema_version=1 and unique case IDs")
    suite = data.get("suite")
    expected_tasks = {}
    if suite is not None:
        if suite not in {"vision13", "vision14"}:
            raise ValueError(f"unsupported campaign suite: {suite}")
        expected = json.loads(VISION13_CASES.read_text(encoding="utf-8"))
        expected_cases = expected["cases"] + ([expected["vision14_additional_case"]]
                                              if suite == "vision14" else [])
        expected_tasks = {item["case_id"]: item["task"] for item in expected_cases}
        if len(cases) != len(expected_tasks) or set(cases) != set(expected_tasks):
            raise ValueError(f"{suite} requires reports for exactly its {len(expected_tasks)} application cases")
    models_dir = campaign_path.parent
    rows = []
    for case in cases:
        result_path = models_dir / case / "reports" / "report-result.json"
        if not result_path.is_file():
            raise ValueError(f"report is absent for {case}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"report result must contain a JSON object for {case}")
        if (result.get("functional_status") not in FUNCTIONAL
                or result.get("capacity_status") not in CAPACITY):
            raise ValueError(f"invalid report status for {case}")
        pipeline_visual = result.get("pipeline_visual_result", "UNTESTED")
        if pipeline_visual not in PIPELINE_VISUAL:
            raise ValueError(f"invalid pipeline/visual result for {case}")
        task = result.get("task")
        expected_prefix = ("benchmark_report_" if task == "detection" and
                           result["functional_status"] == "PASS" else
                           "validation_report_" if result["functional_status"] == "PASS" else
                           "status_report_")
        pdf_value = result.get("pdf")
        if not isinstance(pdf_value, str) or not pdf_value.strip():
            raise ValueError(f"PDF path must be a nonempty string for {case}")
        pdf = Path(pdf_value)
        if pdf.name != f"{expected_prefix}{case}.pdf":
            raise ValueError(f"PDF name does not match its report status for {case}")
        if not pdf.is_absolute():
            pdf = models_dir / pdf
        if not pdf.is_file():
            # A report generated inside a container may record its /work path.
            # Resolve only the exact registered case filename in this campaign;
            # the digest and task/status checks below still bind its identity.
            pdf = models_dir / case / "reports" / pdf.name
        pdf = pdf.resolve()
        if not pdf.is_file() or pdf.stat().st_size == 0:
            raise ValueError(f"nonempty PDF is absent for {case}")
        try:
            relative_pdf = pdf.relative_to(models_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"PDF path escapes campaign directory for {case}") from exc
        digest = _sha256(pdf)
        if result.get("case_id") != case or result.get("pdf_sha256") != digest:
            raise ValueError(f"report identity or PDF digest differs for {case}")
        if expected_tasks and task != expected_tasks[case]:
            raise ValueError(f"vision13 task differs for {case}")
        rows.append({
            "case_id": case, "task": result.get("task"),
            "functional_status": result.get("functional_status"),
            "pipeline_visual_result": pipeline_visual,
            "capacity_status": result.get("capacity_status"),
            "pdf": str(relative_pdf), "pdf_sha256": digest,
        })
    payload = {"schema_version": 1, "campaign_id": data.get("campaign_id", ""),
               **({"suite": suite} if suite else {}), "cases": rows}
    (models_dir / "campaign-report-index.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# DeepStream model campaign", "",
             "| Case | Task | Pipeline + visual | Functional | Capacity | PDF |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['case_id']} | {row['task']} | {row['pipeline_visual_result']} | {row['functional_status']} | {row['capacity_status']} | [report]({row['pdf']}) |")
    (models_dir / "campaign-report-index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="models/campaign.json")
    args = parser.parse_args()
    try:
        payload = build_index(args.campaign)
    except (OSError, ValueError, KeyError) as exc:
        print(f"INDEX BLOCKED: {exc}", file=sys.stderr)
        return 4
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
