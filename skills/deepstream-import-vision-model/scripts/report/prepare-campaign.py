#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create fresh, non-PASS case manifests from a caller's pinned campaign inputs.

This only stages input identity and an optional decoded source frame. It never
runs inference, constructs a visual result, or marks a validation gate PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
TASKS = {"detection", "classification", "segmentation", "pose", "depth",
         "stereo-depth", "stereo-proximity"}
ORACLE_VALIDATOR = Path(__file__).resolve().parents[1] / "multitask" / "validation" / "validate_semantic_oracle.py"
VISION13_CASES = Path(__file__).resolve().parents[2] / "assets" / "vision13-cases.json"


def _validate_suite(data: dict, cases: list[dict]) -> None:
    suite = data.get("suite")
    if suite is None:
        return
    if suite not in {"vision13", "vision14"}:
        raise ValueError(f"unsupported campaign suite: {suite}")
    expected = json.loads(VISION13_CASES.read_text(encoding="utf-8"))
    expected_cases = expected["cases"] + ([expected["vision14_additional_case"]]
                                          if suite == "vision14" else [])
    expected_pairs = {(item["case_id"], item["task"]) for item in expected_cases}
    actual_pairs = {(item["case_id"], item["task"]) for item in cases}
    if len(cases) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise ValueError(f"{suite} requires exactly its {len(expected_pairs)} application case IDs and tasks")


def _artifact(path_value: Any) -> dict[str, str] | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value).resolve()
    record = {"path": str(path)}
    if path.is_file() and path.stat().st_size:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        record["sha256"] = digest.hexdigest()
    return record


def _exclusive_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")


def prepare(input_path: Path, models_dir: Path, *, capture_source_frame: bool = False) -> dict:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("campaign input must be a JSON object")
    cases = data.get("cases")
    if data.get("schema_version") != 1 or not isinstance(cases, list) or not cases:
        raise ValueError("the campaign input must contain one or more application cases")
    if not all(isinstance(case, dict) for case in cases):
        raise ValueError("every campaign case must be an object")
    ids = [case.get("case_id") for case in cases]
    if any(not isinstance(value, str) or not SAFE_NAME.fullmatch(value) for value in ids):
        raise ValueError("case IDs must be unique safe names")
    if len(set(ids)) != len(ids):
        raise ValueError("case IDs must be unique safe names")
    if any(not isinstance(case.get("task"), str) or case.get("task") not in TASKS
           for case in cases):
        raise ValueError("campaign contains an unsupported report task")
    _validate_suite(data, cases)
    for case in cases:
        if not isinstance(case.get("inputs", []), list) or not isinstance(case.get("components", []), list):
            raise ValueError(f"inputs/components must be arrays for {case['case_id']}")
        if not isinstance(case.get("upstream_stages", []), list):
            raise ValueError(f"upstream_stages must be an array for {case['case_id']}")
    if models_dir.exists() and any(models_dir.iterdir()):
        raise FileExistsError("campaign output directory must be empty; do not overwrite prior evidence")
    models_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in cases:
        case_dir = models_dir / case["case_id"]
        case_dir.mkdir()
        source = _artifact(case.get("source"))
        components = [_artifact(value) for value in case.get("components", [])]
        inputs = []
        for item in case.get("inputs", []):
            artifact = _artifact(item.get("path")) if isinstance(item, dict) else None
            if artifact:
                inputs.append({"role": item.get("role", "media"), **artifact})
        calibration_input = case.get("calibration")
        calibration_path = (calibration_input.get("path") if isinstance(calibration_input, dict)
                            else calibration_input)
        calibration = _artifact(calibration_path)
        oracle = _artifact(case.get("oracle"))
        oracle_status = "UNTESTED"
        oracle_validation = None
        if oracle and oracle.get("sha256"):
            try:
                checked = subprocess.run(
                    [sys.executable, str(ORACLE_VALIDATOR), oracle["path"],
                     "--task", case["task"]],
                    capture_output=True, text=True, check=False, timeout=30,
                )
                parsed = json.loads(checked.stdout) if checked.stdout.strip() else {}
                oracle_status = (parsed.get("status", "BLOCKED")
                                 if checked.returncode == 0 and isinstance(parsed, dict)
                                 else "BLOCKED")
                oracle_validation = {
                    "status": oracle_status, "oracle_sha256": oracle["sha256"],
                    "validator_exit_code": checked.returncode,
                    "errors": (parsed.get("errors", []) if isinstance(parsed, dict) and parsed
                               else [checked.stderr.strip()[-1000:]]),
                }
                if checked.returncode != 0:
                    oracle_validation["reported_result"] = parsed
                    diagnostics = oracle_validation["errors"]
                    oracle_validation["errors"] = (list(diagnostics)
                                                   if isinstance(diagnostics, list)
                                                   else [str(diagnostics)])
                    oracle_validation["errors"].append(
                        f"oracle validator exited {checked.returncode}: {checked.stderr.strip()[-1000:]}"
                    )
            except (OSError, UnicodeError, ValueError, subprocess.TimeoutExpired):
                oracle_status = "BLOCKED"
                oracle_validation = {"status": "BLOCKED", "oracle_sha256": oracle["sha256"],
                                     "errors": ["oracle validator did not produce a valid result"]}
        execution_blockers = []
        if not source or not source.get("sha256"):
            execution_blockers.append("exact local model artifact is absent")
        if any(item is None or not item.get("sha256") for item in components):
            execution_blockers.append("one or more application components are absent")
        if not inputs or any(not item.get("sha256") for item in inputs):
            execution_blockers.append("required media fixture is absent")
        if case["task"].startswith("stereo-"):
            roles = [item.get("role") for item in inputs]
            if roles.count("left") != 1 or roles.count("right") != 1:
                execution_blockers.append("stereo requires exactly one left and one right input")
            pair = [item for item in inputs if item.get("role") in {"left", "right"}]
            if len(pair) == 2 and pair[0].get("path") == pair[1].get("path"):
                execution_blockers.append("stereo left and right inputs cannot be the same file")
            if not calibration or not calibration.get("sha256"):
                execution_blockers.append("stereo calibration file is absent or empty")
        validation_limitations = []
        if oracle_status != "PASS":
            validation_limitations.append(f"reviewed semantic oracle is {oracle_status}")
        missing = execution_blockers + validation_limitations + [
            "DeepStream/backend/semantic/visual execution has not run"
        ]
        manifest = {
            "schema_version": 1, "case_id": case["case_id"], "task": case["task"],
            "revision": case.get("revision") or (f"sha256:{source['sha256']}" if source and source.get("sha256") else "unresolved"),
            "source": source, "components": components, "inputs": inputs,
            "oracle": oracle, "oracle_status": oracle_status,
            "upstream_stages": case.get("upstream_stages", []),
            "calibration": calibration if case["task"].startswith("stereo-") else None,
            "commands": [], "gates": {name: {"status": "UNTESTED"} for name in
                                      ("backend", "pipeline", "semantic", "visual")},
            "capacity": {"status": "UNTESTED", "reason": "No compatible multi-stream run is recorded"},
            "evidence": {},
            "input_blockers": execution_blockers,
            "certification_limitations": validation_limitations,
            "first_failure": {"phase": "input" if execution_blockers else "execution",
                              "reason": (execution_blockers[0] if execution_blockers else
                                         "DeepStream/backend/semantic/visual execution has not run")},
            "preflight_findings": missing,
        }
        if oracle_validation is not None:
            oracle_validation_path = case_dir / "oracle-validation.json"
            _exclusive_json(oracle_validation_path, oracle_validation)
            manifest["evidence"]["oracle_validation"] = _artifact(str(oracle_validation_path))
        if capture_source_frame and inputs and inputs[0].get("sha256"):
            frame = case_dir / "source-frame.png"
            attempt = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", inputs[0]["path"],
                 "-frames:v", "1", "-update", "1", str(frame)],
                capture_output=True, text=True, check=False, timeout=45,
            )
            if attempt.returncode == 0:
                frame_record = _artifact(str(frame))
                if frame_record and frame_record.get("sha256"):
                    manifest["evidence"]["source_frame"] = frame_record
                else:
                    manifest["preflight_findings"].append("source frame decode produced no image")
            else:
                manifest["preflight_findings"].append("source frame decode failed")
        _exclusive_json(case_dir / "case-evidence.json", manifest)
        summary.append({"case_id": case["case_id"], "preflight_findings": missing})
    _exclusive_json(models_dir / "campaign.json",
                    {"schema_version": 1, "campaign_id": data.get("campaign_id", ""),
                     **({"suite": data["suite"]} if data.get("suite") else {}),
                     "cases": ids})
    return {"models_dir": str(models_dir), "cases": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("models_dir", type=Path)
    parser.add_argument("--capture-source-frame", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.input, args.models_dir,
                                 capture_source_frame=args.capture_source_frame), indent=2))
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"CAMPAIGN BLOCKED: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
