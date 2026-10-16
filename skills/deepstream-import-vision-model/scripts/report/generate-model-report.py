#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render one evidence-backed Step 8 report from models/<case>/case-evidence.json.

This is a reporter, not an inference runner. A claimed PASS is downgraded unless
the corresponding files and producer-authored validation records are present.
The legacy detection benchmark Markdown and its five charts remain authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


GATES = ("backend", "pipeline", "semantic", "visual")
EVIDENCE_FIELDS = (
    "input_fixture", "backend_parity", "deepstream_runtime",
    "structured_output", "semantic_validation", "overlay_evidence",
    "rendered_output", "visual_validation",
)
DETECTION_CHARTS = (
    "chart_trtexec_bs1_vs_bsmax.png", "chart_trtexec_throughput.png",
    "chart_ds_streams_vs_fps.png", "chart_trt_vs_ds.png",
    "chart_efficiency.png",
)
DETECTION_JSON_EVIDENCE = (
    "backend_parity", "deepstream_runtime", "semantic_validation",
    "visual_validation", "benchmark_data",
)
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
VALID_TASKS = {
    "detection", "classification", "segmentation", "pose", "depth",
    "stereo-depth", "stereo-proximity",
}
VALID_STATUSES = {"PASS", "FAIL", "BLOCKED", "UNTESTED", "ADAPTER_REQUIRED"}
CAPACITY_STATUSES = {"PASS", "FAIL", "BLOCKED", "UNTESTED"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(case_dir: Path, item: Any, *, label: str, errors: list[str]) -> Path | None:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        errors.append(f"{label}: path is missing")
        return None
    path = Path(item["path"])
    if not path.is_absolute():
        path = case_dir / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        errors.append(f"{label}: file is missing or empty: {path}")
        return None
    expected = item.get("sha256")
    if expected and expected != _sha256(path):
        errors.append(f"{label}: SHA-256 differs from the recorded evidence")
        return None
    return path


def _read_json(path: Path, *, label: str, errors: list[str]) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"{label}: invalid JSON: {exc}")
        return {}
    if not isinstance(result, dict):
        errors.append(f"{label}: expected a JSON object")
        return {}
    return result


def _adapter_evidence(status_path: Path, source: Path | None,
                      inputs: list[Path], oracle: Path | None,
                      task: str, errors: list[str],
                      source_frame: Path | None,
                      rendered_frame: Path | None) -> None:
    status = _read_json(status_path, label="adapter_status", errors=errors)
    if (status.get("schema_version") != 1 or status.get("task") != task or
            status.get("status") != "PASS" or status.get("implementation_status") != "tested"):
        errors.append("adapter_status: tested PASS promotion is absent")
        return
    if source and status.get("source_hash") != _sha256(source):
        errors.append("adapter_status: validated model differs from this case")
    evidence = status.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("adapter_status: evidence map is absent")
        return
    parsed: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    for name in EVIDENCE_FIELDS:
        item = evidence.get(name)
        path = _file(status_path.parent, item, label=f"adapter_status.{name}", errors=errors)
        if path is None or not isinstance(item, dict) or not item.get("sha256"):
            errors.append(f"adapter_status.{name}: verified digest is required")
            continue
        parsed[name] = _read_json(path, label=name, errors=errors) if path.suffix == ".json" else {}
        digests[name] = item["sha256"]
        if name == "input_fixture" and path not in inputs:
            errors.append("adapter_status: validated fixture differs from this case")
    common_records = ("backend_parity", "deepstream_runtime", "structured_output",
                      "semantic_validation", "overlay_evidence", "visual_validation")
    campaigns = set()
    fixture_hash = digests.get("input_fixture")
    source_hash = _sha256(source) if source else None
    for name in common_records:
        record = parsed.get(name, {})
        if record.get("schema_version") != 1 or record.get("task") != task:
            errors.append(f"adapter_status.{name}: evidence schema/task mismatch")
        if record.get("source_hash") != source_hash:
            errors.append(f"adapter_status.{name}: model artifact binding mismatch")
        if record.get("input_fixture_sha256") != fixture_hash:
            errors.append(f"adapter_status.{name}: fixture binding mismatch")
        if record.get("campaign_id"):
            campaigns.add(record["campaign_id"])
        elif name != "overlay_evidence":
            errors.append(f"adapter_status.{name}: campaign_id is absent")
    if len(campaigns) != 1:
        errors.append("adapter_status: evidence must share one campaign_id")
    if parsed.get("backend_parity", {}).get("status") != "PASS":
        errors.append("backend_parity: PASS is required")
    runtime = parsed.get("deepstream_runtime", {})
    try:
        buffer_count = int(runtime.get("buffer_count") or 0)
    except (TypeError, ValueError):
        buffer_count = 0
    if not (runtime.get("status") == "PASS" and runtime.get("startup") is True
            and buffer_count > 0 and runtime.get("eos") is True
            and runtime.get("exit_code") == 0):
        errors.append("deepstream_runtime: startup, buffers, EOS, and exit 0 are required")
    if parsed.get("semantic_validation", {}).get("status") != "PASS":
        errors.append("semantic_validation: PASS is required")
    if (oracle is None or parsed.get("semantic_validation", {}).get("oracle_sha256")
            != _sha256(oracle)):
        errors.append("semantic_validation: reviewed oracle digest differs")
    overlay = parsed.get("overlay_evidence", {})
    if source_frame is None or overlay.get("source_frame_sha256") != _sha256(source_frame):
        errors.append("overlay_evidence: input frame digest differs")
    visual = parsed.get("visual_validation", {})
    if visual.get("status") != "RENDERED_PASS" or visual.get("pipeline_visual_result") != "PASS":
        errors.append("visual_validation: model-derived rendered PASS is required")
    if rendered_frame is None or visual.get("rendered_output_sha256") != _sha256(rendered_frame):
        errors.append("visual_validation: rendered frame digest differs")
    structured_hash = digests.get("structured_output")
    rendered_hash = digests.get("rendered_output")
    for name in ("deepstream_runtime", "semantic_validation", "overlay_evidence", "visual_validation"):
        if parsed.get(name, {}).get("structured_output_sha256") != structured_hash:
            errors.append(f"adapter_status.{name}: structured output digest differs")
    for name in ("deepstream_runtime", "visual_validation"):
        if parsed.get(name, {}).get("rendered_output_sha256") != rendered_hash:
            errors.append(f"adapter_status.{name}: rendered output digest differs")


def _gate_status(manifest: dict[str, Any], name: str) -> str:
    gates = manifest.get("gates", {})
    entry = gates.get(name, {}) if isinstance(gates, dict) else {}
    return entry.get("status", "UNTESTED") if isinstance(entry, dict) else "UNTESTED"


def _pipeline_visual_result(manifest: dict[str, Any], evidence: dict[str, Path]) -> str:
    """Keep the runtime-visual verdict orthogonal to full certification."""
    statuses = (_gate_status(manifest, "pipeline"), _gate_status(manifest, "visual"))
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    required = ("model_artifact", "input_0", "source_frame", "rendered_frame",
                "rendered_video", "structured_output", "deepstream_runtime",
                "visual_validation")
    if statuses != ("PASS", "PASS") or not all(name in evidence for name in required):
        return "UNTESTED"
    declared = manifest.get("evidence", {})
    if not isinstance(declared, dict) or any(
        not isinstance(declared.get(name), dict) or not declared[name].get("sha256")
        for name in ("source_frame", "rendered_frame", "rendered_video", "structured_output")
    ):
        return "UNTESTED"
    try:
        runtime = json.loads(evidence["deepstream_runtime"].read_text(encoding="utf-8"))
        visual = json.loads(evidence["visual_validation"].read_text(encoding="utf-8"))
        if not isinstance(runtime, dict) or not isinstance(visual, dict):
            return "UNTESTED"
        source_hash = _sha256(evidence["model_artifact"])
        fixture_hash = _sha256(evidence["input_0"])
        structured_hash = _sha256(evidence["structured_output"])
        rendered_hash = _sha256(evidence["rendered_frame"])
        source_frame_hash = _sha256(evidence["source_frame"])
        for record in (runtime, visual):
            if (record.get("schema_version") != 1 or record.get("task") != manifest.get("task")
                    or record.get("source_hash") != source_hash
                    or record.get("input_fixture_sha256") != fixture_hash
                    or record.get("structured_output_sha256") != structured_hash
                    or not isinstance(record.get("producer"), str)
                    or not record["producer"].strip()):
                return "UNTESTED"
        if (not runtime.get("campaign_id") or
                runtime["campaign_id"] != visual.get("campaign_id") or
                runtime.get("status") != "PASS" or runtime.get("startup") is not True or
                int(runtime.get("buffer_count") or 0) < 1 or runtime.get("eos") is not True or
                runtime.get("exit_code") != 0 or
                visual.get("status") != "RENDERED_PASS" or
                visual.get("pipeline_visual_result") != "PASS"):
            return "UNTESTED"
        if (runtime.get("rendered_output_sha256", rendered_hash) != rendered_hash or
                visual.get("rendered_frame_sha256",
                           visual.get("rendered_output_sha256")) != rendered_hash):
            return "UNTESTED"
        overlay = evidence.get("overlay_evidence")
        overlay_source = None
        if overlay is not None:
            overlay_record = json.loads(overlay.read_text(encoding="utf-8"))
            if isinstance(overlay_record, dict):
                overlay_source = overlay_record.get("source_frame_sha256")
        if visual.get("source_frame_sha256", overlay_source) != source_frame_hash:
            return "UNTESTED"
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        return "UNTESTED"
    return "PASS"


def _capacity(manifest: dict[str, Any], case_dir: Path, errors: list[str]) -> dict[str, Any]:
    entry = manifest.get("capacity", {})
    if not isinstance(entry, dict):
        return {"status": "UNTESTED", "reason": "capacity record is not an object"}
    result = {"status": "UNTESTED", **entry}
    status = result.get("status", "UNTESTED")
    if status not in CAPACITY_STATUSES:
        result["status"] = "UNTESTED"
        result["reason"] = "capacity status is invalid"
        return result
    if status in {"PASS", "FAIL"}:
        try:
            if isinstance(result.get("fps_per_stream"), bool) or isinstance(result.get("target_fps"), bool):
                raise ValueError("boolean FPS value")
            measured = float(result["fps_per_stream"])
            target = float(result["target_fps"])
            streams = int(result["streams"])
        except (KeyError, TypeError, ValueError):
            measured, target, streams = 0.0, 0.0, 0
        log_record = result.get("log")
        log = _file(case_dir, log_record, label="capacity.log", errors=errors)
        if log is not None and (not isinstance(log_record, dict) or not log_record.get("sha256")):
            errors.append("capacity: log SHA-256 is required")
        valid_measurement = (log is not None and math.isfinite(measured)
                             and math.isfinite(target) and target > 0 and streams >= 2
                             and isinstance(log_record, dict) and bool(log_record.get("sha256"))
                             and result.get("task") == manifest.get("task")
                             and bool(result.get("runner")))
        if status == "PASS" and (not valid_measurement or measured < target):
            errors.append("capacity: PASS lacks a measured task-compatible multi-stream run")
            result["status"] = "UNTESTED"
            result["reason"] = "complete task-compatible multi-stream evidence was not verified"
        elif status == "FAIL" and not valid_measurement:
            errors.append("capacity: FAIL lacks a task-compatible runner log")
            result["status"] = "UNTESTED"
            result["reason"] = "failure evidence is incomplete"
    return result


def _validate_upstream_stage(case_dir: Path, stage: str, item: Any,
                             input_paths: list[Path | None],
                             artifact_hashes: set[str], errors: list[str]
                             ) -> tuple[Path | None, Path | None]:
    """Require a passing, fixture-bound stage record and its hashed output."""
    record_path = _file(case_dir, item, label=f"upstream {stage}", errors=errors)
    if record_path is None:
        return None, None
    if not isinstance(item, dict) or not item.get("sha256"):
        errors.append(f"upstream {stage}: record SHA-256 is required")
    record = _read_json(record_path, label=f"upstream {stage}", errors=errors)
    if record.get("schema_version") != 1 or record.get("stage") != stage:
        errors.append(f"upstream {stage}: schema or stage identity differs")
    if record.get("status") != "PASS" or not isinstance(record.get("producer"), str) or not record["producer"].strip():
        errors.append(f"upstream {stage}: passing producer record is required")
    kind = record.get("kind")
    if kind not in {"inference", "transform"}:
        errors.append(f"upstream {stage}: kind must be inference or transform")
    if kind == "inference" and record.get("model_sha256") not in artifact_hashes:
        errors.append(f"upstream {stage}: inference model must match a declared artifact")
    if kind == "transform" and record.get("model_sha256") is not None:
        errors.append(f"upstream {stage}: transform stage cannot claim a model artifact")
    expected_inputs = [_sha256(path) for path in input_paths if path is not None]
    if len(expected_inputs) != len(input_paths) or record.get("input_fixture_sha256s") != expected_inputs:
        errors.append(f"upstream {stage}: input fixture bindings differ")
    output_record = record.get("output")
    output = _file(case_dir, output_record, label=f"upstream {stage} output", errors=errors)
    if output is not None and (not isinstance(output_record, dict) or not output_record.get("sha256")):
        errors.append(f"upstream {stage}: output SHA-256 is required")
    return record_path, output


def _is_renderable_frame(path: Path | None) -> bool:
    if path is None or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return False
    try:
        with path.open("rb") as stream:
            header = stream.read(8)
    except OSError:
        return False
    return header.startswith(b"\x89PNG\r\n\x1a\n") or header.startswith(b"\xff\xd8\xff")


def evaluate(manifest: dict[str, Any], case_dir: Path) -> tuple[str, list[str], dict[str, Path], dict[str, Any]]:
    """Return a truthful result; never use an unverified declared overall status."""
    errors: list[str] = []
    findings = manifest.get("preflight_findings", [])
    if not isinstance(findings, list) or any(not isinstance(item, str) for item in findings):
        errors.append("preflight findings must be a list of strings")
        manifest["preflight_findings"] = []
    evidence_files: dict[str, Path] = {}
    case_id = manifest.get("case_id")
    task = manifest.get("task")
    if manifest.get("schema_version") != 1 or case_id != case_dir.name or not isinstance(case_id, str) or not SAFE_NAME.fullmatch(case_id):
        raise ValueError("case manifest must have schema_version=1 and case_id matching its directory")
    if not isinstance(task, str) or task not in VALID_TASKS:
        raise ValueError(f"unsupported report task: {task}")
    gates = manifest.get("gates")
    if not isinstance(gates, dict):
        gates = {}
        errors.append("gate records are absent")
        manifest["gates"] = gates
    for name in GATES:
        if _gate_status(manifest, name) not in VALID_STATUSES:
            errors.append(f"{name}: invalid gate status")

    source = _file(case_dir, manifest.get("source"), label="model artifact", errors=errors)
    artifact_hashes: set[str] = set()
    if source:
        evidence_files["model_artifact"] = source
        artifact_hashes.add(_sha256(source))
    components = manifest.get("components", [])
    if not isinstance(components, list):
        errors.append("application components must be a list")
        components = []
    for index, item in enumerate(components):
        path = _file(case_dir, item, label=f"component {index}", errors=errors)
        if path:
            evidence_files[f"component_{index}"] = path
            artifact_hashes.add(_sha256(path))
        if isinstance(item, dict) and not item.get("sha256"):
            errors.append(f"component {index}: SHA-256 is absent")
    inputs = manifest.get("inputs", [])
    input_paths: list[Path | None] = []
    if not isinstance(inputs, list) or not inputs:
        errors.append("input fixture list is absent")
        inputs = []
    else:
        for index, item in enumerate(inputs):
            path = _file(case_dir, item, label=f"input {index}", errors=errors)
            input_paths.append(path)
            if path:
                evidence_files[f"input_{index}"] = path
    if not manifest.get("revision") or not isinstance(manifest.get("revision"), str):
        errors.append("model revision is not pinned")
    if isinstance(manifest.get("source"), dict) and not manifest["source"].get("sha256"):
        errors.append("model artifact SHA-256 is absent")
    for index, item in enumerate(inputs):
        if isinstance(item, dict) and not item.get("sha256"):
            errors.append(f"input {index}: SHA-256 is absent")
    oracle = manifest.get("oracle")
    if oracle is None:
        errors.append("reviewed semantic oracle is absent")
    else:
        path = _file(case_dir, oracle, label="semantic oracle", errors=errors)
        if path:
            evidence_files["semantic_oracle"] = path
        if manifest.get("oracle_status") != "PASS":
            errors.append("semantic oracle is not reviewed PASS")
        if not isinstance(oracle, dict) or not oracle.get("sha256"):
            errors.append("semantic oracle SHA-256 is required")
        declared = manifest.get("evidence", {})
        oracle_result = declared.get("oracle_validation") if isinstance(declared, dict) else None
        result_path = _file(case_dir, oracle_result, label="oracle validation", errors=errors)
        if result_path:
            evidence_files["oracle_validation"] = result_path
            result = _read_json(result_path, label="oracle validation", errors=errors)
            if (result.get("status") != "PASS" or path is None
                    or result.get("oracle_sha256") != _sha256(path)):
                errors.append("oracle validation: PASS result is absent or bound to another file")
            validator_errors = result.get("errors", [])
            if isinstance(validator_errors, list):
                errors.extend(f"semantic oracle: {message}" for message in validator_errors)
        else:
            errors.append("oracle validation: producer result is required")
    stages = manifest.get("upstream_stages", [])
    if not isinstance(stages, list) or any(not isinstance(stage, str) or
                                           not SAFE_NAME.fullmatch(stage) for stage in stages):
        errors.append("upstream stage contract is invalid")
        stages = []
        manifest["upstream_stages"] = stages
    commands = manifest.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("executed commands and logs are absent")
    else:
        for index, command in enumerate(commands):
            if not isinstance(command, dict) or command.get("exit_code") != 0:
                errors.append(f"command {index}: successful execution is not recorded")
                continue
            log = _file(case_dir, command.get("log"), label=f"command {index} log", errors=errors)
            if log is not None and not command["log"].get("sha256"):
                errors.append(f"command {index}: log SHA-256 is absent")
    capacity_errors: list[str] = []
    capacity = _capacity(manifest, case_dir, capacity_errors)
    if task.startswith("stereo-"):
        roles = [item.get("role") for item in inputs if isinstance(item, dict)]
        if roles.count("left") != 1 or roles.count("right") != 1:
            errors.append("stereo: exactly one synchronized left and right fixture is required")
        pair = [item for item in inputs if isinstance(item, dict)
                and item.get("role") in {"left", "right"}]
        if len(pair) == 2 and pair[0].get("path") == pair[1].get("path"):
            errors.append("stereo: left and right fixtures cannot be the same file")
        calibration_record = manifest.get("calibration")
        calibration = _file(case_dir, calibration_record, label="stereo calibration", errors=errors)
        if calibration is not None:
            evidence_files["calibration"] = calibration
            if not isinstance(calibration_record, dict) or not calibration_record.get("sha256"):
                errors.append("stereo calibration: SHA-256 is required")

    declared_evidence = manifest.get("evidence", {})
    if not isinstance(declared_evidence, dict):
        declared_evidence = {}
    for name in ("source_frame", "rendered_frame", "rendered_video", "structured_output"):
        evidence_record = declared_evidence.get(name)
        path = _file(case_dir, evidence_record, label=name, errors=errors)
        if path:
            evidence_files[name] = path
            if name in {"source_frame", "rendered_frame"}:
                if not isinstance(evidence_record, dict) or not evidence_record.get("sha256"):
                    errors.append(f"{name}: SHA-256 is required")
                if not _is_renderable_frame(path):
                    errors.append(f"{name}: a valid PNG or JPEG frame is required")
    for name in ("deepstream_runtime", "visual_validation", "overlay_evidence"):
        item = declared_evidence.get(name)
        if item is not None:
            path = _file(case_dir, item, label=name, errors=errors)
            if path is not None and isinstance(item, dict) and item.get("sha256"):
                evidence_files[name] = path
            elif path is not None:
                errors.append(f"{name}: SHA-256 is required")
    for stage in stages:
        stage_path, stage_output = _validate_upstream_stage(
            case_dir, stage, declared_evidence.get(f"upstream_{stage}"), input_paths,
            artifact_hashes, errors)
        if stage_path:
            evidence_files[f"upstream_{stage}"] = stage_path
        if stage_output:
            evidence_files[f"upstream_{stage}_output"] = stage_output
    # A status report may contain a valid pipeline/overlay attempt even when
    # semantic or benchmark certification never ran. Require the complete
    # legacy detection bundle only when every gate claims a functional PASS;
    # otherwise the first causal blocker gets buried under chart/producer noise.
    if task == "detection" and all(_gate_status(manifest, name) == "PASS" for name in GATES):
        for name in ("kitti_output", "benchmark_data", "benchmark_report"):
            path = _file(case_dir, declared_evidence.get(name), label=name, errors=errors)
            if path:
                evidence_files[name] = path
        charts = case_dir / "reports" / "charts"
        for chart in DETECTION_CHARTS:
            _file(case_dir, {"path": str(charts / chart)}, label=f"chart {chart}", errors=errors)
        detection_records: dict[str, dict[str, Any]] = {}
        for name in DETECTION_JSON_EVIDENCE:
            item = declared_evidence.get(name)
            path = _file(case_dir, item, label=name, errors=errors)
            if path is not None:
                if not isinstance(item, dict) or not item.get("sha256"):
                    errors.append(f"{name}: SHA-256 is required")
                detection_records[name] = _read_json(path, label=name, errors=errors)
        for name in ("backend_parity", "deepstream_runtime", "semantic_validation", "visual_validation"):
            if detection_records.get(name, {}).get("status") not in {"PASS", "RENDERED_PASS"}:
                errors.append(f"{name}: passing producer record is required")
        expected_source_hash = _sha256(source) if source else None
        fixture_paths = [evidence_files[key] for key in evidence_files if key.startswith("input_")]
        expected_input_hash = _sha256(fixture_paths[0]) if len(fixture_paths) == 1 else None
        producer_ids = set()
        for name in DETECTION_JSON_EVIDENCE[:-1]:
            record = detection_records.get(name, {})
            if not record.get("producer") or not record.get("campaign_id"):
                errors.append(f"{name}: producer and campaign_id are required")
            producer_ids.add(record.get("campaign_id"))
            if record.get("task") != task or record.get("source_hash") != expected_source_hash:
                errors.append(f"{name}: model/task identity binding differs")
            if record.get("input_fixture_sha256") != expected_input_hash:
                errors.append(f"{name}: input fixture binding differs")
        if len(producer_ids) != 1:
            errors.append("detection evidence must share one campaign_id")
        runtime = detection_records.get("deepstream_runtime", {})
        try:
            runtime_buffers = int(runtime.get("buffer_count") or 0)
        except (TypeError, ValueError):
            runtime_buffers = 0
        if not (runtime.get("startup") is True and runtime_buffers > 0
                and runtime.get("eos") is True and runtime.get("exit_code") == 0):
            errors.append("deepstream_runtime: startup, buffers, EOS, and exit 0 are required")
        visual = detection_records.get("visual_validation", {})
        if visual.get("pipeline_visual_result") != "PASS":
            errors.append("visual_validation: rendered model-output PASS is required")
        rendered_frame = evidence_files.get("rendered_frame")
        source_frame = evidence_files.get("source_frame")
        structured_output = evidence_files.get("structured_output")
        if rendered_frame is None or visual.get("rendered_frame_sha256") != _sha256(rendered_frame):
            errors.append("visual_validation: rendered frame digest differs")
        if source_frame is None or visual.get("source_frame_sha256") != _sha256(source_frame):
            errors.append("visual_validation: input frame digest differs")
        if structured_output is None:
            errors.append("structured_output: runtime result is required")
        else:
            structured_hash = _sha256(structured_output)
            for name in ("deepstream_runtime", "semantic_validation", "visual_validation"):
                if detection_records.get(name, {}).get("structured_output_sha256") != structured_hash:
                    errors.append(f"{name}: structured output digest differs")
        if detection_records.get("semantic_validation", {}).get("oracle_status") != "PASS":
            errors.append("semantic_validation: reviewed oracle PASS is required")
        benchmark_data = detection_records.get("benchmark_data", {})
        if not isinstance(benchmark_data, dict) or not benchmark_data:
            errors.append("benchmark_data: nonempty measured JSON is required")
        oracle_path = evidence_files.get("semantic_oracle")
        oracle_result_path = evidence_files.get("oracle_validation")
        oracle_result = (_read_json(oracle_result_path, label="oracle validation", errors=errors)
                         if oracle_result_path else {})
        if (oracle_path is None or oracle_result.get("status") != "PASS"
                or detection_records.get("semantic_validation", {}).get("oracle_sha256") != _sha256(oracle_path)):
            errors.append("semantic_validation: oracle digest differs")
        benchmark = evidence_files.get("benchmark_report")
        if benchmark:
            body = benchmark.read_text(encoding="utf-8")
            if len(re.findall(r"^##\s+\d+\.", body, flags=re.MULTILINE)) != 12:
                errors.append("benchmark_report: exactly 12 numbered sections are required")
            for chart in DETECTION_CHARTS:
                if chart not in body:
                    errors.append(f"benchmark_report: missing chart reference {chart}")
            for name in ("source_frame", "rendered_frame"):
                if not re.search(r"!\[[^]]*\]\(assets/" + name + r"\.(?:png|jpe?g)\)", body):
                    errors.append(f"benchmark_report: missing embedded {name} reference")
                original = evidence_files.get(name)
                if original:
                    embedded = case_dir / "reports" / "assets" / f"{name}{original.suffix.lower()}"
                    if not embedded.is_file() or _sha256(embedded) != _sha256(original):
                        errors.append(f"benchmark_report: embedded {name} differs from runtime evidence")
    if task != "detection":
        adapter = _file(case_dir, declared_evidence.get("adapter_status"), label="adapter_status", errors=errors)
        if adapter:
            _adapter_evidence(adapter, source,
                              [evidence_files[key] for key in evidence_files if key.startswith("input_")],
                              evidence_files.get("semantic_oracle"), task, errors,
                              evidence_files.get("source_frame"),
                              evidence_files.get("rendered_frame"))
            adapter_record = _read_json(adapter, label="adapter_status", errors=errors)
            bound = adapter_record.get("evidence", {})
            if isinstance(bound, dict):
                for name in ("deepstream_runtime", "visual_validation", "overlay_evidence"):
                    item = bound.get(name)
                    if name not in evidence_files and isinstance(item, dict) and item.get("sha256"):
                        path = _file(adapter.parent, item, label=f"adapter_status.{name}", errors=errors)
                        if path is not None:
                            evidence_files[name] = path
    if all(_gate_status(manifest, name) == "PASS" for name in GATES):
        for name in ("source_frame", "rendered_frame", "structured_output"):
            if name not in evidence_files:
                errors.append(f"{name}: required for a functional PASS")
    if all(_gate_status(manifest, name) == "PASS" for name in GATES):
        for name, item in declared_evidence.items():
            if isinstance(item, dict) and not item.get("sha256"):
                errors.append(f"{name}: SHA-256 is absent")
    if any(_gate_status(manifest, name) == "FAIL" for name in GATES):
        return "FAIL", errors + capacity_errors, evidence_files, capacity
    if task.startswith("stereo-"):
        complete_pair = (source is not None and
                         len([key for key in evidence_files if key.startswith("input_")]) >= 2 and
                         manifest.get("oracle_status") == "PASS" and
                         "calibration" in evidence_files)
        return ("ADAPTER_REQUIRED" if complete_pair else "BLOCKED"), (
            errors + capacity_errors), evidence_files, capacity
    if all(_gate_status(manifest, name) == "PASS" for name in GATES) and not errors:
        return "PASS", capacity_errors, evidence_files, capacity
    if (manifest.get("capability_status") == "ADAPTER_REQUIRED" or
            any(_gate_status(manifest, name) == "ADAPTER_REQUIRED" for name in GATES)) and source:
        verified_inputs = [key for key in evidence_files if key.startswith("input_")]
        intake_ready = (len(verified_inputs) == len(inputs)
                        and len([key for key in evidence_files if key.startswith("component_")]) == len(components)
                        and bool(evidence_files.get("semantic_oracle"))
                        and manifest.get("oracle_status") == "PASS"
                        and bool(manifest.get("revision")))
        if intake_ready:
            return "ADAPTER_REQUIRED", errors + capacity_errors, evidence_files, capacity
    return "BLOCKED", errors + capacity_errors, evidence_files, capacity


def _safe(value: Any) -> str:
    return html.escape(str(value).replace("\n", " ").replace("\r", " "))


def _write_markdown(manifest: dict[str, Any], case_dir: Path, outcome: str,
                    errors: list[str], evidence: dict[str, Path], capacity: dict[str, Any],
                    kind: str) -> Path:
    reports = case_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    assets = reports / "assets"
    assets.mkdir(exist_ok=True)
    lines = [f"# DeepStream {_safe(kind.title())} Report — {_safe(manifest['case_id'])}", "",
             f"**Result: {outcome}**", "", "## 1. Model Configuration", "",
             f"- Task: `{_safe(manifest['task'])}`",
             f"- Revision: `{_safe(manifest.get('revision', 'unresolved'))}`"]
    for name in ("model_artifact", *[key for key in evidence if key.startswith("component_")],
                 *[key for key in evidence if key.startswith("input_")],
                 "semantic_oracle", "oracle_validation"):
        path = evidence.get(name)
        if path:
            lines.append(f"- {_safe(name)}: `{_safe(path)}` (SHA-256 `{_sha256(path)}`)")
    lines.append(f"- Oracle status: `{_safe(manifest.get('oracle_status', 'UNTESTED'))}`")
    scope_note = manifest.get("scope_note")
    if isinstance(scope_note, str) and scope_note.strip():
        lines.append(f"- Scope and limitations: {_safe(scope_note.strip())}")
    stages = manifest.get("upstream_stages", [])
    if isinstance(stages, list) and stages and all(isinstance(stage, str) for stage in stages):
        lines.append("- Required upstream stages: `" +
                     _safe(", ".join(stages)) + "`")
    for n, title, field in ((2, "System Configuration", "environment"),
                            (3, "Preprocessing", "preprocessing")):
        lines += ["", f"## {n}. {title}", ""]
        values = manifest.get(field)
        if isinstance(values, dict) and values:
            lines += [f"- {_safe(k)}: `{_safe(v)}`" for k, v in sorted(values.items())]
        else:
            lines.append("- UNTESTED: no bound record is available.")
    lines += ["", "## 4. Engine Build Summary", "",
              "- Executed build commands and findings are recorded in Section 12.", "",
              "## 5. trtexec Results", "", "- UNTESTED: no verified benchmark is bound.", "",
              "## 6. PEAK_GPU_STREAMS Derivation", "", "- UNTESTED: no measured camera ceiling is inferred.", "",
              "## 7. Single-Stream Validation", "", "| Gate | Result |", "|---|---|"]
    for gate in GATES:
        lines.append(f"| {_safe(gate)} | {_safe(_gate_status(manifest, gate))} |")
    lines.append(f"| pipeline + visual (recorded) | {_pipeline_visual_result(manifest, evidence)} |")
    for name, label in (("source_frame", "Input frame"), ("rendered_frame", "Model-derived output frame")):
        path = evidence.get(name)
        if path and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            local = assets / f"{name}{path.suffix.lower()}"
            if path != local.resolve():
                shutil.copyfile(path, local)
            lines += ["", '<div style="page-break-before: always;"></div>', "",
                      f"### {label}", "", f"![{label}](assets/{local.name})"]
    lines += ["", "## 8. DeepStream Benchmark Results", "",
              "- Batched realtime stream capacity: UNTESTED until two verified benchmark runs are bound.", "",
              "### Separate task-capacity diagnostic", "", f"- Status: `{_safe(capacity['status'])}`"]
    if capacity.get("reason"):
        lines.append(f"- Finding: {_safe(capacity['reason'])}")
    if capacity.get("status") in {"PASS", "FAIL"}:
        lines.append(f"- FPS per stream: `{_safe(capacity.get('fps_per_stream', 'unavailable'))}`")
        lines.append(f"- Target FPS: `{_safe(capacity.get('target_fps', 'unavailable'))}`")
    lines += ["", "## 9. trtexec vs DeepStream Comparison", "",
              "- UNTESTED: comparable two-run measurements are not bound.", "",
              "| Metric | trtexec MAX_BS | DS Run 1 | DS Run 2 |",
              "|---|---|---|---|",
              "| Comparable throughput | UNTESTED | UNTESTED | UNTESTED |", "",
              "## 10. Efficiency Analysis", "", "- UNTESTED: no comparable throughput is bound.", "",
              "## 11. Pipeline Timing", "", "- UNTESTED: no per-phase timing record is bound.", "",
              "## 12. Reference Commands", ""]
    commands = manifest.get("commands", [])
    for item in commands if isinstance(commands, list) else []:
        if isinstance(item, dict):
            log = item.get("log")
            log_text = log.get("path", "none") if isinstance(log, dict) else log
            lines.append(f"- {_safe(item.get('phase', 'unknown'))}: exit `{_safe(item.get('exit_code', 'not run'))}`; log `{_safe(log_text)}`")
    if not commands:
        lines.append("- None recorded; execution must not be inferred.")
    lines += ["", "### Evidence", ""]
    for name, path in sorted(evidence.items()):
        lines.append(f"- {_safe(name)}: `{_safe(path)}` (SHA-256 `{_sha256(path)}`)")
    lines += ["", "### Findings and next action", ""]
    failure = manifest.get("first_failure")
    if outcome != "PASS":
        if isinstance(failure, dict):
            lines.append(f"- First recorded failure: {_safe(failure.get('phase', 'unknown'))}: {_safe(failure.get('reason', 'unspecified'))}")
        for error in errors:
            lines.append(f"- {_safe(error)}")
        findings = manifest.get("preflight_findings", [])
        for finding in findings if isinstance(findings, list) else []:
            lines.append(f"- Preflight: {_safe(finding)}")
        if not failure and errors:
            lines.append(f"- First validation failure: {_safe(errors[0])}")
        if not errors and not failure:
            lines.append("- One or more required validation gates did not pass.")
        lines.append("- Full functional certification is incomplete; the recorded pipeline and visual gate results above remain separate from this report outcome.")
    else:
        lines.append("- All required functional, semantic, and rendered-output gates passed for this exact fixture.")
    path = reports / f"{kind}_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate(manifest_path: Path, *, converter: Path | None = None) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("case manifest must contain a JSON object")
    case_dir = manifest_path.parent
    outcome, errors, evidence, capacity = evaluate(manifest, case_dir)
    kind = "benchmark" if outcome == "PASS" and manifest["task"] == "detection" else (
        "validation" if outcome == "PASS" else "status")
    reports = case_dir / "reports"
    if kind == "benchmark":
        markdown = evidence["benchmark_report"]
    else:
        markdown = _write_markdown(manifest, case_dir, outcome, errors, evidence, capacity, kind)
    if kind != "benchmark" and "benchmark" in manifest:
        subprocess.run([sys.executable, str(Path(__file__).with_name("benchmark-sections.py")),
                        str(markdown), str(manifest_path)], check=True, timeout=120)
    converter = converter or Path(__file__).with_name("md-to-html-pdf.py")
    css = Path(__file__).with_name("report-style.css")
    result = subprocess.run(
        [sys.executable, str(converter), str(markdown), str(css), str(reports),
         manifest["case_id"], "--report-kind", kind],
        text=True, capture_output=True, check=False, timeout=360,
    )
    if result.returncode:
        raise RuntimeError(f"PDF conversion failed: {result.stderr.strip()[:500]}")
    pdf = reports / f"{kind}_report_{manifest['case_id']}.pdf"
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError("PDF converter did not produce a nonempty report")
    summary = {
        "schema_version": 1, "case_id": manifest["case_id"], "task": manifest["task"],
        "functional_status": outcome, "capacity_status": capacity["status"],
        "pipeline_visual_result": _pipeline_visual_result(manifest, evidence),
        "errors": errors, "pdf": str(pdf), "pdf_sha256": _sha256(pdf),
        "markdown": str(markdown), "html": str(reports / f"{kind}_report.html"),
    }
    (reports / "report-result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="models/<case>/case-evidence.json")
    args = parser.parse_args()
    try:
        result = generate(args.manifest)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"REPORT BLOCKED: {exc}", file=sys.stderr)
        return 4
    print(json.dumps(result, indent=2))
    return 0 if result["functional_status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
