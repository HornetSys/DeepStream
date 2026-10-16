#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Promote an implemented type adapter only after every evidence gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "model"
DEEPSTREAM_DIR = SCRIPT_DIR.parent / "deepstream"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(DEEPSTREAM_DIR))

from generate_configs import _adapter_static_errors  # noqa: E402
from model_pipeline import (  # noqa: E402
    ADAPTER_EVIDENCE_FIELDS,
    ManifestError,
    manifest_from_dict,
)


GATE_FIELDS = {
    "backend_parity",
    "deepstream_runtime",
    "semantic_validation",
    "visual_validation",
}
JSON_FIELDS = GATE_FIELDS | {"structured_output", "overlay_evidence"}


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(manifest_path: Path, value: Any) -> Path:
    path = Path(os.fspath(value))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return Path(os.path.abspath(os.fspath(path)))


def _pass_status(data: Mapping[str, Any]) -> bool:
    for key in ("status", "overall_status", "result"):
        if key in data:
            return str(data[key]).strip().upper() in {"PASS", "RENDERED_PASS"}
    return False

DEEPSTREAM_RUNNER = DEEPSTREAM_DIR / "run_type_adapter_pipeline.py"
REQUIRED_VISUAL_CHECKS = {
    "decoded_frame",
    "structured_output",
    "overlay_difference",
    "overlay_provenance",
    "video",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _common_evidence_errors(
    field: str,
    data: Mapping[str, Any],
    *,
    task: str,
    source_hash: str,
    input_hash: str,
) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append(f"{field}.schema_version must equal 1")
    if not str(data.get("campaign_id", "")).strip():
        errors.append(f"{field}.campaign_id is required")
    if not str(data.get("producer", "")).strip():
        errors.append(f"{field}.producer is required")
    if str(data.get("task", "")).strip() != task:
        errors.append(f"{field}.task must equal {task!r}")
    if str(data.get("source_hash", "")).strip().lower() != source_hash.lower():
        errors.append(f"{field}.source_hash does not match the routed model")
    if str(data.get("input_fixture_sha256", "")).strip().lower() != input_hash:
        errors.append(f"{field}.input_fixture_sha256 does not match input_fixture")
    return errors


def _evidence_contract_errors(
    manifest: Any,
    parsed: Mapping[str, Mapping[str, Any]],
    digests: Mapping[str, str],
) -> list[str]:
    """Enforce evidence semantics and cross-file bindings, not self-reported PASS alone."""

    if manifest.task != "classification":
        return [
            f"no trusted DeepStream evidence runner is shipped for task {manifest.task!r}; "
            "keep the adapter ADAPTER_REQUIRED"
        ]

    errors: list[str] = []
    missing_json = sorted(JSON_FIELDS - set(parsed))
    contract = manifest.adapter_contract
    if contract.get("preprocess_strategy") != "nvinfer-native":
        return [
            "classification readiness requires the shipped runner's "
            "nvinfer-native preprocessing contract"
        ]
    reference_runner = Path(str(contract.get("reference_pipeline_runner", "")))
    if reference_runner.resolve() != DEEPSTREAM_RUNNER.resolve():
        return [
            "classification readiness requires adapter_contract."
            "reference_pipeline_runner to name the shipped trusted runner"
        ]

    if missing_json:
        return ["evidence schema validation unavailable for: " + ", ".join(missing_json)]

    input_hash = digests.get("input_fixture", "")
    evidence_source_hash = str(
        parsed["deepstream_runtime"].get("source_hash", "")
    ).strip().lower()
    if not _is_digest(evidence_source_hash):
        errors.append("deepstream_runtime.source_hash must be a SHA-256 digest")
    declared_source_hash = str(manifest.source_hash or "").strip().lower()
    if declared_source_hash and declared_source_hash != evidence_source_hash:
        errors.append(
            "deepstream_runtime.source_hash does not match the declared model hash"
        )

    structured_hash = digests.get("structured_output", "")
    rendered_hash = digests.get("rendered_output", "")
    campaigns: set[str] = set()
    for field in sorted(JSON_FIELDS):
        data = parsed[field]
        errors.extend(
            _common_evidence_errors(
                field,
                data,
                task=manifest.task,
                source_hash=evidence_source_hash,
                input_hash=input_hash,
            )
        )
        campaign = str(data.get("campaign_id", "")).strip()
        if campaign:
            campaigns.add(campaign)
    if len(campaigns) > 1:
        errors.append("all JSON evidence must use the same campaign_id")
    expected_producers = {
        "backend_parity": "compare_outputs.py",
        "deepstream_runtime": "deepstream-type-adapter-runner",
        "structured_output": "deepstream-type-adapter-runner",
        "semantic_validation": "adapter-semantic-validator",
        "overlay_evidence": "adapter-renderer",
        "visual_validation": "validate_rendered_output.py",
    }
    for field, expected in expected_producers.items():
        if str(parsed[field].get("producer", "")) != expected:
            errors.append(
                f"{field}.producer must equal {expected!r}"
            )

    backend = parsed["backend_parity"]
    metric = backend.get("metric")
    if str(backend.get("status", "")).upper() != "PASS":
        errors.append("backend_parity.status must be PASS")
    if not str(backend.get("reference_backend", "")).strip():
        errors.append("backend_parity.reference_backend is required")
    if not str(backend.get("candidate_backend", "")).strip():
        errors.append("backend_parity.candidate_backend is required")
    if not isinstance(metric, Mapping) or metric.get("passed") is not True or metric.get("finite") is not True:
        errors.append("backend_parity.metric must prove passed=true and finite=true")

    runtime = parsed["deepstream_runtime"]
    if str(runtime.get("status", "")).upper() != "PASS":
        errors.append("deepstream_runtime.status must be PASS")
    if runtime.get("startup") is not True:
        errors.append("deepstream_runtime.startup must be true")
    try:
        if int(runtime.get("buffer_count", 0)) <= 0:
            errors.append("deepstream_runtime.buffer_count must be greater than zero")
    except (TypeError, ValueError):
        errors.append("deepstream_runtime.buffer_count must be an integer")
    if runtime.get("eos") is not True:
        errors.append("deepstream_runtime.eos must be true")
    if runtime.get("exit_code") != 0:
        errors.append("deepstream_runtime.exit_code must equal 0")
    task_metadata = runtime.get("task_metadata")
    if not isinstance(task_metadata, Mapping):
        errors.append("deepstream_runtime.task_metadata must be an object")
    else:
        try:
            metadata_count = int(task_metadata.get("count", 0))
        except (TypeError, ValueError):
            metadata_count = 0
        tensor_names = task_metadata.get("tensor_names")
        if metadata_count <= 0 or not isinstance(tensor_names, list) or not tensor_names:
            errors.append("deepstream_runtime.task_metadata must prove a positive count and tensor names")
    assertion = runtime.get("numeric_assertion")
    if not isinstance(assertion, Mapping) or assertion.get("passed") is not True or assertion.get("finite") is not True:
        errors.append("deepstream_runtime.numeric_assertion must prove passed=true and finite=true")
    if str(runtime.get("structured_output_sha256", "")).lower() != structured_hash:
        errors.append("deepstream_runtime.structured_output_sha256 does not match structured_output")
    if str(runtime.get("rendered_output_sha256", "")).lower() != rendered_hash:
        errors.append("deepstream_runtime.rendered_output_sha256 does not match rendered_output")
    if str(runtime.get("producer", "")) != "deepstream-type-adapter-runner":
        errors.append("deepstream_runtime.producer must be deepstream-type-adapter-runner")
    try:
        runner_hash = _digest(DEEPSTREAM_RUNNER)
    except OSError as exc:
        errors.append(f"trusted DeepStream runner is unreadable: {exc}")
    else:
        if str(runtime.get("producer_sha256", "")).lower() != runner_hash:
            errors.append("deepstream_runtime.producer_sha256 does not match the trusted runner")

    structured = parsed["structured_output"]
    if structured.get("schema_version") != 1:
        errors.append("structured_output.schema_version must equal 1")
    identity = structured.get("identity")
    result = structured.get("result")
    records = structured.get("records")
    single_record = isinstance(identity, Mapping) and isinstance(result, Mapping) and bool(result)
    record_list = isinstance(records, list) and bool(records)
    if not single_record and not record_list:
        errors.append("structured_output must contain identity/result or non-empty records")

    semantic = parsed["semantic_validation"]
    if str(semantic.get("status", "")).upper() != "PASS":
        errors.append("semantic_validation.status must be PASS")
    if not isinstance(semantic.get("metric"), Mapping) or not semantic["metric"]:
        errors.append("semantic_validation.metric must be a non-empty object")
    if not _is_digest(semantic.get("oracle_sha256")):
        errors.append("semantic_validation.oracle_sha256 must be a SHA-256 digest")
    if str(semantic.get("structured_output_sha256", "")).lower() != structured_hash:
        errors.append("semantic_validation.structured_output_sha256 does not match structured_output")

    overlay = parsed["overlay_evidence"]
    for name in ("frame_id", "renderer", "overlay_fields"):
        if overlay.get(name) in (None, "", [], {}):
            errors.append(f"overlay_evidence.{name} is required")
    if str(overlay.get("structured_output_sha256", "")).lower() != structured_hash:
        errors.append("overlay_evidence.structured_output_sha256 does not match structured_output")
    if str(overlay.get("rendered_frame_sha256", "")).lower() != rendered_hash:
        errors.append("overlay_evidence.rendered_frame_sha256 does not match rendered_output")

    visual = parsed["visual_validation"]
    if str(visual.get("status", "")).upper() != "RENDERED_PASS":
        errors.append("visual_validation.status must be RENDERED_PASS")
    if str(visual.get("visual_status", "")).upper() != "PASS":
        errors.append("visual_validation.visual_status must be PASS")
    if str(visual.get("deepstream_status", "")).upper() != "PASS":
        errors.append("visual_validation.deepstream_status must be PASS")
    if str(visual.get("pipeline_visual_result", "")).upper() != "PASS":
        errors.append("visual_validation.pipeline_visual_result must be PASS")
    if visual.get("passed") is not True:
        errors.append("visual_validation.passed must be true")
    checks = visual.get("checks")
    if not isinstance(checks, Mapping):
        errors.append("visual_validation.checks must be an object")
    else:
        failed = sorted(
            name
            for name in REQUIRED_VISUAL_CHECKS
            if not isinstance(checks.get(name), Mapping)
            or str(checks[name].get("status", "")).upper() != "PASS"
        )
        if failed:
            errors.append("visual_validation checks did not pass: " + ", ".join(failed))
    if str(visual.get("structured_output_sha256", "")).lower() != structured_hash:
        errors.append("visual_validation.structured_output_sha256 does not match structured_output")
    if str(visual.get("rendered_output_sha256", "")).lower() != rendered_hash:
        errors.append("visual_validation.rendered_output_sha256 does not match rendered_output")

    return errors


def validate(
    manifest_path: Path,
    *,
    status_path: Path,
    promoted_manifest_path: Path,
) -> tuple[int, dict[str, Any]]:
    absolute_status = Path(os.path.abspath(os.fspath(status_path)))
    absolute_promoted = Path(os.path.abspath(os.fspath(promoted_manifest_path)))
    try:
        source_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = manifest_from_dict(source_data, manifest_path=manifest_path)
    except (OSError, json.JSONDecodeError, ManifestError, ValueError) as exc:
        return 2, {"status": "BLOCKED", "errors": [f"invalid manifest: {exc}"]}

    contract = manifest.adapter_contract
    implementation = str(contract.get("implementation_status", "")).strip().lower()
    errors: list[str] = []
    if implementation not in {"implemented", "tested"}:
        errors.append(
            "adapter_contract.implementation_status must be implemented before readiness validation"
        )
    errors.extend(_adapter_static_errors(manifest))
    declared = contract.get("validation_evidence")
    if not isinstance(declared, Mapping):
        errors.append("adapter_contract.validation_evidence must be a JSON object")
        declared = {}

    evidence: dict[str, dict[str, Any]] = {}
    parsed_evidence: dict[str, Mapping[str, Any]] = {}
    evidence_digests: dict[str, str] = {}
    for field in ADAPTER_EVIDENCE_FIELDS:
        value = declared.get(field)
        if value in (None, "", [], {}):
            errors.append(f"missing validation evidence: {field}")
            continue
        try:
            path = _resolve(manifest_path, value)
        except (TypeError, ValueError):
            errors.append(f"validation evidence {field} must be a filesystem path")
            continue
        try:
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"validation evidence {field} is missing or empty: {path}")
                continue
            digest = _digest(path)
        except OSError as exc:
            errors.append(f"validation evidence {field} is unreadable: {exc}")
            continue
        if field in JSON_FIELDS:
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"validation evidence {field} is invalid JSON: {exc}")
                continue
            if not isinstance(parsed, Mapping) or not parsed:
                errors.append(f"validation evidence {field} must contain a JSON object")
                continue
            if field in GATE_FIELDS and not _pass_status(parsed):
                errors.append(f"validation gate {field} did not report PASS")
                continue
            parsed_evidence[field] = parsed
        evidence_digests[field] = digest
        evidence[field] = {
            "path": os.path.relpath(path, start=absolute_status.parent),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }

    if set(evidence_digests) == set(ADAPTER_EVIDENCE_FIELDS):
        errors.extend(
            _evidence_contract_errors(manifest, parsed_evidence, evidence_digests)
        )
    else:
        errors.append("evidence contract validation requires all eight readable artifacts")

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    validated_source_hash = str(
        parsed_evidence.get("deepstream_runtime", {}).get(
            "source_hash", manifest.source_hash
        )
    ).strip().lower()
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "BLOCKED" if errors else "PASS",
        "implementation_status": "implemented" if errors else "tested",
        "validation_status": "generated-untested" if errors else "tested",
        "task": manifest.task,
        "model_name": manifest.model_name,
        "source_hash": validated_source_hash,
        "validated_at": now,
        "evidence": evidence,
        "errors": errors,
    }
    if errors:
        return 4, status

    promoted = dict(source_data)
    promoted["source_hash"] = validated_source_hash
    promoted_contract = dict(contract)
    promoted_contract["implementation_status"] = "tested"
    promoted_contract["validation_status"] = "tested"
    promoted_contract["readiness_status"] = os.path.relpath(
        absolute_status,
        start=absolute_promoted.parent,
    )
    promoted["adapter_contract"] = promoted_contract
    promoted_graph = dict(manifest.pipeline_graph)
    promoted_graph["status"] = "RENDERED_PASS"
    promoted_graph["evidence"] = {
        field: os.path.relpath(
            _resolve(manifest_path, declared[field]),
            start=absolute_promoted.parent,
        )
        for field in ADAPTER_EVIDENCE_FIELDS
    }
    promoted["pipeline_graph"] = promoted_graph

    _atomic_json(absolute_status, status)
    try:
        checked = manifest_from_dict(promoted, manifest_path=absolute_promoted)
        if checked.capability_status != "READY":
            raise ManifestError("; ".join(checked.capability_reasons))
    except (ManifestError, OSError, ValueError) as exc:
        absolute_status.unlink(missing_ok=True)
        return 4, {
            **status,
            "status": "BLOCKED",
            "implementation_status": "implemented",
            "validation_status": "generated-untested",
            "errors": [f"promotion manifest is not READY: {exc}"],
        }

    _atomic_json(absolute_promoted, promoted)
    return 0, status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    code, result = validate(
        args.manifest,
        status_path=args.status_output,
        promoted_manifest_path=args.manifest_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
