#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate a category-level semantic-oracle document before runtime testing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


TASK_RULES: dict[str, dict[str, Any]] = {
    "detection": {
        "metric": "map",
        "comparator": "gte",
        "limit": 0.0,
        "expected": ("boxes",),
    },
    "classification": {
        "metric": "top1_accuracy",
        "comparator": "gte",
        "limit": 1.0,
        "expected": ("class_id",),
    },
    "segmentation": {
        "metric": "mean_iou",
        "comparator": "gte",
        "limit": 0.99,
        "int8_limit": 0.98,
        "expected_any": ("mask", "mask_path"),
    },
    "pose": {
        "metric": "pck_oks_agreement",
        "comparator": "gte",
        "limit": 0.99,
        "expected": ("keypoints",),
    },
    "depth": {
        "metric": "aligned_rmse_fraction",
        "comparator": "lte",
        "limit": 0.01,
        "expected_any": ("depth", "depth_path"),
    },
}

TASK_ALIASES = {
    "detection": "detection",
    "object-detection": "detection",
    "classification": "classification",
    "image-classification": "classification",
    "segmentation": "segmentation",
    "semantic-segmentation": "segmentation",
    "pose": "pose",
    "2d-pose": "pose",
    "keypoints": "pose",
    "depth": "depth",
    "monocular-depth": "depth",
}


def _normalise_task(value: Any) -> str:
    key = str(value or "").strip().lower().replace("_", "-")
    return TASK_ALIASES.get(key, "")


def _timezone_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _artifact_error(root: Path, value: Any, field: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return f"{field} must be a non-empty path"
    candidate = Path(value)
    if candidate.is_absolute():
        return f"{field} must be relative to the oracle document"
    joined = root / candidate
    root_resolved = root.resolve()
    try:
        resolved = joined.resolve(strict=True)
    except OSError:
        return f"{field} does not resolve to a regular, non-symlink file: {value}"
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        return f"{field} escapes the oracle directory"
    if not resolved.is_file() or joined.is_symlink():
        return f"{field} does not resolve to a regular, non-symlink file: {value}"
    return None


def validate(data: Any, *, oracle_path: Path, required_task: str = "") -> list[str]:
    errors: list[str] = []
    if not isinstance(data, Mapping):
        return ["oracle document must be a JSON object"]
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    raw_task = str(data.get("task", "")).strip().lower()
    task = _normalise_task(raw_task)
    if not task:
        errors.append("task must be one of: " + ", ".join(sorted(TASK_ALIASES)))
        return errors
    normalised_required = _normalise_task(required_task)
    if required_task and not normalised_required:
        errors.append(f"required task {required_task!r} is unsupported")
    elif normalised_required and task != normalised_required:
        errors.append(
            f"task {raw_task!r} does not match required task {required_task!r}"
        )

    oracle_type = str(data.get("oracle_type", "")).strip().lower()
    if oracle_type not in {"ground-truth", "reviewed"}:
        errors.append("oracle_type must be ground-truth or reviewed")
    provenance = data.get("provenance")
    if not isinstance(provenance, Mapping) or not str(provenance.get("source", "")).strip():
        errors.append("provenance.source is required")
    elif oracle_type == "reviewed":
        if not str(provenance.get("reviewer", "")).strip():
            errors.append("provenance.reviewer is required for a reviewed oracle")
        if not _timezone_timestamp(provenance.get("reviewed_at")):
            errors.append("provenance.reviewed_at must be a timezone-bearing ISO-8601 timestamp")

    acceptance = data.get("acceptance")
    rule = TASK_RULES[task]
    if not isinstance(acceptance, Mapping):
        errors.append("acceptance must be a JSON object")
        acceptance = {}
    metric = str(acceptance.get("metric", "")).strip()
    comparator = str(acceptance.get("comparator", "")).strip().lower()
    if metric != rule["metric"]:
        errors.append(f"acceptance.metric must be {rule['metric']!r} for {task}")
    if comparator != rule["comparator"]:
        errors.append(f"acceptance.comparator must be {rule['comparator']!r} for {task}")
    threshold = acceptance.get("threshold")
    try:
        numeric_threshold = float(threshold)
    except (TypeError, ValueError):
        errors.append("acceptance.threshold must be numeric")
    else:
        if str(data.get("precision", "")).lower() == "int8":
            limit = float(rule.get("int8_limit", rule["limit"]))
        else:
            limit = float(rule["limit"])
        relaxed = numeric_threshold < limit if comparator == "gte" else numeric_threshold > limit
        if relaxed:
            errors.append(f"acceptance.threshold {numeric_threshold:g} relaxes the {task} minimum {limit:g}")
    minimum_samples = acceptance.get("minimum_samples")
    if not isinstance(minimum_samples, int) or isinstance(minimum_samples, bool) or minimum_samples < 1:
        errors.append("acceptance.minimum_samples must be an explicit positive integer")
        minimum_samples = 1

    samples = data.get("samples")
    if not isinstance(samples, list):
        errors.append("samples must be a JSON array")
        return errors
    if len(samples) < minimum_samples:
        errors.append(
            f"samples contains {len(samples)} entries; "
            f"acceptance.minimum_samples requires {minimum_samples}"
        )
    fixture_ids: set[str] = set()
    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, Mapping):
            errors.append(f"{prefix} must be a JSON object")
            continue
        fixture_id = str(sample.get("fixture_id", "")).strip()
        if not fixture_id:
            errors.append(f"{prefix}.fixture_id is required")
        elif fixture_id in fixture_ids:
            errors.append(f"{prefix}.fixture_id is duplicated: {fixture_id}")
        fixture_ids.add(fixture_id)
        media = sample.get("media")
        if not isinstance(media, Mapping) or not str(media.get("path", "")).strip():
            errors.append(f"{prefix}.media.path is required")
        expected = sample.get("expected")
        if not isinstance(expected, Mapping):
            errors.append(f"{prefix}.expected must be a JSON object")
            continue
        for field in rule.get("expected", ()):
            if field not in expected or expected.get(field) is None:
                errors.append(f"{prefix}.expected.{field} is required")
            elif field != "boxes" and expected.get(field) in ("", [], {}):
                errors.append(f"{prefix}.expected.{field} must not be empty")
        any_fields = rule.get("expected_any", ())
        if any_fields and not any(
            expected.get(field) not in (None, "", [], {}) for field in any_fields
        ):
            errors.append(f"{prefix}.expected requires one of: {', '.join(any_fields)}")
        for field in ("mask_path", "depth_path"):
            if field in expected:
                error = _artifact_error(oracle_path.parent, expected[field], f"{prefix}.expected.{field}")
                if error:
                    errors.append(error)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--task", choices=tuple(sorted(TASK_ALIASES)), default="")
    args = parser.parse_args()
    try:
        data = json.loads(args.oracle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "errors": [str(exc)]}, indent=2))
        return 2
    errors = validate(data, oracle_path=args.oracle, required_task=args.task)
    sample_count = (
        len(data.get("samples", []))
        if isinstance(data, Mapping) and isinstance(data.get("samples"), list)
        else 0
    )
    result = {
        "status": "PASS" if not errors else "BLOCKED",
        "task": _normalise_task(data.get("task")) if isinstance(data, Mapping) else None,
        "sample_count": sample_count,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
