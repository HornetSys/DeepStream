#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a model format/task into a validated, machine-readable plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_pipeline import ManifestError, route_model
from task_registry import SUPPORTED_TASK_ALIASES


INVALID_INPUT_EXIT = 2
UNSUPPORTED_EXIT = 3
BLOCKED_EXIT = 4


def _write(payload: str, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        print(payload, end="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="model file or model directory")
    parser.add_argument(
        "--task",
        default="auto",
        help="auto or one of: " + ", ".join(sorted(SUPPORTED_TASK_ALIASES)),
    )
    parser.add_argument("--metadata", type=Path, help="optional JSON object with task/input/preprocess metadata")
    parser.add_argument("--output", type=Path, help="write JSON plan to this path instead of stdout")
    args = parser.parse_args()
    try:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else None
        if metadata is not None and not isinstance(metadata, dict):
            raise ManifestError("--metadata must contain a JSON object")
        requested_task = str(args.task).strip().lower().replace("_", "-")
        if requested_task == "auto" and metadata:
            requested_task = str(metadata.get("task", "auto")).strip().lower().replace("_", "-")
        if requested_task not in {"auto", *SUPPORTED_TASK_ALIASES}:
            payload = json.dumps(
                {
                    "status": "unsupported",
                    "capability_status": "BLOCKED",
                    "requested_task": requested_task,
                    "reason": "task is outside the five-category integration scope",
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
            _write(payload, args.output)
            return UNSUPPORTED_EXIT
        emitted_path = args.output or args.metadata
        manifest_dir = emitted_path.absolute().parent if emitted_path else None
        manifest = route_model(
            args.source,
            task=args.task,
            metadata=metadata,
            manifest_dir=manifest_dir,
        )
        payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        _write(payload, args.output)
    except (OSError, ValueError, ManifestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return INVALID_INPUT_EXIT
    if manifest.status == "unsupported":
        return UNSUPPORTED_EXIT
    if (
        manifest.status != "ready-for-validation"
        or manifest.capability_status in {"BLOCKED", "ADAPTER_REQUIRED"}
    ):
        return BLOCKED_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
