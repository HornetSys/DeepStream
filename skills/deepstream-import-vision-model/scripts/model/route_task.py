#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Route a supported visual model into the unified DeepStream import workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[2]
MULTITASK_MODEL_DIR = SKILL_ROOT / "scripts" / "multitask" / "model"
sys.path.insert(0, str(MULTITASK_MODEL_DIR))

from model_pipeline import ManifestError, route_model  # noqa: E402
from task_registry import (  # noqa: E402
    CATEGORY_ALIASES,
    SUPPORTED_CATEGORIES,
    task_key,
)


PUBLIC_SKILL = "deepstream-import-vision-model"
BLOCKED_EXIT_CODE = 4


def _write(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _unsupported(requested: str) -> dict[str, Any]:
    return {
        "public_skill": PUBLIC_SKILL,
        "requested_task": requested,
        "status": "unsupported",
        "capability_status": "BLOCKED",
        "reason": "task is outside the five-category integration scope",
        "supported_categories": list(SUPPORTED_CATEGORIES),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local model file or directory")
    parser.add_argument("--task", default="auto", help="declared task/category")
    parser.add_argument("--metadata", type=Path, help="JSON task and tensor contract")
    parser.add_argument("--output", type=Path, help="write the route manifest here")
    args = parser.parse_args()

    try:
        requested = task_key(str(args.task))
        if requested != "auto" and CATEGORY_ALIASES.get(requested) not in SUPPORTED_CATEGORIES:
            _write(_unsupported(requested), args.output)
            return 3
        metadata = (
            json.loads(args.metadata.read_text(encoding="utf-8"))
            if args.metadata
            else {}
        )
        if not isinstance(metadata, dict):
            raise ManifestError("--metadata must contain a JSON object")
        emitted_path = args.output or args.metadata
        manifest_dir = emitted_path.absolute().parent if emitted_path else None

        if requested == "auto":
            declared = task_key(str(metadata.get("task", "")))
            if not declared:
                manifest = route_model(
                    args.source,
                    task="auto",
                    metadata=metadata,
                    manifest_dir=manifest_dir,
                ).to_dict()
                manifest.update({
                    "public_skill": PUBLIC_SKILL,
                    "category": "undetermined",
                    "supported_categories": list(SUPPORTED_CATEGORIES),
                    "status": "needs-task-identification",
                    "capability_status": "BLOCKED",
                    "capability_reasons": [
                        "declare one of the five supported visual categories before conversion"
                    ],
                })
                _write(manifest, args.output)
                return BLOCKED_EXIT_CODE
            requested = declared
        category = CATEGORY_ALIASES.get(requested)
        if category not in SUPPORTED_CATEGORIES:
            _write(_unsupported(requested or "auto"), args.output)
            return 3

        manifest = route_model(
            args.source,
            task=requested,
            metadata=metadata,
            manifest_dir=manifest_dir,
        ).to_dict()
        manifest["public_skill"] = PUBLIC_SKILL
        manifest["category"] = category
        manifest["supported_categories"] = list(SUPPORTED_CATEGORIES)

        if category == "object-detection":
            manifest["execution_family"] = "native-object-detection"
        else:
            manifest["execution_family"] = f"multitask-{category}"

        _write(manifest, args.output)
        if manifest.get("status") == "unsupported":
            return 3
        if manifest.get("capability_status") in {"BLOCKED", "ADAPTER_REQUIRED"}:
            return BLOCKED_EXIT_CODE
        return 0
    except (OSError, ValueError, ManifestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
