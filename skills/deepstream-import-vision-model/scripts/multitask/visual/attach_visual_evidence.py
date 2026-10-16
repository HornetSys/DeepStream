#!/usr/bin/env python3
"""Attach a visual I/O report to every fit-catalog artifact case."""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[3]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON beside the destination, then atomically replace it."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if path.exists():
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise

    print(f"Updated visual evidence catalog atomically: {path}", file=sys.stderr)


def attach(catalog_path: Path, report_path: Path) -> dict[str, Any]:
    catalog = load(catalog_path)
    report = load(report_path)
    rows = {row["id"]: row for row in report.get("entries", [])}
    for entry in catalog.get("entries", []):
        row = rows.get(entry["id"], {})
        entry["visual_validation"] = {
            "report": str(report_path),
            "case_image": row.get("case_image", ""),
            "status": row.get("result", "UNTESTED"),
            "input_status": row.get("input_visual_status", "UNTESTED"),
            "output_status": row.get("output_visual_status", "UNTESTED"),
            "fixture_kind": report.get("fixture_kind", ""),
            "fixture_sha256": row.get("fixture_sha256"),
            "executed_input_count": sum(item.get("executed") is True for item in row.get("inputs", [])),
            "executed_output_count": sum(item.get("executed") is True for item in row.get("outputs", [])),
            "backend": row.get("runtime_backend", "ONNXRuntime" if row.get("result") == "PASS" else None),
            "trt_raw_bindings": row.get("trt_raw_bindings", []),
            "constant_output_warnings": row.get("constant_output_warnings", []),
            "runtime_error": row.get("runtime_error"),
        }
    catalog["visual_validation_summary"] = {
        "report": str(report_path),
        "fixture_kind": report.get("fixture_kind", ""),
        "cases": report.get("counts", {}).get("cases", len(catalog.get("entries", []))),
        "pass": report.get("counts", {}).get("pass", 0),
        "fail": report.get("counts", {}).get("fail", 0),
        "untested": report.get("counts", {}).get("untested", 0),
        "input_visual_pass": report.get("counts", {}).get("input_visual_pass", 0),
        "output_visual_pass": report.get("counts", {}).get("output_visual_pass", 0),
        "backend_counts": report.get("backend_counts", {}),
        "semantic_limit": "raw tensor visuals are diagnostic; task decoder/parity is still required",
    }
    _atomic_write_json(catalog_path, catalog)
    return catalog["visual_validation_summary"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        catalog = args.catalog.resolve()
        if catalog == SKILL_ROOT or SKILL_ROOT in catalog.parents:
            raise ValueError("visual evidence catalog must be outside the installed skill")
        print(json.dumps(attach(catalog, args.report), indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"status": "BLOCKED", "errors": [str(exc)]}, indent=2),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
