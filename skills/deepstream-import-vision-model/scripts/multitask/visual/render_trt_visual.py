#!/usr/bin/env python3
"""Render real TensorRT raw bindings into an existing visual I/O report.

Use ``trtexec --dumpRawBindingsToFile`` with the deterministic fixture first,
then pass one or more ``CASE=RAW_DIR`` mappings to this command. This closes
the visual-output gap for graphs that ONNX Runtime cannot load because they
contain NVIDIA TensorRT plugins.
"""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def load_visual_module() -> Any:
    path = Path(__file__).with_name("visual_io_probe.py")
    spec = importlib.util.spec_from_file_location("visual_io_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_dtype(token: str) -> np.dtype:
    dtypes = {
        "fp32": np.dtype("float32"), "float": np.dtype("float32"),
        "fp16": np.dtype("float16"), "half": np.dtype("float16"),
        "int64": np.dtype("int64"), "int32": np.dtype("int32"),
        "uint8": np.dtype("uint8"), "int8": np.dtype("int8"),
        "float32": np.dtype("float32"), "float16": np.dtype("float16"),
        "float64": np.dtype("float64"), "bool": np.dtype("bool"),
        "uint16": np.dtype("uint16"), "uint32": np.dtype("uint32"),
        "uint64": np.dtype("uint64"), "int16": np.dtype("int16"),
    }
    try:
        return dtypes[token.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported TensorRT raw-binding dtype: {token}") from exc


def safe_name(name: str) -> str:
    allowed = set("._-,")
    return "".join(char if char.isalnum() or char in allowed else "_" for char in name)


def read_output(raw_dir: Path, name: str) -> tuple[np.ndarray, Path]:
    prefix = safe_name(name) + ".output."
    candidates = sorted(raw_dir.glob(prefix + "*.raw"))
    if not candidates:
        raise FileNotFoundError(f"no TensorRT raw output for {name} in {raw_dir}")
    path = candidates[0]
    stem = path.name[:-4]
    if not stem.startswith(prefix):
        raise ValueError(f"{path}: unexpected raw-binding filename format")
    tokens = stem[len(prefix):].split(".")
    if len(tokens) < 2:
        raise ValueError(f"{path}: missing dimensions or dtype token")
    try:
        dims = tuple(int(item) for item in tokens[:-1])
    except ValueError as exc:
        raise ValueError(f"{path}: invalid raw-binding dimensions") from exc
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"{path}: raw-binding dimensions must be positive")
    dtype = raw_dtype(tokens[-1])
    values = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(dims, dtype=np.int64))
    if values.size != expected:
        raise ValueError(f"{path}: {values.size} values, expected {expected} for {dims}")
    return values.reshape(dims), path


def _stage_attach(
    report_path: Path,
    inventory_path: Path,
    mappings: list[str],
    *,
    confirm_overwrite: bool,
    staging_dir: Path,
) -> tuple[dict[str, Any], list[tuple[Path, Path]]]:
    visual = load_visual_module()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory_by_id = {entry["id"]: entry for entry in inventory.get("entries", [])}
    rows = {row["id"]: row for row in report.get("entries", [])}
    replacements: list[tuple[Path, Path]] = []
    if not confirm_overwrite:
        raise PermissionError(
            "this command updates the supplied report, case images, and montage in place; "
            "rerun with --confirm-overwrite after reviewing those paths"
        )
    for mapping in mappings:
        if "=" not in mapping:
            raise ValueError(f"case mapping must be CASE=RAW_DIR: {mapping}")
        case_id, raw_dir_text = mapping.split("=", 1)
        entry = inventory_by_id.get(case_id)
        row = rows.get(case_id)
        if entry is None or row is None:
            raise ValueError(f"case {case_id!r} is not present in report and inventory")
        raw_dir = Path(raw_dir_text)
        input_tiles = []
        input_rows = []
        inputs = {}
        for index, spec in enumerate(entry.get("inspection", {}).get("inputs", [])):
            name = spec.get("name", f"input_{index}")
            value = visual.fixture_for_input(case_id, name, spec.get("shape", []), spec.get("dtype", 1))
            inputs[name] = value
            tile, rendered = visual.render_tile(name, value, entry.get("task", "generic"), "input")
            input_tiles.append(tile)
            input_rows.append({
                "name": name, "executed": True, "backend": "TensorRT",
                "fixture_shape": list(value.shape), "fixture_dtype": str(value.dtype),
                "fixture_visual": rendered,
            })
        output_tiles = []
        output_rows = []
        raw_paths = []
        for spec in entry.get("inspection", {}).get("outputs", []):
            name = spec["name"]
            value, raw_path = read_output(raw_dir, name)
            tile, rendered = visual.render_tile(name, value, entry.get("task", "generic"), "output")
            output_tiles.append(tile)
            output_rows.append({
                "name": name, "executed": True, "backend": "TensorRT",
                "runtime_visual": {"name": name, **rendered},
            })
            raw_paths.append(str(raw_path))
        row.update({
            "input_visual_status": "PASS" if input_rows else "UNTESTED",
            "output_visual_status": "PASS" if output_rows and all(item["runtime_visual"]["stats"]["finite"] for item in output_rows) else "FAIL",
            "result": "PASS" if input_rows and output_rows and all(item["runtime_visual"]["stats"]["finite"] for item in output_rows) else "FAIL",
            "inputs": input_rows, "outputs": output_rows,
            "fixture_sha256": visual.fixture_hash(inputs) if inputs else None,
            "runtime_backend": "TensorRT", "trt_raw_bindings": raw_paths,
            "runtime_error": None,
            "constant_output_warnings": [
                item["name"] for item in output_rows
                if item["runtime_visual"]["stats"]["min"] == item["runtime_visual"]["stats"]["max"]
            ],
        })
        image = visual._case_canvas(entry, input_tiles, output_tiles, row["result"], "TensorRT raw bindings rendered")
        image_path = Path(row["case_image"])
        staged_image = staging_dir / f"case-{len(replacements)}.png"
        image.save(staged_image)
        replacements.append((staged_image, image_path))
    report["entries"] = [rows[entry["id"]] for entry in inventory.get("entries", [])]
    report["counts"] = {
        "cases": len(report["entries"]),
        "pass": sum(row["result"] == "PASS" for row in report["entries"]),
        "fail": sum(row["result"] == "FAIL" for row in report["entries"]),
        "untested": sum(row["result"] == "UNTESTED" for row in report["entries"]),
        "input_visual_pass": sum(row["input_visual_status"] == "PASS" for row in report["entries"]),
        "output_visual_pass": sum(row["output_visual_status"] == "PASS" for row in report["entries"]),
    }
    report["backend_counts"] = {
        "onnxruntime": sum(row.get("runtime_backend") in (None, "ONNXRuntime") and row["result"] == "PASS" for row in report["entries"]),
        "tensorrt": sum(row.get("runtime_backend") == "TensorRT" and row["result"] == "PASS" for row in report["entries"]),
    }
    staged_report = staging_dir / "report.json"
    staged_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staged_images = {str(target): str(staged) for staged, target in replacements}
    montage_rows = [{**row, "case_image": staged_images.get(row["case_image"], row["case_image"])}
                    for row in report["entries"]]
    staged_montage = staging_dir / "montage.png"
    visual._make_montage(montage_rows, staged_montage)
    replacements.extend([(staged_montage, report_path.parent / "visual-montage.png"),
                         (staged_report, report_path)])
    return report["counts"], replacements


def attach(
    report_path: Path,
    inventory_path: Path,
    mappings: list[str],
    *,
    confirm_overwrite: bool,
) -> dict[str, Any]:
    """Validate and render every mapping before replacing any existing artifact."""
    with tempfile.TemporaryDirectory(prefix="trt-visual-", dir=report_path.parent) as td:
        staging_dir = Path(td)
        counts, replacements = _stage_attach(
            report_path, inventory_path, mappings,
            confirm_overwrite=confirm_overwrite, staging_dir=staging_dir,
        )
        backups = []
        for index, (_, target) in enumerate(replacements):
            backup = staging_dir / f"backup-{index}" if target.exists() else None
            if backup is not None:
                shutil.copy2(target, backup)
            backups.append((target, backup))
        committed = 0
        try:
            for staged, target in replacements:
                os.replace(staged, target)
                committed += 1
        except OSError:
            for target, backup in reversed(backups[:committed]):
                if backup is None:
                    target.unlink()
                else:
                    os.replace(backup, target)
            raise
        return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--case", action="append", required=True, help="CASE=RAW_DIR; repeat for each TensorRT raw-binding directory")
    parser.add_argument(
        "--confirm-overwrite",
        action="store_true",
        help="authorize replacing the report's case images, montage, and report JSON",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            attach(
                args.report,
                args.inventory,
                args.case,
                confirm_overwrite=args.confirm_overwrite,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
