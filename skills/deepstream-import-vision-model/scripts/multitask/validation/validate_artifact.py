#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run the static artifact gates and optional TensorRT smoke validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(SCRIPT_DIR))
from model_pipeline import (  # noqa: E402
    ManifestError,
    inspect_onnx_contract,
    manifest_from_dict,
    validate_onnx_manifest_contract,
)


def _status(ok: bool, detail: str, *, negative: str = "FAIL") -> dict[str, str]:
    return {"status": "PASS" if ok else negative, "detail": detail}


def _finish(result: dict[str, object]) -> dict[str, object]:
    """Every exit path must return the same schema: manifest, gates, and status."""

    gates = list(result["gates"].values())
    statuses = {item["status"] for item in gates}
    if "FAIL" in statuses:
        result["status"] = "FAIL"
    elif "BLOCKED" in statuses:
        result["status"] = "BLOCKED"
    elif "UNTESTED" in statuses:
        result["status"] = "UNTESTED"
    else:
        result["status"] = "PASS" if gates else "UNTESTED"
    return result


def _first_symlink_component(path: Path) -> Path | None:
    """Return the first symbolic-link component without resolving the path."""

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return current
    return None


def _trusted_engine(engine_path: Path, trusted_root: Path) -> Path:
    root_input = Path(os.path.abspath(os.fspath(trusted_root)))
    engine_input = Path(os.path.abspath(os.fspath(engine_path)))
    symlink = _first_symlink_component(root_input)
    if symlink is not None:
        raise ValueError(f"trusted root contains a symbolic-link component: {symlink}")
    symlink = _first_symlink_component(engine_input)
    if symlink is not None:
        raise ValueError(f"TensorRT engine path contains a symbolic-link component: {symlink}")
    root = root_input.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"trusted root must be a directory: {root}")
    engine = engine_input.resolve(strict=True)
    try:
        engine.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"TensorRT engine must resolve inside trusted root {root}") from exc
    if not engine.is_file():
        raise ValueError("TensorRT engine must be a regular file")
    if engine.suffix.lower() not in {".engine", ".plan"}:
        raise ValueError("TensorRT engine must use an .engine or .plan suffix")
    return engine


def _supplied_engine_approval_error(manifest, trusted_root: Path) -> str:
    """Validate caller-recorded approval before deserializing a supplied plan."""

    approval = manifest.execution_approval
    if approval.get("approved") is not True:
        return "execution_approval.approved=true is required for a supplied TensorRT engine"
    for field in ("approver", "approved_at", "trusted_root", "scope"):
        if not str(approval.get(field, "")).strip():
            return f"execution_approval.{field} is required for a supplied TensorRT engine"
    if approval["scope"] != "deserialize-supplied-tensorrt-engine":
        return (
            "execution_approval.scope must be "
            "'deserialize-supplied-tensorrt-engine'"
        )
    try:
        approved_at = datetime.fromisoformat(
            str(approval["approved_at"]).strip().replace("Z", "+00:00")
        )
    except ValueError:
        return "execution_approval.approved_at must be an ISO-8601 timestamp"
    if approved_at.tzinfo is None:
        return "execution_approval.approved_at must include a timezone"
    try:
        declared_value = Path(os.fspath(approval["trusted_root"]))
    except TypeError:
        return "execution_approval.trusted_root must be an absolute path string"
    if not declared_value.is_absolute():
        return "execution_approval.trusted_root must be an absolute path"
    declared_root = Path(os.path.abspath(os.fspath(declared_value)))
    requested_root = Path(os.path.abspath(os.fspath(trusted_root)))
    if declared_root != requested_root:
        return "execution_approval.trusted_root must match --trusted-root exactly"
    return ""


def validate(
    manifest_path: Path,
    *,
    onnx_path: Path | None,
    engine_path: Path | None,
    run_trtexec: bool,
    trusted_root: Path | None = None,
    engine_provenance: str = "",
    timeout: float = 300.0,
) -> dict[str, object]:
    result: dict[str, object] = {"manifest": str(manifest_path), "gates": {}}
    try:
        manifest = manifest_from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            manifest_path=manifest_path,
        )
    except (OSError, ValueError, ManifestError) as exc:
        result["gates"]["manifest"] = _status(False, str(exc))
        return _finish(result)
    result["gates"]["manifest"] = _status(True, f"{manifest.task}/{manifest.source_format}")
    result["route_status"] = manifest.capability_status

    source = Path(manifest.source)
    if not source.is_absolute():
        source = manifest_path.resolve().parent / source
    result["gates"]["source"] = _status(
        source.exists(), str(source), negative="BLOCKED"
    )
    if manifest.status == "unsupported":
        result["gates"]["format"] = _status(
            False, "manifest marks this format unsupported", negative="BLOCKED"
        )
        return _finish(result)
    implementation_status = str(
        manifest.adapter_contract.get("implementation_status", "")
    ).strip().lower()
    implemented_adapter = (
        manifest.capability_status == "ADAPTER_REQUIRED"
        and implementation_status == "implemented"
    )
    backend_ready_adapter = implemented_adapter and bool(manifest.capability_reasons) and all(
        reason.startswith("adapter implementation_status must be tested")
        for reason in manifest.capability_reasons
    )
    if manifest.capability_status != "READY" and not backend_ready_adapter:
        detail = "; ".join(manifest.capability_reasons) or "route is not executable by this skill"
        result["gates"]["capability"] = _status(False, detail, negative="BLOCKED")
        return _finish(result)
    if backend_ready_adapter:
        result["gates"]["capability"] = _status(
            False,
            "implemented adapter may run backend artifact checks; production route remains ADAPTER_REQUIRED",
            negative="UNTESTED",
        )

    if onnx_path:
        if not onnx_path.exists():
            result["gates"]["onnx-file"] = _status(False, f"missing ONNX: {onnx_path}", negative="BLOCKED")
        elif importlib.util.find_spec("onnx") is None:
            result["gates"]["onnx"] = _status(False, "onnx Python package is not installed", negative="BLOCKED")
        else:
            try:
                contract = inspect_onnx_contract(onnx_path)
                validate_onnx_manifest_contract(manifest, contract)
                result["gates"]["onnx"] = _status(
                    True,
                    f"checked {contract['node_count']} nodes and matched manifest tensor contracts",
                )
            except (ManifestError, OSError, ValueError) as exc:
                result["gates"]["onnx"] = _status(False, str(exc))

    if engine_path:
        if not engine_path.exists():
            result["gates"]["engine-file"] = _status(False, f"missing TensorRT engine: {engine_path}", negative="BLOCKED")
        elif not run_trtexec:
            result["gates"]["engine"] = _status(
                False,
                "engine exists but was not deserialized or verified; pass --run-trtexec "
                "with trusted-root and provenance",
                negative="UNTESTED",
            )
        elif trusted_root is None:
            result["gates"]["engine"] = _status(
                False,
                "--trusted-root is required before TensorRT engine deserialization",
                negative="BLOCKED",
            )
        elif not engine_provenance.strip():
            result["gates"]["engine"] = _status(
                False,
                "--engine-provenance is required before TensorRT engine deserialization",
                negative="BLOCKED",
            )
        elif (
            manifest.source_format == "tensorrt"
            and (approval_error := _supplied_engine_approval_error(manifest, trusted_root))
        ):
            result["gates"]["engine-approval"] = _status(
                False, approval_error, negative="BLOCKED"
            )
        elif (trtexec_path := shutil.which("trtexec")) is None:
            result["gates"]["engine"] = _status(False, "trtexec is not installed", negative="BLOCKED")
        else:
            try:
                trusted_engine = _trusted_engine(engine_path, trusted_root)
                with tempfile.TemporaryFile() as log:
                    completed = subprocess.run(
                        [trtexec_path, f"--loadEngine={trusted_engine}", "--skipInference"],
                        cwd=trusted_engine.parent,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=timeout,
                        start_new_session=True,
                    )
                    end = log.tell()
                    log.seek(max(0, end - 4000))
                    detail = log.read().decode("utf-8", errors="replace")
            except (OSError, ValueError) as exc:
                result["gates"]["engine"] = _status(False, str(exc))
            except subprocess.TimeoutExpired:
                result["gates"]["engine"] = _status(False, f"trtexec smoke test exceeded {timeout:g}s")
            else:
                result["engine_provenance"] = engine_provenance.strip()
                if manifest.source_format == "tensorrt":
                    result["execution_approval"] = {
                        field: manifest.execution_approval[field]
                        for field in ("approver", "approved_at", "trusted_root", "scope")
                    }
                result["gates"]["engine"] = _status(completed.returncode == 0, detail)

    return _finish(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--run-trtexec", action="store_true")
    parser.add_argument(
        "--trusted-root",
        type=Path,
        help="caller-approved canonical model directory required for engine deserialization",
    )
    parser.add_argument(
        "--engine-provenance",
        default="",
        help="required source/build provenance record when --run-trtexec deserializes an engine",
    )
    parser.add_argument("--trtexec-timeout", type=float, default=300.0, help="seconds before the trtexec smoke test is aborted")
    args = parser.parse_args()
    if args.trtexec_timeout <= 0:
        parser.error("--trtexec-timeout must be positive")
    result = validate(
        args.manifest,
        onnx_path=args.onnx,
        engine_path=args.engine,
        run_trtexec=args.run_trtexec,
        trusted_root=args.trusted_root,
        engine_provenance=args.engine_provenance,
        timeout=args.trtexec_timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
