#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Emit the selected conversion/build commands without executing them."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from model_pipeline import EXPERIMENTAL_FORMATS, ManifestError, manifest_from_dict, validate_backend_manifest


def _script(relative: str) -> str:
    """Use the actual source/installed skill root, independent of the caller's cwd."""
    return shlex.quote(str(Path(__file__).resolve().parents[1] / relative))


def _shape_specs(manifest, batch_size: int) -> tuple[str, str, str] | None:
    if len(manifest.input_shape) != 4 or manifest.layout != "NCHW":
        raise ManifestError("command emission requires a 4D NCHW image contract")
    if batch_size < 1:
        raise ManifestError("batch size must be a positive integer")
    profiles = (manifest.min_shape, manifest.opt_shape, manifest.max_shape)
    has_dynamic_dim = any(isinstance(dim, str) for dim in manifest.input_shape)
    if any(profiles) and not all(profiles):
        raise ManifestError("min_shape, opt_shape, and max_shape must be supplied together")
    if all(profiles):
        if not has_dynamic_dim:
            raise ManifestError("static models must not use explicit TensorRT shape profiles")
        return tuple("x".join(str(dim) for dim in profile) for profile in profiles)
    dims = list(manifest.input_shape)
    if not all(isinstance(dim, int) for dim in dims):
        raise ManifestError("dynamic dimensions require explicit min/opt/max shapes")
    if dims[0] != batch_size:
        raise ManifestError(
            f"batch size {batch_size} does not match the static input batch dimension {dims[0]}"
        )
    return None


def _precision_flags(precision: str) -> str:
    return {
        "fp16": " --fp16",
        "fp16-no-tf32": " --fp16 --noTF32",
        "fp32": " --noTF32",
        "int8": " --int8",
    }[precision]


def _precision_engine_path(engine_path: str, precision: str, *, primary: bool) -> str:
    if primary:
        return engine_path
    path = Path(engine_path)
    suffix = path.suffix or ".engine"
    return str(path.with_name(f"{path.stem}.{precision}{suffix}"))


def _precision_build_attempts(
    manifest, onnx_source: str, engine_path: str, batch_size: int
) -> list[dict[str, str]]:
    shape_specs = _shape_specs(manifest, batch_size)
    attempts: list[dict[str, str]] = []
    for index, precision in enumerate(manifest.precision_policy):
        target = _precision_engine_path(engine_path, precision, primary=index == 0)
        command = f"trtexec --onnx={shlex.quote(onnx_source)} --saveEngine={shlex.quote(target)}"
        if shape_specs:
            minimum, optimum, maximum = shape_specs
            input_name = shlex.quote(manifest.input_name)
            command += (
                f" --minShapes={input_name}:{minimum}"
                f" --optShapes={input_name}:{optimum}"
                f" --maxShapes={input_name}:{maximum}"
            )
        command += _precision_flags(precision)
        attempts.append({
            "precision": precision,
            "engine": target,
            "command": command,
            "run_when": "always" if index == 0 else "only after the previous precision fails parity",
        })
    return attempts


def _onnx_validation_and_build_commands(manifest, onnx_source: str, engine: str, batch_size: int) -> list[str]:
    """Emit the common post-export ONNX validation and TensorRT build stages."""

    shape_specs = _shape_specs(manifest, batch_size)
    inline = "import onnx,sys; m=onnx.load(sys.argv[1]); onnx.checker.check_model(m); onnx.shape_inference.infer_shapes(m)"
    trtexec = f"trtexec --onnx={onnx_source} --saveEngine={engine}"
    if shape_specs:
        minimum, optimum, maximum = shape_specs
        input_name = shlex.quote(manifest.input_name)
        trtexec += (
            f" --minShapes={input_name}:{minimum}"
            f" --optShapes={input_name}:{optimum}"
            f" --maxShapes={input_name}:{maximum}"
        )
    return [
        f"python3 -c {shlex.quote(inline)} {onnx_source}",
        "# Run source/reference -> ONNX Runtime parity on the recorded fixed fixtures before TensorRT.",
        trtexec + _precision_flags(manifest.precision_policy[0]),
        "# Run TensorRT -> reference parity on the same fixed fixtures before DeepStream config generation.",
    ]


def _wrapper_export_command(manifest, source: str, onnx_out: str, input_name: str) -> str:
    if not manifest.export_wrapper:
        raise ManifestError(
            "this model requires metadata.export_wrapper pointing to a model-specific ONNX export script; "
            "the skill does not invent architecture-specific loading or forward semantics"
        )
    wrapper = shlex.quote(manifest.export_wrapper)
    return (
        f"python3 {wrapper} --source {source} --output {onnx_out} "
        f"--input-name {input_name} --opset 17  # model-specific torch.onnx.export wrapper"
    )


def _supplied_engine_validation_command(manifest, source: str, manifest_path: str) -> str:
    """Emit only the approval-gated validator for a supplied TensorRT engine."""

    approval = manifest.execution_approval
    required = ("approver", "approved_at", "trusted_root", "scope", "engine_provenance")
    missing = [field for field in required if not str(approval.get(field, "")).strip()]
    if approval.get("approved") is not True or missing:
        detail = ", ".join(missing) or "approved=true"
        raise ManifestError(
            "supplied TensorRT engines require explicit execution_approval; missing " + detail
        )
    if approval["scope"] != "deserialize-supplied-tensorrt-engine":
        raise ManifestError(
            "execution_approval.scope must be 'deserialize-supplied-tensorrt-engine'"
        )
    validator = _script("validation/validate_artifact.py")
    return (
        f"python3 {validator} {shlex.quote(manifest_path)} --engine {source} "
        f"--run-trtexec --trusted-root {shlex.quote(str(approval['trusted_root']))} "
        f"--engine-provenance {shlex.quote(str(approval['engine_provenance']))}"
    )


def _implemented_adapter_config_command(
    manifest_path: str,
    output_dir: str,
    engine_path: str,
    batch_size: int,
) -> str:
    """Bind generated DeepStream configs to the engine that passed parity."""

    return (
        f"python3 {_script('deepstream/generate_configs.py')} {shlex.quote(manifest_path)} "
        f"{shlex.quote(f'{output_dir}/candidate-config')} "
        f"--engine {shlex.quote(engine_path)} --batch-size {batch_size} "
        "--allow-implemented-adapter"
    )


def emit(
    manifest,
    *,
    output_dir: str,
    engine_path: str,
    batch_size: int,
    manifest_path: str = "<MANIFEST_JSON>",
) -> dict[str, object]:
    if manifest.status == "unsupported":
        raise ManifestError("cannot emit commands for an unsupported format")
    if manifest.task == "detection":
        raise ManifestError(
            "detection uses this skill's native parser/KITTI workflow; "
            "emit_commands.py handles only classification, segmentation, pose, and depth"
        )
    if manifest.status != "ready-for-validation":
        raise ManifestError(f"cannot emit commands while manifest status is {manifest.status!r}")
    if manifest.source_format in EXPERIMENTAL_FORMATS and not manifest.experimental_reviewed:
        raise ManifestError(
            "experimental conversion is blocked until operator coverage and numerical parity are reviewed; "
            "set metadata.experimental_reviewed=true only after that review"
        )
    validate_backend_manifest(manifest)
    adapter_commands: list[str] = []
    implementation_status = str(
        manifest.adapter_contract.get("implementation_status", "")
    ).strip().lower()
    if manifest.capability_status == "ADAPTER_REQUIRED":
        quoted_manifest = shlex.quote(manifest_path)
        adapter_dir = shlex.quote(f"{output_dir}/adapter")
        if not implementation_status:
            adapter_commands.append(
                f"python3 {_script('deepstream/generate_adapter.py')} {quoted_manifest} {adapter_dir}"
            )
        elif implementation_status == "generated":
            adapter_commands.append(
                "# Implement every generated adapter hook from the supplied tensor and task contracts; "
                "then set implementation_status=implemented."
            )
        elif implementation_status == "implemented":
            if manifest.source_format == "tensorrt":
                adapter_commands.append(
                    _implemented_adapter_config_command(
                        manifest_path,
                        output_dir,
                        manifest.source,
                        batch_size,
                    )
                )
            else:
                adapter_commands.append(
                    "# After precision parity selects a PASS attempt, run that "
                    "attempt's config_command; never configure a failed engine."
                )
        else:
            raise ManifestError(
                "adapter claims tested but is still ADAPTER_REQUIRED: "
                + "; ".join(manifest.capability_reasons)
            )
    source = shlex.quote(manifest.source)
    engine = shlex.quote(engine_path)
    onnx_out = shlex.quote(f"{output_dir}/model.onnx")
    input_name = shlex.quote(manifest.input_name)
    commands: list[str] = []
    fallback_commands: list[str] = []
    fallback_requirement: str | None = None
    method: str
    if manifest.source_format == "tensorrt":
        method = "validate-approved-tensorrt-engine"
        commands.append(
            _supplied_engine_validation_command(manifest, source, manifest_path)
        )
    elif manifest.source_format == "onnx":
        method = "onnx-validate-then-trtexec"
        commands.extend(_onnx_validation_and_build_commands(manifest, source, engine, batch_size))
    elif manifest.source_format in {"pytorch", "safetensors", "torchscript"}:
        method = "torch-wrapper-export"
        commands.append(_wrapper_export_command(manifest, source, onnx_out, input_name))
        commands.extend(
            _onnx_validation_and_build_commands(manifest, onnx_out, engine, batch_size)
        )
    elif manifest.source_format == "tflite":
        method = "experimental-tf2onnx-tflite"
        commands.append(f"python3 -m tf2onnx.convert --tflite {source} --output {onnx_out} --opset 17")
        commands.extend(_onnx_validation_and_build_commands(manifest, onnx_out, engine, batch_size))
    elif manifest.source_format == "tensorflow":
        # route_model maps SavedModel dirs, .h5, and .keras to this format; tf2onnx needs different flags.
        if Path(manifest.source).suffix.lower() in {".h5", ".keras"}:
            method = "experimental-tf2onnx-keras"
            commands.append(f"python3 -m tf2onnx.convert --keras {source} --output {onnx_out} --opset 17")
        else:
            method = "experimental-tf2onnx-savedmodel"
            commands.append(f"python3 -m tf2onnx.convert --saved-model {source} --output {onnx_out} --opset 17")
        commands.extend(_onnx_validation_and_build_commands(manifest, onnx_out, engine, batch_size))
    else:
        raise ManifestError(f"no conversion command for source format {manifest.source_format}")
    precision_onnx = None
    if manifest.source_format == "onnx":
        precision_onnx = manifest.source
    elif manifest.source_format in {"pytorch", "safetensors", "torchscript", "tflite", "tensorflow"}:
        precision_onnx = f"{output_dir}/model.onnx"
    precision_attempts = (
        _precision_build_attempts(manifest, precision_onnx, engine_path, batch_size)
        if precision_onnx else []
    )
    if manifest.capability_status == "ADAPTER_REQUIRED" and implementation_status == "implemented":
        for attempt in precision_attempts:
            attempt["config_command"] = _implemented_adapter_config_command(
                manifest_path,
                output_dir,
                attempt["engine"],
                batch_size,
            )
    return {
        "method": method,
        "route_status": manifest.capability_status,
        "adapter_commands": adapter_commands,
        "production_pipeline_ready": manifest.capability_status == "READY",
        "commands": commands,
        "fallback_commands": fallback_commands,
        "fallback_requirement": fallback_requirement,
        "validation": list(manifest.validation),
        "parity_thresholds": dict(manifest.parity_thresholds),
        "precision_policy": list(manifest.precision_policy),
        "precision_attempts": precision_attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", default="model/export")
    parser.add_argument("--engine", default="model/model.engine")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    try:
        manifest = manifest_from_dict(
            json.loads(args.manifest.read_text(encoding="utf-8")),
            manifest_path=args.manifest,
        )
        result = emit(
            manifest,
            output_dir=args.output_dir,
            engine_path=args.engine,
            batch_size=args.batch_size,
            manifest_path=str(args.manifest),
        )
    except (OSError, ValueError, ManifestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
