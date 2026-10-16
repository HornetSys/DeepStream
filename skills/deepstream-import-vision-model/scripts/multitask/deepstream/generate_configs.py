#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Generate conservative nvinfer/nvdspreprocess config templates from a manifest."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "model"
SKILL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
from model_pipeline import (  # noqa: E402
    ADAPTER_EVIDENCE_FIELDS,
    TASK_CONTRACT_REQUIRED_FIELDS,
    ManifestError,
    manifest_from_dict,
    validate_backend_manifest,
    validate_execution_manifest,
)


TENSOR_DATA_TYPES = {
    "float32": 0,
    "uint8": 1,
    "int8": 2,
    "uint32": 3,
    "int32": 4,
    "int64": 5,
    "uint64": 6,
    "float16": 7,
}


def _resolved_image_shape(manifest) -> tuple[int, ...]:
    if len(manifest.input_shape) != 4:
        raise ManifestError("config generation requires a 4D image tensor; provide a task-specific config for non-image tensors")
    shape = manifest.opt_shape if manifest.opt_shape else manifest.input_shape
    if not all(isinstance(dim, int) for dim in shape):
        raise ManifestError("dynamic H/W requires explicit min/opt/max profiles before config generation")
    return tuple(int(dim) for dim in shape)


def _spatial(manifest) -> tuple[int, int]:
    shape = _resolved_image_shape(manifest)
    if manifest.layout == "NCHW":
        height, width = shape[2:4]
    elif manifest.layout == "NHWC":
        height, width = shape[1:3]
    else:
        raise ManifestError(f"layout {manifest.layout} is not a supported image layout")
    return height, width


def _expand_channels(values: tuple[float, ...], channels: int | None, name: str) -> tuple[float, ...]:
    if not values:
        return ()
    if channels is None:
        raise ManifestError(f"cannot expand {name}; input channel count is not known")
    if len(values) == 1:
        return values * channels
    if len(values) != channels:
        raise ManifestError(f"{name} must contain one value or exactly {channels} channel values")
    return values


def _normalization(manifest, channels: int | None = None) -> tuple[float, tuple[float, ...]]:
    """Return the DeepStream `(x - offset) * factor` pair for the manifest units.

    `preprocess_units='pixel'`      : mean/std are 0-255 pixel values, y = (x - mean) / std.
    `preprocess_units='normalized'` : mean/std apply after `scale`,   y = (x * scale - mean) / std.
    """

    if channels is None and len(manifest.input_shape) == 4:
        channel_index = 1 if manifest.layout == "NCHW" else 3
        channel_value = manifest.input_shape[channel_index]
        if isinstance(channel_value, int):
            channels = channel_value
    mean = _expand_channels(manifest.mean, channels, "mean")
    std = _expand_channels(manifest.std, channels, "std")
    if std:
        first_std = std[0]
        if not all(value == first_std for value in std):
            raise ManifestError(
                "per-channel std is not supported by nvinfer or stock libcustom2d_preprocess; "
                "use a custom preprocessing library or supply a single shared std"
            )
    if manifest.preprocess_units == "normalized":
        factor = manifest.scale / std[0] if std else manifest.scale
        offsets = tuple(value / manifest.scale for value in mean)
    else:
        factor = 1.0 / std[0] if std else manifest.scale
        offsets = tuple(mean)
    return factor, offsets


def _nvinfer(manifest, engine_path: str, onnx_path: str, batch_size: int) -> str:
    height, width = _spatial(manifest)
    shape = _resolved_image_shape(manifest)
    channels = shape[1] if manifest.layout == "NCHW" else shape[3]
    network_type = {"classification": 1, "detection": 0}.get(manifest.task, 100)
    color_format = {"RGB": 0, "BGR": 1, "GRAY": 2}[manifest.color_format]
    input_order = {"NCHW": 0, "NHWC": 1}[manifest.layout]
    reference_runner = str(
        manifest.adapter_contract.get("reference_pipeline_runner", "")
    ).strip()
    # The classification reference runner decodes the tensor itself before it
    # attaches model-derived display metadata. Native nvinfer classification
    # metadata is therefore insufficient even when postprocess is "native".
    raw_tensor_task = (
        manifest.postprocess != "native"
        or manifest.task not in {"classification", "detection"}
        or (manifest.task == "classification" and bool(reference_runner))
    )
    lines = [
        "[property]",
        "gpu-id=0",
        f"gie-unique-id={manifest.gie_unique_id}",
        f"batch-size={batch_size}",
        f"process-mode={1 if manifest.process_mode == 'primary' else 2}",
        f"network-type={network_type}",
        f"infer-dims={channels};{height};{width}",
        f"network-input-order={input_order}",
        f"model-color-format={color_format}",
    ]
    adapter_input_tensor = (
        manifest.adapter_contract.get("preprocess_strategy") == "adapter-input-tensor"
    )
    if manifest.nvdspreprocess or adapter_input_tensor:
        # nvdspreprocess or the adapter emits the normalized input tensor.
        lines.extend(["net-scale-factor=1.0", "input-tensor-from-meta=1"])
    else:
        factor, offsets = _normalization(manifest, channels)
        lines.append(f"net-scale-factor={factor:.12g}")
        if offsets:
            lines.append("offsets=" + ";".join(f"{value:.12g}" for value in offsets))
        lines.append(f"maintain-aspect-ratio={int(manifest.maintain_aspect_ratio)}")
        lines.append(f"symmetric-padding={int(manifest.symmetric_padding)}")
    if onnx_path:
        lines.append(f"onnx-file={onnx_path}")
    if engine_path:
        lines.append(f"model-engine-file={engine_path}")
    if manifest.labels:
        lines.append(f"labelfile-path={manifest.labels}")
    if manifest.process_mode == "secondary":
        if manifest.operate_on_gie_id is None:
            raise ManifestError("secondary nvinfer configuration requires operate_on_gie_id")
        lines.append(f"operate-on-gie-id={manifest.operate_on_gie_id}")
        if manifest.operate_on_class_ids:
            lines.append("operate-on-class-ids=" + ";".join(str(value) for value in manifest.operate_on_class_ids))
    if raw_tensor_task:
        lines.extend([
            "# The adapter/probe decodes named output tensors using the manifest task contract.",
            "# Tensor metadata is required even when native classification parsing is enabled.",
            "output-tensor-meta=1",
        ])
    lines.append("")
    return "\n".join(lines)


def _nvdspreprocess(manifest, batch_size: int) -> str:
    height, width = _spatial(manifest)
    shape = _resolved_image_shape(manifest)
    channels = shape[1] if manifest.layout == "NCHW" else shape[3]
    color = {"RGB": 0, "BGR": 1, "GRAY": 2}[manifest.color_format]
    network_input_order = {"NCHW": 0, "NHWC": 1}[manifest.layout]
    network_input_shape = (
        f"{batch_size};{channels};{height};{width}"
        if manifest.layout == "NCHW"
        else f"{batch_size};{height};{width};{channels}"
    )
    tensor_data_type = TENSOR_DATA_TYPES[manifest.input_dtype]
    expanded_mean = _expand_channels(manifest.mean, channels, "mean")
    expanded_std = _expand_channels(manifest.std, channels, "std")
    per_channel_std = bool(expanded_std) and len(set(expanded_std)) > 1
    if per_channel_std:
        if not expanded_mean:
            expanded_mean = (0.0,) * channels
        custom_lib = str(manifest.adapter_contract.get("preprocess_library", "")).strip()
        custom_function = str(manifest.adapter_contract.get("preprocess_function", "")).strip()
        if not custom_lib or not custom_function:
            raise ManifestError(
                "per-channel std requires adapter_contract.preprocess_library and "
                "adapter_contract.preprocess_function"
            )
        user_config_lines = [
            f"preprocess-units={manifest.preprocess_units}",
            f"input-scale={manifest.scale:.12g}",
            "channel-mean=" + ";".join(f"{value:.12g}" for value in expanded_mean),
            "channel-std=" + ";".join(f"{value:.12g}" for value in expanded_std),
        ]
    else:
        factor, offsets = _normalization(manifest, channels)
        custom_lib = "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so"
        custom_function = "CustomTensorPreparation"
        offsets_str = ";".join(f"{value:.12g}" for value in offsets) or ";".join("0" for _ in range(channels))
        user_config_lines = [
            f"pixel-normalization-factor={factor:.12g}",
            f"offsets={offsets_str}",
        ]
    property_lines = [
        "[property]",
        "enable=1",
        "gpu-id=0",
        "target-unique-ids=" + ";".join(str(value) for value in manifest.target_unique_ids),
        f"unique-id={manifest.preprocess_unique_id}",
        f"process-on-frame={1 if manifest.preprocess_scope == 'frame' else 0}",
        f"network-input-order={network_input_order}",
        f"network-input-shape={network_input_shape}",
        f"network-color-format={color}",
        f"processing-width={width}",
        f"processing-height={height}",
        f"tensor-data-type={tensor_data_type}",
        f"scaling-buf-pool-size={batch_size}",
        f"tensor-buf-pool-size={batch_size}",
        f"tensor-name={manifest.input_name}",
        "scaling-pool-memory-type=0",
        "scaling-pool-compute-hw=0",
        "scaling-filter=0",
        f"custom-lib-path={custom_lib}",
        f"custom-tensor-preparation-function={custom_function}",
        f"maintain-aspect-ratio={int(manifest.maintain_aspect_ratio)}",
        f"symmetric-padding={int(manifest.symmetric_padding)}",
        "[user-configs]",
        *user_config_lines,
        "[group-0]",
        "src-ids=-1",
    ]
    if manifest.preprocess_scope == "object":
        property_lines.insert(4, f"operate-on-gie-id={manifest.operate_on_gie_id}")
        if manifest.operate_on_class_ids:
            property_lines.append("operate-on-class-ids=" + ";".join(str(value) for value in manifest.operate_on_class_ids))
        property_lines.append(f"process-on-all-objects={0 if manifest.roi_params else 1}")
    else:
        property_lines.append(f"process-on-roi={1 if manifest.roi_params else 0}")
    for source_id, coordinates in sorted(manifest.roi_params.items(), key=lambda item: int(item[0])):
        property_lines.append(f"roi-params-src-{source_id}=" + ";".join(str(value) for value in coordinates))
    property_lines.append("")
    return "\n".join(property_lines)


def _postprocess_contract(manifest) -> dict[str, object]:
    assertions = {
        "segmentation": ["mask geometry", "class range", "mean IoU", "resize/alignment policy"],
        "pose": ["keypoint count/order", "visibility/confidence", "coordinate range", "skeleton topology", "PCK/OKS"],
        "depth": ["finite ratio", "depth units/scale", "invalid mask", "aligned RMSE"],
        "embedding": ["vector dimension", "finite ratio", "normalization", "cosine/L2 agreement"],
        "optical-flow": ["vector layout", "coordinate units", "validity/occlusion", "endpoint error"],
        "multimodal": ["adapter-owned schema", "per-output fixture parity"],
        "generic": ["adapter-owned schema", "per-output fixture parity"],
    }
    required_fields = list(TASK_CONTRACT_REQUIRED_FIELDS.get(manifest.task, ()))
    missing_fields = [field for field in required_fields if field not in manifest.task_contract]
    return {
        "task": manifest.task,
        "task_variant": manifest.task_variant or None,
        "network_type": 100,
        "output_tensor_meta": True,
        "output_names": list(manifest.output_names),
        "decoder": "custom-library-or-application-probe",
        "required_assertions": assertions.get(manifest.task, ["task-specific output parity"]),
        "required_fields": required_fields,
        "task_contract": dict(manifest.task_contract),
        "adapter_contract": dict(manifest.adapter_contract),
        "contract_status": "incomplete" if missing_fields else "complete",
        "parity_thresholds": dict(manifest.parity_thresholds),
    }


def _nvdspostprocess(manifest) -> str:
    if manifest.task == "classification":
        network_type = 1
        task_lines = [
            "  is-classifier: 1",
            "  classifier-type: generic",
            f"  labelfile-path: {manifest.labels or 'labels.txt'}",
            "  classifier-threshold: 0.0",
        ]
    elif manifest.task == "detection":
        network_type = 0
        task_lines = [
            "  num-detected-classes: 1",
            f"  labelfile-path: {manifest.labels or 'labels.txt'}",
        ]
    else:
        raise ManifestError(
            f"nvdspostprocess in DeepStream 9.1 is not supported for task '{manifest.task}'; "
            "use a custom postprocessor or application probe"
        )
    lines = [
        "%YAML:1.0",
        "# Declare output-blob-names and the postprocess library contract before running.",
        "property:",
        "  gpu-id: 0",
        f"  process-mode: {1 if manifest.process_mode == 'primary' else 2}",
        f"  gie-unique-id: {manifest.gie_unique_id}",
    ]
    if manifest.process_mode == "secondary":
        if manifest.operate_on_gie_id is None:
            raise ManifestError("secondary nvdspostprocess configuration requires operate_on_gie_id")
        lines.append(f"  operate-on-gie-id: {manifest.operate_on_gie_id}")
    lines.extend([
        f"  output-blob-names: {';'.join(manifest.output_names) or 'output_0'}",
        f"  network-type: {network_type}",
        *task_lines,
        "",
    ])
    return "\n".join(lines)



def _adapter_static_errors(manifest) -> list[str]:
    """Check generated hook references without importing or executing adapter code."""

    contract = manifest.adapter_contract
    required_hooks = contract.get("required_hooks")
    if not isinstance(required_hooks, list) or not required_hooks:
        return ["adapter_contract.required_hooks must be a non-empty list"]
    errors: list[str] = []
    parsed: dict[Path, ast.Module] = {}
    for hook in required_hooks:
        if not isinstance(hook, str) or not hook.strip():
            errors.append("adapter_contract.required_hooks contains an invalid name")
            continue
        value = contract.get(hook)
        if value in (None, "", [], {}):
            errors.append(f"adapter_contract.{hook} is unresolved")
            continue
        if hook == "preprocess_library":
            library = Path(str(value))
            if not library.is_absolute() or not library.is_file():
                errors.append("adapter_contract.preprocess_library must be an existing absolute file")
            continue
        target = str(value)
        if ":" not in target:
            continue
        file_name, symbol = target.rsplit(":", 1)
        source = Path(file_name)
        if not source.is_absolute() and manifest._manifest_dir:
            source = Path(manifest._manifest_dir) / source
        if source.suffix != ".py":
            continue
        if not source.is_file():
            errors.append(f"adapter hook file is missing: {source}")
            continue
        try:
            tree = parsed.setdefault(source, ast.parse(source.read_text(encoding="utf-8")))
        except (OSError, SyntaxError) as exc:
            errors.append(f"adapter hook file is invalid: {source}: {exc}")
            continue
        functions = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol
        ]
        if not functions:
            errors.append(f"adapter hook {symbol!r} is missing from {source}")
            continue
        function = functions[0]
        unresolved = any(
            isinstance(node, ast.Raise)
            and "NotImplemented" in ast.unparse(node.exc or ast.Constant(value=""))
            for node in ast.walk(function)
        )
        empty = all(
            isinstance(node, ast.Pass)
            or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
            for node in function.body
        )
        if unresolved or empty:
            errors.append(f"adapter hook {symbol!r} remains an unimplemented scaffold")
    return list(dict.fromkeys(errors))


def _validate_for_config_generation(manifest, allow_implemented_adapter: bool) -> bool:
    """Validate a production config or a narrowly scoped adapter test config."""

    if manifest.capability_status == "READY":
        validate_execution_manifest(manifest)
        return False
    if not allow_implemented_adapter:
        validate_execution_manifest(manifest)
    if manifest.capability_status != "ADAPTER_REQUIRED":
        raise ManifestError(
            f"candidate adapter configs require ADAPTER_REQUIRED, got {manifest.capability_status}"
        )
    implementation_status = str(
        manifest.adapter_contract.get("implementation_status", "")
    ).strip().lower()
    if implementation_status != "implemented":
        raise ManifestError(
            "--allow-implemented-adapter requires implementation_status=implemented; "
            "generated hooks are not executable"
        )
    static_errors = _adapter_static_errors(manifest)
    if static_errors:
        raise ManifestError(
            "adapter static checks failed: " + "; ".join(static_errors)
        )
    allowed_prefix = "adapter implementation_status must be tested"
    blockers = [
        reason for reason in manifest.capability_reasons
        if not reason.startswith(allowed_prefix)
    ]
    if blockers:
        raise ManifestError(
            "candidate config is blocked by unresolved adapter requirements: " + "; ".join(blockers)
        )
    validate_backend_manifest(manifest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--engine", default="")
    parser.add_argument("--onnx", default="")
    parser.add_argument(
        "--allow-implemented-adapter",
        action="store_true",
        help="emit candidate test configs for a complete implementation_status=implemented adapter",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.engine.strip() and not args.onnx.strip():
        parser.error("provide an explicit --engine or --onnx artifact")
    try:
        output_dir = args.output_dir.resolve()
        if output_dir == SKILL_ROOT or SKILL_ROOT in output_dir.parents:
            raise ManifestError("config output directory must be outside the installed skill")
        manifest = manifest_from_dict(
            json.loads(args.manifest.read_text(encoding="utf-8")),
            manifest_path=args.manifest,
        )
        if manifest.status != "ready-for-validation":
            raise ManifestError(f"cannot generate configs while manifest status is {manifest.status!r}")
        candidate_adapter = _validate_for_config_generation(
            manifest, args.allow_implemented_adapter
        )
        # Render every config before writing so a rejected manifest never leaves a partial set.
        configs = {"nvinfer.txt": _nvinfer(manifest, args.engine, args.onnx, args.batch_size)}
        configs["pipeline-plan.json"] = json.dumps(
            manifest.pipeline_graph, indent=2, sort_keys=True
        ) + "\n"
        if manifest.postprocess != "native" or manifest.task not in {"classification", "detection"}:
            configs["postprocess-contract.json"] = json.dumps(
                _postprocess_contract(manifest), indent=2, sort_keys=True
            ) + "\n"
        if manifest.nvdspreprocess:
            configs["nvdspreprocess.txt"] = _nvdspreprocess(manifest, args.batch_size)
        if manifest.nvdspostprocess:
            configs["nvdspostprocess.yml"] = _nvdspostprocess(manifest)
        if candidate_adapter:
            configs["adapter-validation-required.json"] = json.dumps(
                {
                    "status": "ADAPTER_REQUIRED",
                    "implementation_status": "implemented",
                    "production_ready": False,
                    "required_evidence": list(ADAPTER_EVIDENCE_FIELDS),
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        existing = [output_dir / name for name in configs if (output_dir / name).exists()]
        if existing:
            raise FileExistsError(
                "generated configuration destination already exists; choose a fresh output directory: "
                + ", ".join(str(path) for path in existing)
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in configs.items():
            destination = output_dir / name
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(content)
    except (OSError, ValueError, ManifestError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
