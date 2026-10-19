#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Routing and manifest validation for multitask model import.

The common path is pure stdlib. ONNX is imported lazily only when an ONNX
artifact must be inspected; torch, TensorRT, and Transformers are never loaded
by the dry-run router.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from task_registry import INTERNAL_TASKS, TASK_ALIASES, task_key


TASKS = INTERNAL_TASKS


FORMAT_BY_SUFFIX = {
    ".engine": "tensorrt",
    ".plan": "tensorrt",
    ".onnx": "onnx",
    ".safetensors": "safetensors",
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".ckpt": "pytorch",
    ".jit": "torchscript",
    ".tflite": "tflite",
    ".pb": "tensorflow",
    ".h5": "tensorflow",
    ".keras": "tensorflow",
    ".prototxt": "caffe-unsupported",
    ".caffemodel": "caffe-unsupported",
}

FIRST_CLASS_FORMATS = {"tensorrt", "onnx", "pytorch", "safetensors", "torchscript"}
EXPERIMENTAL_FORMATS = {"tflite", "tensorflow"}
UNSUPPORTED_FORMATS = {"caffe-unsupported"}

INPUT_DTYPE_ALIASES = {
    "float": "float32",
    "fp32": "float32",
    "float32": "float32",
    "half": "float16",
    "fp16": "float16",
    "float16": "float16",
    "uint8": "uint8",
    "int8": "int8",
    "uint32": "uint32",
    "int32": "int32",
    "int64": "int64",
    "uint64": "uint64",
}
SUPPORTED_INPUT_DTYPES = set(INPUT_DTYPE_ALIASES.values())


DEFAULT_PARITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "classification": {"top1_agreement": 1.0, "topk_set_agreement": 1.0},
    "segmentation": {"min_mean_iou": 0.99},
    "pose": {"min_pck_oks_agreement": 0.99},
    "depth": {"max_aligned_rmse_fraction": 0.01},
}

TASK_CONTRACT_REQUIRED_FIELDS = {
    "classification": ("class_count", "label_map", "score_semantics", "top_k", "threshold"),
    "segmentation": ("mask_layout", "class_count", "resize_alignment", "ignore_label"),
    "pose": ("keypoint_names", "skeleton_edges", "coordinate_space"),
    "depth": (
        "units",
        "inverse_depth",
        "scale",
        "invalid_value",
        "alignment",
        "render_min",
        "render_max",
    ),
}


CAPABILITY_STATUSES = {"READY", "DELEGATED", "ADAPTER_REQUIRED", "BLOCKED"}
ADAPTER_EVIDENCE_FIELDS = (
    "input_fixture",
    "backend_parity",
    "deepstream_runtime",
    "structured_output",
    "semantic_validation",
    "overlay_evidence",
    "rendered_output",
    "visual_validation",
)
INPUT_DOMAINS = {"image", "audio", "stereo", "temporal", "multimodal", "tensor"}

# These are execution-family rules, not model-name exceptions. They prevent a
# single-image nvinfer graph from being emitted for a task that owns additional
# inputs, upstream metadata, or semantic decoding state.
VARIANT_INPUT_DOMAINS = {
    "audio": "audio",
    "audio-classification": "audio",
    "stereo-depth": "stereo",
    "stereo-features": "stereo",
    "tracking": "temporal",
    "tracking-memory": "temporal",
}
VARIANT_UPSTREAM_STAGES = {
    "3d-pose": ("person-detection", "2d-pose"),
    "6d-pose": ("object-detection",),
    "pose-classification": ("person-detection", "2d-pose", "tracking"),
    "license-plate-recognition": ("vehicle-detection", "license-plate-detection", "ocr-crop"),
    "text-recognition": ("text-detection", "ocr-crop"),
    "ocr": ("text-detection", "ocr-crop"),
    "embedding/re-identification": ("person-or-object-detection", "tracking", "roi-crop"),
    "re-identification": ("person-or-object-detection", "tracking", "roi-crop"),
}

PIPELINE_GRAPH_STATUSES = {
    "PLANNED",
    "RENDERED_PASS",
    "RUNTIME_ONLY",
    "ADAPTER_REQUIRED",
    "BLOCKED",
}
PIPELINE_GRAPH_REQUIRED_FIELDS = (
    "nodes",
    "edges",
    "input_ownership",
    "coordinate_transforms",
    "decoder",
    "renderer",
    "structured_output",
    "evidence",
)


class ManifestError(ValueError):
    """Raised when a model manifest is incomplete or contradictory."""


@dataclass(frozen=True)
class ModelManifest:
    source: str
    source_hash: str
    source_format: str
    task: str
    model_name: str
    task_variant: str = ""
    input_name: str = ""
    output_names: tuple[str, ...] = ()
    input_shape: tuple[int | str, ...] = ()
    min_shape: tuple[int, ...] = ()
    opt_shape: tuple[int, ...] = ()
    max_shape: tuple[int, ...] = ()
    input_dtype: str = "float32"
    color_format: str = "RGB"
    scale: float = 1.0 / 255.0
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()
    preprocess_units: str = "pixel"
    layout: str = "NCHW"
    process_mode: str = "primary"
    gie_unique_id: int = 1
    preprocess_unique_id: int = 5
    target_unique_ids: tuple[int, ...] = ()
    preprocess_scope: str = "frame"
    operate_on_gie_id: int | None = None
    operate_on_class_ids: tuple[int, ...] = ()
    maintain_aspect_ratio: bool = False
    symmetric_padding: bool = False
    roi_params: dict[str, tuple[int, ...]] = field(default_factory=dict)
    postprocess: str = "custom"
    nvdspreprocess: bool = False
    nvdspostprocess: bool = False
    labels: str = ""
    engine: str = ""
    export_wrapper: str = ""
    opset: int | None = None
    environment_snapshot: str = ""
    environment: dict[str, str] = field(default_factory=dict)
    execution_approval: dict[str, Any] = field(default_factory=dict)
    fixture_hashes: tuple[str, ...] = ()
    parity_thresholds: dict[str, float] = field(default_factory=dict)
    task_contract: dict[str, Any] = field(default_factory=dict)
    pipeline_graph: dict[str, Any] = field(default_factory=dict)
    input_domain: str = "image"
    upstream_stages: tuple[str, ...] = ()
    adapter_contract: dict[str, Any] = field(default_factory=dict)
    capability_status: str = "READY"
    capability_reasons: tuple[str, ...] = ()
    precision_policy: tuple[str, ...] = ("fp16", "fp16-no-tf32", "fp32")
    delegate_skill: str = ""
    delegate_path: str = ""
    experimental: bool = False
    experimental_reviewed: bool = False
    status: str = "ready-for-validation"
    warnings: tuple[str, ...] = ()
    conversion: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()
    _manifest_dir: str = field(default="", repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_manifest_dir", None)
        data["input_shape"] = list(self.input_shape)
        data["min_shape"] = list(self.min_shape)
        data["opt_shape"] = list(self.opt_shape)
        data["max_shape"] = list(self.max_shape)
        data["output_names"] = list(self.output_names)
        data["target_unique_ids"] = list(self.target_unique_ids)
        data["operate_on_class_ids"] = list(self.operate_on_class_ids)
        data["roi_params"] = {key: list(value) for key, value in self.roi_params.items()}
        data["mean"] = list(self.mean)
        data["std"] = list(self.std)
        data["preprocess_units"] = self.preprocess_units
        data["environment"] = dict(self.environment)
        data["execution_approval"] = dict(self.execution_approval)
        data["fixture_hashes"] = list(self.fixture_hashes)
        data["parity_thresholds"] = dict(self.parity_thresholds)
        data["task_contract"] = dict(self.task_contract)
        data["pipeline_graph"] = dict(self.pipeline_graph)
        data["upstream_stages"] = list(self.upstream_stages)
        data["adapter_contract"] = dict(self.adapter_contract)
        data["capability_reasons"] = list(self.capability_reasons)
        data["precision_policy"] = list(self.precision_policy)
        data["warnings"] = list(self.warnings)
        data["conversion"] = list(self.conversion)
        data["validation"] = list(self.validation)
        return data


def infer_format(source: str | Path) -> str:
    path = Path(source)
    if path.is_dir():
        if (path / "saved_model.pb").exists():
            return "tensorflow"
        if any(path.glob("*.safetensors")):
            return "safetensors"
        if (path / "config.json").exists() and any(path.glob("*.bin")):
            return "pytorch"
        return "directory"
    return FORMAT_BY_SUFFIX.get(path.suffix.lower(), "unknown")


def _task_key(task: str) -> str:
    return task_key(task)


def _normalise_task(task: str) -> str:
    task = _task_key(task)
    task = TASK_ALIASES.get(task, task)
    if task not in TASKS:
        raise ManifestError(f"unsupported task '{task}'; choose one of: {', '.join(sorted(TASKS - {'auto'}))}")
    return task


def _normalise_dtype(value: Any) -> str:
    dtype = str(value).strip().lower()
    try:
        return INPUT_DTYPE_ALIASES[dtype]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_INPUT_DTYPES))
        raise ManifestError(f"unsupported input_dtype {value!r}; choose one of: {supported}") from exc


def _shape(value: Any) -> tuple[int | str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not value:
        raise ManifestError("input_shape must be a non-empty list of integers or symbolic strings")
    result: list[int | str] = []
    for dim in value:
        if isinstance(dim, bool):
            raise ManifestError("input_shape dimensions cannot be booleans")
        if isinstance(dim, int) and dim > 0:
            result.append(dim)
        elif isinstance(dim, str) and dim.strip():
            result.append(dim.strip())
        else:
            raise ManifestError(f"invalid input_shape dimension: {dim!r}")
    return tuple(result)


def _onnx_value_shape(value_info: Any) -> tuple[int | str, ...]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ManifestError(f"ONNX tensor {value_info.name!r} has no shape contract")
    shape: list[int | str] = []
    for index, dim in enumerate(tensor_type.shape.dim):
        if dim.HasField("dim_value") and dim.dim_value > 0:
            shape.append(int(dim.dim_value))
        elif dim.dim_param:
            shape.append(str(dim.dim_param))
        else:
            shape.append(f"{value_info.name}_dim_{index}")
    if not shape:
        raise ManifestError(f"ONNX tensor {value_info.name!r} has an empty shape contract")
    return tuple(shape)


def inspect_onnx_contract(path: str | Path) -> dict[str, Any]:
    """Return graph-owned input/output contracts after checker and inference."""

    try:
        import onnx
    except ImportError as exc:
        raise ManifestError(
            "the onnx package is required to inspect an ONNX source; "
            "install the shared validation dependencies"
        ) from exc

    source = Path(path)
    try:
        model = onnx.load(str(source))
        onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        raise ManifestError(f"cannot inspect ONNX graph {source}: {exc}") from exc

    initializer_names = {item.name for item in model.graph.initializer}
    graph_inputs = [item for item in model.graph.input if item.name not in initializer_names]
    if not graph_inputs:
        raise ManifestError("ONNX graph has no runtime inputs")

    def tensor_contract(value_info: Any, *, allow_scalar: bool = False) -> dict[str, Any]:
        tensor_type = value_info.type.tensor_type
        try:
            dtype = str(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type))
        except Exception as exc:
            raise ManifestError(
                f"cannot resolve ONNX dtype for tensor {value_info.name!r}: {exc}"
            ) from exc
        shape = () if allow_scalar and not tensor_type.shape.dim else _onnx_value_shape(value_info)
        return {"shape": shape, "dtype": dtype}

    return {
        "inputs": {item.name: tensor_contract(item) for item in graph_inputs},
        "outputs": {
            item.name: tensor_contract(item, allow_scalar=True) for item in model.graph.output
        },
        "node_count": len(model.graph.node),
    }


def _validate_shape_against_graph(
    declared: tuple[int | str, ...], graph_shape: tuple[int | str, ...]
) -> None:
    if len(declared) != len(graph_shape):
        raise ManifestError(
            f"input_shape rank {len(declared)} does not match ONNX graph rank {len(graph_shape)}"
        )
    for index, (declared_dim, graph_dim) in enumerate(zip(declared, graph_shape)):
        if isinstance(graph_dim, int) and declared_dim != graph_dim:
            raise ManifestError(
                f"input_shape dimension {index} ({declared_dim!r}) does not match "
                f"ONNX graph dimension {graph_dim}"
            )


def reconcile_onnx_metadata(
    metadata: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[str, tuple[str, ...], tuple[int | str, ...], str, bool]:
    """Derive omitted tensor metadata and reject declarations that contradict ONNX."""

    inputs = dict(contract["inputs"])
    outputs = dict(contract["outputs"])
    declared_name = str(metadata.get("input_name", "")).strip()
    if not declared_name:
        if len(inputs) != 1:
            raise ManifestError(
                "input_name is required for a multi-input ONNX graph; available inputs: "
                + ", ".join(inputs)
            )
        declared_name = next(iter(inputs))
    if declared_name not in inputs:
        raise ManifestError(
            f"input_name {declared_name!r} is not present in the ONNX graph; "
            f"available inputs: {', '.join(inputs)}"
        )

    graph_shape = tuple(inputs[declared_name]["shape"])
    declared_shape = _shape(metadata.get("input_shape"))
    if declared_shape:
        _validate_shape_against_graph(declared_shape, graph_shape)
    resolved_shape = graph_shape

    graph_dtype = _normalise_dtype(inputs[declared_name]["dtype"])
    if metadata.get("input_dtype") not in (None, ""):
        declared_dtype = _normalise_dtype(metadata["input_dtype"])
        if declared_dtype != graph_dtype:
            raise ManifestError(
                f"input_dtype {declared_dtype!r} does not match ONNX graph dtype {graph_dtype!r}"
            )
    resolved_dtype = graph_dtype

    declared_outputs = _tuple_strings(metadata.get("output_names"), "output_names")
    unknown_outputs = [name for name in declared_outputs if name not in outputs]
    if unknown_outputs:
        raise ManifestError(
            "output_names are not present in the ONNX graph: "
            + ", ".join(unknown_outputs)
            + "; available outputs: "
            + ", ".join(outputs)
        )
    resolved_outputs = declared_outputs or tuple(outputs)
    graph_dynamic = any(isinstance(dim, str) for dim in graph_shape)
    return declared_name, resolved_outputs, resolved_shape, resolved_dtype, graph_dynamic


def validate_onnx_manifest_contract(
    manifest: "ModelManifest", contract: Mapping[str, Any]
) -> None:
    """Cross-check a persisted manifest against its ONNX graph contract."""

    metadata = {
        "input_name": manifest.input_name,
        "output_names": list(manifest.output_names),
        "input_shape": list(manifest.input_shape),
        "input_dtype": manifest.input_dtype,
    }
    *_, graph_dynamic = reconcile_onnx_metadata(metadata, contract)
    if graph_dynamic and not all(
        (manifest.min_shape, manifest.opt_shape, manifest.max_shape)
    ):
        raise ManifestError(
            "ONNX graph has symbolic dimensions; explicit min_shape, opt_shape, "
            "and max_shape are required even when input_shape metadata is concrete"
        )


def _profile_shape(value: Any, name: str) -> tuple[int, ...]:
    if value is None or value == "" or (isinstance(value, (list, tuple)) and not value):
        return ()
    shape = _shape(value)
    if not all(isinstance(dim, int) for dim in shape):
        raise ManifestError(f"{name} must contain only positive integers")
    return tuple(int(dim) for dim in shape)


def _string_mapping(value: Any, name: str) -> dict[str, str]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise ManifestError(f"{name} must be a JSON object")
    return {str(key): str(item) for key, item in value.items()}


def _object_mapping(value: Any, name: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise ManifestError(f"{name} must be a JSON object")
    return dict(value)


def _threshold_mapping(value: Any, task: str) -> dict[str, float]:
    thresholds = dict(DEFAULT_PARITY_THRESHOLDS.get(task, {}))
    if value in (None, ""):
        return thresholds
    if not isinstance(value, Mapping):
        raise ManifestError("parity_thresholds must be a JSON object")
    try:
        supplied = {str(key): float(item) for key, item in value.items()}
    except (TypeError, ValueError) as exc:
        raise ManifestError("parity_thresholds values must be numeric") from exc
    if any(value < 0 for value in supplied.values()):
        raise ManifestError("parity_thresholds values must be non-negative")
    for key, default in thresholds.items():
        if key not in supplied:
            continue
        relaxes_default = supplied[key] > default if key.startswith("max_") else supplied[key] < default
        if relaxes_default:
            raise ManifestError(f"parity threshold {key!r} cannot relax the default {default:g}")
    thresholds.update(supplied)
    return thresholds


def _tuple_strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None or value == "" or (isinstance(value, (list, tuple)) and not value):
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ManifestError(f"{name} must be a list of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ManifestError(f"{name} cannot contain empty values")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ManifestError(f"{name} must be a positive integer")
    return value


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{name} must be a non-negative integer")
    return value


def _tuple_nonnegative_ints(value: Any, name: str) -> tuple[int, ...]:
    if value is None or value == "" or (isinstance(value, (list, tuple)) and not value):
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ManifestError(f"{name} must be a list of non-negative integers")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ManifestError(f"{name} must contain only non-negative integers")
    return tuple(value)


def _roi_mapping(value: Any) -> dict[str, tuple[int, ...]]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise ManifestError("roi_params must be an object keyed by source id")
    result: dict[str, tuple[int, ...]] = {}
    for raw_source_id, raw_coordinates in value.items():
        source_id = str(raw_source_id)
        if not source_id.isdigit():
            raise ManifestError("roi_params keys must be non-negative source ids")
        coordinates = _tuple_nonnegative_ints(raw_coordinates, f"roi_params.{source_id}")
        if not coordinates or len(coordinates) % 4:
            raise ManifestError(f"roi_params.{source_id} must contain x;y;width;height groups")
        for index in range(0, len(coordinates), 4):
            if coordinates[index + 2] < 1 or coordinates[index + 3] < 1:
                raise ManifestError(f"roi_params.{source_id} width and height must be positive")
        result[source_id] = coordinates
    return result


def _tuple_floats(value: Any, name: str) -> tuple[float, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ManifestError(f"{name} must be a list of numbers")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name} must contain only numbers") from exc


def _adapter_readiness_errors(
    adapter_contract: Mapping[str, Any],
    *,
    source_hash: str,
    manifest_dir: Path | None = None,
) -> list[str]:
    """Verify the validator-authored readiness record and immutable evidence hashes."""

    raw_path = str(adapter_contract.get("readiness_status", "")).strip()
    if not raw_path:
        return ["tested adapter requires adapter_contract.readiness_status"]
    status_path = Path(raw_path)
    if not status_path.is_absolute():
        if manifest_dir is None:
            return ["relative adapter readiness_status requires the manifest path"]
        status_path = manifest_dir / status_path
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"adapter readiness status is unreadable: {exc}"]
    if not isinstance(status, Mapping):
        return ["adapter readiness status must be a JSON object"]

    errors: list[str] = []
    if str(status.get("status", "")).upper() != "PASS":
        errors.append("adapter readiness status must be PASS")
    if str(status.get("implementation_status", "")).lower() != "tested":
        errors.append("adapter readiness implementation_status must be tested")
    if not source_hash or str(status.get("source_hash", "")).lower() != source_hash.lower():
        errors.append(
            "adapter readiness source_hash must match the promoted manifest"
        )
    evidence = status.get("evidence")
    if not isinstance(evidence, Mapping):
        return [*errors, "adapter readiness evidence must be a JSON object"]

    for evidence_field in ADAPTER_EVIDENCE_FIELDS:
        item = evidence.get(evidence_field)
        if not isinstance(item, Mapping):
            errors.append(f"adapter readiness evidence is missing {evidence_field}")
            continue
        path_value = str(item.get("path", "")).strip()
        expected_hash = str(item.get("sha256", "")).strip().lower()
        if not path_value or len(expected_hash) != 64:
            errors.append(
                f"adapter readiness evidence {evidence_field} lacks path or SHA-256"
            )
            continue
        evidence_path = Path(path_value)
        if not evidence_path.is_absolute():
            evidence_path = status_path.parent / evidence_path
        if not evidence_path.is_file():
            errors.append(
                f"adapter readiness evidence {evidence_field} is not an existing file"
            )
            continue
        try:
            digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        except OSError as exc:
            errors.append(
                f"adapter readiness evidence {evidence_field} is unreadable: {exc}"
            )
            continue
        if digest != expected_hash:
            errors.append(
                f"adapter readiness evidence {evidence_field} SHA-256 changed"
            )
    return errors


def _capability_for(
    task: str,
    task_variant: str,
    input_domain: str,
    postprocess: str,
    source_hash: str,
    std: tuple[float, ...],
    nvdspreprocess: bool,
    upstream_stages: tuple[str, ...],
    adapter_contract: Mapping[str, Any],
    *,
    manifest_dir: Path | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Classify pipeline readiness by execution family, never by model name."""

    if task == "detection":
        return "READY", ()

    reasons: list[str] = []
    required_adapter_fields: set[str] = set()
    if input_domain != "image":
        required_adapter_fields.update({"input_adapter", "config_generator"})
    if len(std) > 1 and len(set(std)) > 1:
        if adapter_contract.get("preprocess_strategy") == "adapter-input-tensor":
            required_adapter_fields.update({"input_adapter", "pipeline_builder"})
        else:
            required_adapter_fields.update({"preprocess_library", "preprocess_function"})
            if not nvdspreprocess:
                reasons.append(
                    "per-channel standard deviation requires an adapter-owned "
                    "input tensor or nvdspreprocess custom library"
                )
            library = adapter_contract.get("preprocess_library")
            if library and not Path(str(library)).is_absolute():
                reasons.append("preprocess_library must be an absolute path")
    if task != "classification" or postprocess != "native" or task_variant in {
        "license-plate-recognition", "text-recognition", "ocr", "pose-classification"
    }:
        required_adapter_fields.update(
            {"decoder", "renderer", "structured_output", "pipeline_runner"}
        )

    required_upstream = VARIANT_UPSTREAM_STAGES.get(task_variant, ())
    if required_upstream or upstream_stages:
        required_adapter_fields.add("pipeline_builder")
    missing_upstream = [stage for stage in required_upstream if stage not in upstream_stages]
    if missing_upstream:
        reasons.append("missing required upstream stages: " + ", ".join(missing_upstream))

    missing_adapter = [
        field for field in sorted(required_adapter_fields)
        if adapter_contract.get(field) in (None, "", [], {})
    ]
    if missing_adapter:
        reasons.append("missing adapter contract fields: " + ", ".join(missing_adapter))

    if (required_adapter_fields or adapter_contract) and not missing_adapter:
        implementation_status = str(adapter_contract.get("implementation_status", "")).strip().lower()
        if implementation_status != "tested":
            reasons.append(
                "adapter implementation_status must be tested after real-input DeepStream and visual validation"
            )
        else:
            evidence = adapter_contract.get("validation_evidence")
            missing_evidence = [
                field for field in ADAPTER_EVIDENCE_FIELDS
                if not isinstance(evidence, Mapping) or evidence.get(field) in (None, "", [], {})
            ]
            if missing_evidence:
                reasons.append(
                    "tested adapter is missing validation_evidence: " + ", ".join(missing_evidence)
                )
            else:
                reasons.extend(
                    _adapter_readiness_errors(
                        adapter_contract,
                        source_hash=source_hash,
                        manifest_dir=manifest_dir,
                    )
                )

    if reasons:
        return "ADAPTER_REQUIRED", tuple(reasons)
    return "READY", ()


def _conversion_for(source_format: str, task: str) -> tuple[str, ...]:
    if task == "detection":
        return (
            "use the native acquisition/export and TensorRT build path",
            "configure or build the verified DeepStream bbox parser",
        )
    if source_format == "tensorrt":
        return ("validate TensorRT plan with trtexec", "use engine directly with nvinfer")
    if source_format == "onnx":
        return ("onnx.checker.check_model", "onnx.shape_inference.infer_shapes", "build TensorRT engine with trtexec")
    if source_format in {"pytorch", "safetensors", "torchscript"}:
        return (
            "use torch.onnx.export with an explicit deterministic wrapper and example input",
            "optionally use Optimum only after a compatible route environment passes import checks",
            "validate ONNX, then build TensorRT engine with trtexec",
        )
    if source_format == "tflite":
        return ("experimental: tf2onnx.convert --tflite", "validate ONNX, then build TensorRT engine with trtexec")
    if source_format == "tensorflow":
        return ("experimental: tf2onnx.convert from SavedModel/Keras", "validate ONNX, then build TensorRT engine with trtexec")
    if source_format in UNSUPPORTED_FORMATS:
        return ("stop: Caffe/UFF import is not supported by current DeepStream nvinfer",)
    return ("stop: identify the source format and export to ONNX first",)


def _validation_for(source_format: str, task: str) -> tuple[str, ...]:
    if task == "detection":
        return (
            "run TensorRT validation",
            "run one-stream DeepStream visual validation",
            "validate structured detections and the KITTI gate",
            "run the multi-stream benchmark after functional gates pass",
        )
    checks = [
        "record environment snapshot, validate manifest schema, source, fixtures, and tensor contracts",
    ]
    if source_format == "tensorrt":
        checks.extend([
            "validate supplied TensorRT version, GPU compatibility, profiles, plugins, and bindings",
            "deserialize the supplied engine with trtexec only from an explicitly trusted root",
            "generate DeepStream configs from the validated binding contract",
            "run one-stream DeepStream nvinfer startup with GST_DEBUG=2",
            "compare reference and DeepStream task outputs on the fixed fixtures",
        ])
        return tuple(checks)
    if source_format in EXPERIMENTAL_FORMATS:
        checks.append("record conversion as experimental and require operator/parity review")
    checks.extend([
        "run ONNX checker and shape inference; inspect inputs, outputs, opset, dtypes, dimensions, and operators",
        "compare source-framework/reference and ONNX Runtime outputs on fixed fixtures",
        "build the TensorRT engine and run the trtexec smoke test with explicit profiles",
        "compare TensorRT and reference outputs on the same fixed fixtures",
        "generate DeepStream nvinfer and optional preprocess/postprocess configs",
        "run one-stream DeepStream nvinfer startup with GST_DEBUG=2",
        "compare DeepStream and reference task outputs before profiling",
    ])
    return tuple(checks)


def route_model(
    source: str | Path,
    *,
    task: str = "auto",
    metadata: Mapping[str, Any] | None = None,
    manifest_dir: str | Path | None = None,
) -> ModelManifest:
    """Select the safest supported path without executing external tools."""

    # Persist a stable local path so generated manifests may move between phase
    # directories without changing what their source field identifies.
    source_path = Path(source).expanduser().resolve()
    manifest_base = Path(manifest_dir).absolute() if manifest_dir else None
    source_format = infer_format(source_path)
    meta = dict(metadata or {})
    requested_key = _task_key(task)
    requested_task = _normalise_task(requested_key)
    detected_key = _task_key(str(meta.get("task", "generic"))) if meta.get("task") else "generic"
    detected_task = _normalise_task(detected_key)
    resolved_task = detected_task if requested_task == "auto" and detected_task != "generic" else requested_task
    if resolved_task == "auto":
        resolved_task = "generic"
    task_variant = ""
    if requested_key not in {"auto"} and requested_key != resolved_task:
        task_variant = requested_key
    elif requested_key == "auto" and detected_key != detected_task:
        task_variant = detected_key
    model_name = str(meta.get("model_name") or source_path.stem or "model")
    if not model_name.replace("_", "").replace("-", "").replace(".", "").isalnum():
        raise ManifestError("model_name must contain only letters, numbers, '.', '_' or '-'")

    warnings: list[str] = []
    status = "ready-for-validation"
    experimental = source_format in EXPERIMENTAL_FORMATS
    experimental_reviewed = meta.get("experimental_reviewed", False)
    if not isinstance(experimental_reviewed, bool):
        raise ManifestError("experimental_reviewed must be a boolean")
    if not source_path.exists():
        status = "blocked-missing-source"
        warnings.append("source model path does not exist")
    elif source_format == "unknown" or source_format == "directory":
        status = "needs-format-identification"
        warnings.append("source format is not identifiable from the path; provide a model file or explicit export metadata")
    elif source_format in UNSUPPORTED_FORMATS:
        status = "unsupported"
        warnings.append("Caffe/UFF import is intentionally rejected for current DeepStream releases")
    elif experimental:
        warnings.append("conversion is experimental; operator coverage and numerical parity must be reviewed")
        if experimental_reviewed:
            warnings.append("experimental conversion review was explicitly approved in metadata")
        else:
            status = "experimental-review-required"
            warnings.append("set experimental_reviewed=true only after operator and numerical parity review")
    if resolved_task == "generic":
        warnings.append("task is generic; output decoding must be supplied by the application or a custom library")

    input_name = str(meta.get("input_name", "images")).strip()
    output_names = _tuple_strings(meta.get("output_names"), "output_names")
    shape = _shape(meta.get("input_shape"))
    input_dtype = _normalise_dtype(meta.get("input_dtype", "float32"))
    graph_dynamic = False
    if source_format == "onnx" and source_path.exists() and resolved_task != "detection":
        onnx_contract = inspect_onnx_contract(source_path)
        input_name, output_names, shape, input_dtype, graph_dynamic = reconcile_onnx_metadata(
            meta, onnx_contract
        )
    elif not shape and resolved_task != "detection":
        if status == "ready-for-validation":
            status = "needs-input-contract"
        warnings.append("input_shape is required for non-ONNX sources; no default shape is guessed")
    min_shape = _profile_shape(meta.get("min_shape"), "min_shape")
    opt_shape = _profile_shape(meta.get("opt_shape"), "opt_shape")
    max_shape = _profile_shape(meta.get("max_shape"), "max_shape")
    profiles = (min_shape, opt_shape, max_shape)
    if (graph_dynamic or any(isinstance(dim, str) for dim in shape)) and not all(profiles):
        if status == "ready-for-validation":
            status = "needs-shape-profile"
        warnings.append("dynamic dimensions require explicit min_shape, opt_shape, and max_shape metadata")
    mean = _tuple_floats(meta.get("mean"), "mean")
    std = _tuple_floats(meta.get("std"), "std")
    input_domain = str(
        meta.get("input_domain") or VARIANT_INPUT_DOMAINS.get(task_variant, "image")
    ).strip().lower()
    if input_domain not in INPUT_DOMAINS:
        supported_domains = ", ".join(sorted(INPUT_DOMAINS))
        raise ManifestError(f"unsupported input_domain {input_domain!r}; choose one of: {supported_domains}")
    upstream_stages = _tuple_strings(meta.get("upstream_stages"), "upstream_stages")
    adapter_contract = _object_mapping(meta.get("adapter_contract"), "adapter_contract")
    precision_policy = _tuple_strings(
        meta.get("precision_policy", ["fp16", "fp16-no-tf32", "fp32"]), "precision_policy"
    )
    allowed_precisions = {"fp16", "fp16-no-tf32", "fp32", "int8"}
    if not precision_policy or any(item not in allowed_precisions for item in precision_policy):
        raise ManifestError(
            "precision_policy must be a non-empty ordered list containing fp16, fp16-no-tf32, fp32, or int8"
        )
    nvdspreprocess = _boolean(meta.get("nvdspreprocess", False), "nvdspreprocess")
    preprocess_units = str(meta.get("preprocess_units", "pixel")).strip().lower()
    if preprocess_units not in {"pixel", "normalized"}:
        raise ManifestError("preprocess_units must be 'pixel' (0-255) or 'normalized' (applied after scale)")
    if preprocess_units == "pixel" and mean and all(0.0 < value <= 1.0 for value in mean):
        warnings.append("mean values look normalized to [0,1] but preprocess_units is 'pixel'; set preprocess_units='normalized' or supply 0-255 values")
    postprocess = str(meta.get("postprocess", "native" if resolved_task == "classification" else "custom"))
    nvdspostprocess = _boolean(meta.get("nvdspostprocess", False), "nvdspostprocess")
    if nvdspostprocess and resolved_task not in {"detection", "classification"}:
        warnings.append("DeepStream 9.1 nvdspostprocess native task coverage is detection/classification; use a custom/app postprocessor for this task")

    environment = _string_mapping(meta.get("environment"), "environment")
    environment_snapshot = str(meta.get("environment_snapshot", ""))
    execution_approval = _object_mapping(
        meta.get("execution_approval"), "execution_approval"
    )
    if not environment or not environment_snapshot:
        warnings.append("capture environment.json and record environment_snapshot plus version fields before execution")
    fixture_hashes = _tuple_strings(meta.get("fixture_hashes"), "fixture_hashes")
    source_hash = str(meta.get("source_hash", "")).strip()
    task_contract = _object_mapping(meta.get("task_contract"), "task_contract")
    if resolved_task != "detection" and not task_contract:
        warnings.append("task_contract is required before config generation, postprocessing, and parity validation")
    pipeline_graph = _object_mapping(meta.get("pipeline_graph"), "pipeline_graph")
    if resolved_task != "detection" and not pipeline_graph:
        warnings.append(
            "pipeline_graph is required before pipeline creation; declare ordered nodes, edge contracts, "
            "input ownership, decoder, renderer, structured output, and evidence"
        )
    if not output_names:
        warnings.append("output_names are required before engine/config generation and tensor parity validation")
    capability_status, capability_reasons = _capability_for(
        resolved_task,
        task_variant,
        input_domain,
        postprocess,
        source_hash,
        std,
        nvdspreprocess,
        upstream_stages,
        adapter_contract,
        manifest_dir=manifest_base,
    )
    if status != "ready-for-validation":
        capability_status = "BLOCKED"
        capability_reasons = (f"format route status is {status}", *capability_reasons)
    if capability_status in {"ADAPTER_REQUIRED", "BLOCKED"}:
        warnings.extend(capability_reasons)
    parity_thresholds = _threshold_mapping(meta.get("parity_thresholds"), resolved_task)
    gie_unique_id = _positive_int(meta.get("gie_unique_id", 1), "gie_unique_id")
    preprocess_unique_id = _positive_int(meta.get("preprocess_unique_id", 5), "preprocess_unique_id")
    target_unique_ids = _tuple_nonnegative_ints(meta.get("target_unique_ids", [gie_unique_id]), "target_unique_ids")
    if not target_unique_ids or any(value < 1 for value in target_unique_ids):
        raise ManifestError("target_unique_ids must contain at least one positive GIE id")
    preprocess_scope = str(meta.get("preprocess_scope", "frame")).strip().lower()
    operate_on_gie_id = _optional_nonnegative_int(meta.get("operate_on_gie_id"), "operate_on_gie_id")
    operate_on_class_ids = _tuple_nonnegative_ints(meta.get("operate_on_class_ids"), "operate_on_class_ids")
    maintain_aspect_ratio = _boolean(meta.get("maintain_aspect_ratio", False), "maintain_aspect_ratio")
    symmetric_padding = _boolean(meta.get("symmetric_padding", False), "symmetric_padding")
    roi_params = _roi_mapping(meta.get("roi_params"))

    manifest = ModelManifest(
        source=str(source_path),
        source_hash=source_hash,
        source_format=source_format,
        task=resolved_task,
        model_name=model_name,
        task_variant=task_variant,
        input_name=input_name,
        output_names=output_names,
        input_shape=shape,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        input_dtype=input_dtype,
        color_format=str(meta.get("color_format", "RGB")).upper(),
        scale=float(meta.get("scale", 1.0 / 255.0)),
        mean=mean,
        std=std,
        layout=str(meta.get("layout", "NCHW")).upper(),
        process_mode=str(meta.get("process_mode", "primary")),
        gie_unique_id=gie_unique_id,
        preprocess_unique_id=preprocess_unique_id,
        target_unique_ids=target_unique_ids,
        preprocess_scope=preprocess_scope,
        operate_on_gie_id=operate_on_gie_id,
        operate_on_class_ids=operate_on_class_ids,
        maintain_aspect_ratio=maintain_aspect_ratio,
        symmetric_padding=symmetric_padding,
        roi_params=roi_params,
        postprocess=postprocess,
        nvdspreprocess=nvdspreprocess,
        nvdspostprocess=nvdspostprocess,
        preprocess_units=preprocess_units,
        labels=str(meta.get("labels", "")),
        engine=str(meta.get("engine", "")),
        export_wrapper=str(meta.get("export_wrapper", "")),
        opset=int(meta["opset"]) if meta.get("opset") is not None else None,
        environment_snapshot=environment_snapshot,
        environment=environment,
        execution_approval=execution_approval,
        fixture_hashes=fixture_hashes,
        parity_thresholds=parity_thresholds,
        task_contract=task_contract,
        pipeline_graph=pipeline_graph,
        input_domain=input_domain,
        upstream_stages=upstream_stages,
        adapter_contract=adapter_contract,
        capability_status=capability_status,
        capability_reasons=capability_reasons,
        precision_policy=precision_policy,
        delegate_skill="",
        delegate_path="",
        experimental=experimental,
        experimental_reviewed=experimental_reviewed,
        status=status,
        warnings=tuple(warnings),
        conversion=_conversion_for(source_format, resolved_task),
        validation=_validation_for(source_format, resolved_task),
        _manifest_dir=str(manifest_base) if manifest_base else "",
    )
    validate_manifest(manifest)
    return manifest


def validate_manifest(
    manifest: ModelManifest,
    *,
    manifest_dir: Path | None = None,
) -> None:
    """Validate invariants before any conversion or engine build."""

    if manifest.source_format not in FIRST_CLASS_FORMATS | EXPERIMENTAL_FORMATS | UNSUPPORTED_FORMATS | {"unknown", "directory"}:
        raise ManifestError(f"unknown source format: {manifest.source_format}")
    if manifest.task not in TASKS - {"auto"}:
        raise ManifestError(f"invalid resolved task: {manifest.task}")
    if manifest.input_domain not in INPUT_DOMAINS:
        raise ManifestError(f"invalid input_domain: {manifest.input_domain}")
    if manifest.status == "ready-for-validation" and manifest.task != "detection" and not manifest.input_name:
        raise ManifestError("ready manifests require input_name")
    if manifest.status == "ready-for-validation" and manifest.task != "detection" and not manifest.input_shape:
        raise ManifestError("ready manifests require input_shape; no default shape is inferred")
    if manifest.capability_status not in CAPABILITY_STATUSES:
        raise ManifestError(f"invalid capability_status: {manifest.capability_status}")
    allowed_precisions = {"fp16", "fp16-no-tf32", "fp32", "int8"}
    if not manifest.precision_policy or any(item not in allowed_precisions for item in manifest.precision_policy):
        raise ManifestError("precision_policy contains an unsupported precision")
    if len(set(manifest.precision_policy)) != len(manifest.precision_policy):
        raise ManifestError("precision_policy must not contain duplicate entries")
    if manifest.layout not in {"NCHW", "NHWC", "NC", "NCL", "NLC"}:
        raise ManifestError(f"unsupported input layout: {manifest.layout}")
    if manifest.color_format not in {"RGB", "BGR", "GRAY"}:
        raise ManifestError(
            f"stock nvinfer/nvdspreprocess does not support color format {manifest.color_format!r}; "
            "use RGB, BGR, or GRAY, or supply a tested custom preprocessing adapter"
        )
    if manifest.input_dtype not in SUPPORTED_INPUT_DTYPES:
        raise ManifestError(f"unsupported input dtype: {manifest.input_dtype}")
    if manifest.preprocess_units not in {"pixel", "normalized"}:
        raise ManifestError(f"unsupported preprocess_units: {manifest.preprocess_units}; use 'pixel' or 'normalized'")
    if manifest.scale <= 0:
        raise ManifestError("scale must be greater than zero")
    if manifest.process_mode not in {"primary", "secondary"}:
        raise ManifestError("process_mode must be 'primary' or 'secondary'")
    if manifest.process_mode == "secondary" and manifest.operate_on_gie_id is None:
        raise ManifestError("secondary processing requires operate_on_gie_id")
    if manifest.operate_on_gie_id is not None and manifest.operate_on_gie_id < 1:
        raise ManifestError("operate_on_gie_id must be a positive GIE id")
    if manifest.preprocess_scope not in {"frame", "object"}:
        raise ManifestError("preprocess_scope must be 'frame' or 'object'")
    if manifest.preprocess_scope == "object" and manifest.operate_on_gie_id is None:
        raise ManifestError("object preprocessing requires operate_on_gie_id")
    if (
        manifest.preprocess_scope == "frame"
        and manifest.process_mode != "secondary"
        and manifest.operate_on_gie_id is not None
    ):
        raise ManifestError("operate_on_gie_id requires secondary inference or object preprocessing")
    if manifest.symmetric_padding and not manifest.maintain_aspect_ratio:
        raise ManifestError("symmetric_padding requires maintain_aspect_ratio=true")
    if manifest.gie_unique_id < 1 or manifest.preprocess_unique_id < 1:
        raise ManifestError("GIE and preprocess unique ids must be positive")
    if not manifest.target_unique_ids or any(value < 1 for value in manifest.target_unique_ids):
        raise ManifestError("target_unique_ids must contain positive GIE ids")
    if len(set(manifest.target_unique_ids)) != len(manifest.target_unique_ids):
        raise ManifestError("target_unique_ids must not contain duplicates")
    if manifest.gie_unique_id not in manifest.target_unique_ids:
        raise ManifestError("target_unique_ids must include gie_unique_id")
    if manifest.experimental != (manifest.source_format in EXPERIMENTAL_FORMATS):
        raise ManifestError("manifest experimental flag does not match source_format")
    profiles = (manifest.min_shape, manifest.opt_shape, manifest.max_shape)
    if any(profiles) and not all(profiles):
        raise ManifestError("min_shape, opt_shape, and max_shape must be supplied together")
    if all(profiles):
        rank = len(manifest.input_shape)
        if any(len(profile) != rank for profile in profiles):
            raise ManifestError("all shape profiles must have the same rank as input_shape")
        for minimum, optimum, maximum in zip(*profiles):
            if not minimum <= optimum <= maximum:
                raise ManifestError("shape profiles must satisfy min_shape <= opt_shape <= max_shape")
    has_dynamic_dim = any(isinstance(dim, str) for dim in manifest.input_shape)
    if has_dynamic_dim and not all(profiles) and manifest.status != "needs-shape-profile":
        raise ManifestError("dynamic input_shape requires explicit min/opt/max profiles")
    if manifest.delegate_skill or manifest.delegate_path:
        raise ManifestError("unified vision manifests must not delegate to another skill")
    readiness_base = manifest_dir or (
        Path(manifest._manifest_dir) if manifest._manifest_dir else None
    )
    expected_capability, _ = _capability_for(
        manifest.task,
        manifest.task_variant,
        manifest.input_domain,
        manifest.postprocess,
        manifest.source_hash,
        manifest.std,
        manifest.nvdspreprocess,
        manifest.upstream_stages,
        manifest.adapter_contract,
        manifest_dir=readiness_base,
    )
    if manifest.status != "ready-for-validation":
        expected_capability = "BLOCKED"
    if manifest.capability_status != expected_capability:
        raise ManifestError(
            f"capability_status {manifest.capability_status!r} contradicts the task-family contract; "
            f"expected {expected_capability!r}"
        )
    if manifest.mean and manifest.std and len(manifest.mean) != len(manifest.std):
        raise ManifestError("mean and std must have the same number of channels")
    if any(value <= 0 for value in manifest.std):
        raise ManifestError("std values must be greater than zero")
    if manifest.source_format == "tensorrt" and not manifest.engine and not manifest.source:
        raise ManifestError("TensorRT manifests require an engine or plan source")
    if manifest.status == "unsupported":
        return
    if manifest.task == "classification" and manifest.postprocess == "native" and manifest.nvdspostprocess:
        raise ManifestError("choose native nvinfer classification parsing or nvdspostprocess, not both")
    if not isinstance(manifest.pipeline_graph, dict):
        raise ManifestError("pipeline_graph must be a JSON object")


def pipeline_graph_errors(
    graph: Mapping[str, Any],
    *,
    require_complete: bool = True,
    required_upstream_stages: tuple[str, ...] = (),
) -> list[str]:
    """Return actionable errors for the model-first graph contract."""

    if not isinstance(graph, Mapping) or not graph:
        return ["pipeline_graph"] if require_complete else []
    errors: list[str] = []
    missing = [field for field in PIPELINE_GRAPH_REQUIRED_FIELDS if field not in graph]
    if missing:
        errors.extend(f"pipeline_graph.{field}" for field in missing)
    status = str(graph.get("status", "")).strip().upper()
    if status not in PIPELINE_GRAPH_STATUSES:
        errors.append("pipeline_graph.status (PLANNED, RENDERED_PASS, RUNTIME_ONLY, ADAPTER_REQUIRED, or BLOCKED)")

    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append("pipeline_graph.nodes (non-empty ordered list)")
    else:
        node_tokens: list[str] = []
        for index, node in enumerate(nodes):
            if not isinstance(node, Mapping):
                errors.append(f"pipeline_graph.nodes[{index}] (object with id and role)")
                continue
            for field in ("id", "role"):
                if not str(node.get(field, "")).strip():
                    errors.append(f"pipeline_graph.nodes[{index}].{field}")
            node_tokens.append(
                " ".join(
                    str(node.get(field, "")).strip().lower()
                    for field in ("id", "role")
                )
            )
        for stage in required_upstream_stages:
            normalized = stage.strip().lower()
            if normalized and not any(normalized in token for token in node_tokens):
                errors.append(f"pipeline_graph.nodes (missing upstream stage {stage!r})")

    edges = graph.get("edges")
    if not isinstance(edges, list) or not edges:
        errors.append("pipeline_graph.edges (non-empty list with from, to, and contract)")
    else:
        for index, edge in enumerate(edges):
            if not isinstance(edge, Mapping):
                errors.append(f"pipeline_graph.edges[{index}] (object with from, to, and contract)")
                continue
            for field in ("from", "to", "contract"):
                if not str(edge.get(field, "")).strip():
                    errors.append(f"pipeline_graph.edges[{index}].{field}")

    for field in ("input_ownership", "coordinate_transforms", "decoder", "renderer", "structured_output", "evidence"):
        if graph.get(field) in (None, "", [], {}):
            errors.append(f"pipeline_graph.{field}")
    if not isinstance(graph.get("evidence"), Mapping):
        errors.append("pipeline_graph.evidence (JSON object of evidence paths)")
    return list(dict.fromkeys(errors))


def validate_backend_manifest(manifest: ModelManifest) -> None:
    """Require contracts needed for conversion, backend parity, and adapter testing."""

    validate_manifest(manifest)
    if manifest.status != "ready-for-validation":
        raise ManifestError(f"cannot execute while manifest status is {manifest.status!r}")
    missing: list[str] = []
    if not manifest.output_names:
        missing.append("output_names")
    if not manifest.environment_snapshot:
        missing.append("environment_snapshot")
    if not manifest.environment:
        missing.append("environment")
    required = TASK_CONTRACT_REQUIRED_FIELDS.get(manifest.task, ())
    missing_contract = [field for field in required if field not in manifest.task_contract]
    if missing_contract:
        missing.append(f"task_contract ({', '.join(missing_contract)})")
    graph_missing = pipeline_graph_errors(
        manifest.pipeline_graph,
        required_upstream_stages=manifest.upstream_stages,
    )
    if graph_missing:
        missing.append("; ".join(graph_missing))
    if missing:
        raise ManifestError("execution manifest is incomplete; provide " + ", ".join(missing))


def validate_execution_manifest(manifest: ModelManifest) -> None:
    """Require backend contracts plus evidence-backed production readiness."""

    validate_backend_manifest(manifest)
    if manifest.capability_status != "READY":
        detail = "; ".join(manifest.capability_reasons) or "required capability is unavailable"
        raise ManifestError(
            f"cannot execute while capability_status is {manifest.capability_status!r}: {detail}"
        )


def manifest_from_dict(
    data: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
) -> ModelManifest:
    """Load a manifest written by route_model or a hand-authored JSON file."""

    required = {"source", "source_format", "task", "model_name"}
    missing = sorted(required - set(data))
    if missing:
        raise ManifestError(f"manifest missing required fields: {', '.join(missing)}")
    experimental_value = data.get("experimental", str(data["source_format"]) in EXPERIMENTAL_FORMATS)
    experimental_reviewed_value = data.get("experimental_reviewed", False)
    if not isinstance(experimental_value, bool):
        raise ManifestError("experimental must be a boolean")
    if not isinstance(experimental_reviewed_value, bool):
        raise ManifestError("experimental_reviewed must be a boolean")
    raw_task = _task_key(str(data["task"]))
    resolved_task = _normalise_task(raw_task)
    postprocess = str(data.get("postprocess", "native" if resolved_task == "classification" else "custom"))
    task_variant = str(data.get("task_variant", ""))
    if not task_variant and raw_task != resolved_task:
        task_variant = raw_task
    input_domain = str(
        data.get("input_domain") or VARIANT_INPUT_DOMAINS.get(task_variant, "image")
    ).strip().lower()
    upstream_stages = _tuple_strings(data.get("upstream_stages"), "upstream_stages")
    adapter_contract = _object_mapping(data.get("adapter_contract"), "adapter_contract")
    execution_approval = _object_mapping(
        data.get("execution_approval"), "execution_approval"
    )
    std = _tuple_floats(data.get("std"), "std")
    nvdspreprocess = _boolean(data.get("nvdspreprocess", False), "nvdspreprocess")
    manifest_dir = Path(manifest_path).absolute().parent if manifest_path else None
    computed_capability, computed_reasons = _capability_for(
        resolved_task,
        task_variant,
        input_domain,
        postprocess,
        str(data.get("source_hash", "")).strip(),
        std,
        nvdspreprocess,
        upstream_stages,
        adapter_contract,
        manifest_dir=manifest_dir,
    )
    route_status = str(data.get("status", "ready-for-validation"))
    if route_status != "ready-for-validation":
        computed_capability = "BLOCKED"
        computed_reasons = (f"format route status is {route_status}", *computed_reasons)
    manifest = ModelManifest(
        source=str(data["source"]),
        source_hash=str(data.get("source_hash", "")).strip(),
        source_format=str(data["source_format"]),
        task=resolved_task,
        model_name=str(data["model_name"]),
        task_variant=task_variant,
        input_name=str(data.get("input_name", "")),
        output_names=_tuple_strings(data.get("output_names"), "output_names"),
        input_shape=_shape(data.get("input_shape") or None),
        min_shape=_profile_shape(data.get("min_shape"), "min_shape"),
        opt_shape=_profile_shape(data.get("opt_shape"), "opt_shape"),
        max_shape=_profile_shape(data.get("max_shape"), "max_shape"),
        input_dtype=_normalise_dtype(data.get("input_dtype", "float32")),
        color_format=str(data.get("color_format", "RGB")).upper(),
        scale=float(data.get("scale", 1.0 / 255.0)),
        mean=_tuple_floats(data.get("mean"), "mean"),
        std=std,
        layout=str(data.get("layout", "NCHW")).upper(),
        process_mode=str(data.get("process_mode", "primary")),
        gie_unique_id=_positive_int(data.get("gie_unique_id", 1), "gie_unique_id"),
        preprocess_unique_id=_positive_int(data.get("preprocess_unique_id", 5), "preprocess_unique_id"),
        target_unique_ids=_tuple_nonnegative_ints(
            data.get("target_unique_ids", [data.get("gie_unique_id", 1)]), "target_unique_ids"
        ),
        preprocess_scope=str(data.get("preprocess_scope", "frame")).strip().lower(),
        operate_on_gie_id=_optional_nonnegative_int(data.get("operate_on_gie_id"), "operate_on_gie_id"),
        operate_on_class_ids=_tuple_nonnegative_ints(data.get("operate_on_class_ids"), "operate_on_class_ids"),
        maintain_aspect_ratio=_boolean(data.get("maintain_aspect_ratio", False), "maintain_aspect_ratio"),
        symmetric_padding=_boolean(data.get("symmetric_padding", False), "symmetric_padding"),
        roi_params=_roi_mapping(data.get("roi_params")),
        postprocess=postprocess,
        nvdspreprocess=nvdspreprocess,
        nvdspostprocess=_boolean(data.get("nvdspostprocess", False), "nvdspostprocess"),
        preprocess_units=str(data.get("preprocess_units", "pixel")).lower(),
        labels=str(data.get("labels", "")),
        engine=str(data.get("engine", "")),
        export_wrapper=str(data.get("export_wrapper", "")),
        opset=int(data["opset"]) if data.get("opset") is not None else None,
        environment_snapshot=str(data.get("environment_snapshot", "")),
        environment=_string_mapping(data.get("environment"), "environment"),
        execution_approval=execution_approval,
        fixture_hashes=_tuple_strings(data.get("fixture_hashes"), "fixture_hashes"),
        parity_thresholds=_threshold_mapping(data.get("parity_thresholds"), resolved_task),
        task_contract=_object_mapping(data.get("task_contract"), "task_contract"),
        pipeline_graph=_object_mapping(data.get("pipeline_graph"), "pipeline_graph"),
        input_domain=input_domain,
        upstream_stages=upstream_stages,
        adapter_contract=adapter_contract,
        capability_status=computed_capability,
        capability_reasons=computed_reasons,
        precision_policy=_tuple_strings(
            data.get("precision_policy", ["fp16", "fp16-no-tf32", "fp32"]), "precision_policy"
        ),
        delegate_skill=str(data.get("delegate_skill", "")),
        delegate_path=str(data.get("delegate_path", "")),
        experimental=experimental_value,
        experimental_reviewed=experimental_reviewed_value,
        status=route_status,
        warnings=_tuple_strings(data.get("warnings"), "warnings"),
        conversion=_tuple_strings(data.get("conversion"), "conversion"),
        validation=_tuple_strings(data.get("validation"), "validation"),
        _manifest_dir=str(manifest_dir) if manifest_dir else "",
    )
    validate_manifest(manifest)
    return manifest
