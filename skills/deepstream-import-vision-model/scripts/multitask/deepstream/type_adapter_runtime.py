#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Executable, model-name-independent adapters for supported task families.

The generated adapter supplies the model contract. This module implements only
type-level tensor decoding, rendering, serialization, and bounded pipeline
ownership. Classification has a shipped reference DeepStream runner; other
categories require an explicitly supplied runner and remain ADAPTER_REQUIRED.
No generated adapter claims readiness without real DeepStream and visual evidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


class AdapterContractError(ValueError):
    """Raised when supplied tensors do not satisfy the declared task contract."""


def _array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not array.size:
        raise AdapterContractError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise AdapterContractError(f"{name} contains non-finite values")
    return array


def _output(
    spec: dict[str, Any],
    outputs: dict[str, Any],
    *,
    require_finite: bool = True,
) -> tuple[str, np.ndarray]:
    if not isinstance(outputs, dict) or not outputs:
        raise AdapterContractError("outputs must be a non-empty tensor mapping")
    names = list(spec.get("output_names", []))
    name = next((candidate for candidate in names if candidate in outputs), "")
    if not name:
        if len(outputs) != 1:
            raise AdapterContractError(
                "none of the declared output_names is present and output selection is ambiguous"
            )
        name = next(iter(outputs))
    if require_finite:
        return name, _array(outputs[name], name)
    array = np.asarray(outputs[name])
    if not array.size:
        raise AdapterContractError(f"{name} is empty")
    return name, array


def _label(contract: dict[str, Any], index: int) -> str:
    labels = contract.get("label_map", {})
    if isinstance(labels, list) and 0 <= index < len(labels):
        return str(labels[index])
    if isinstance(labels, dict):
        return str(labels.get(str(index), labels.get(index, index)))
    return str(index)


def _classification(spec: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    name, scores = _output(spec, outputs)
    scores = np.squeeze(scores)
    if scores.ndim != 1:
        raise AdapterContractError(f"classification output must reduce to rank 1, got {scores.shape}")
    contract = spec["task_contract"]
    count = int(contract["class_count"])
    if scores.size != count:
        raise AdapterContractError(f"expected {count} classes, got {scores.size}")
    semantics = str(contract["score_semantics"]).lower()
    if semantics in {"logit", "logits"}:
        shifted = scores.astype(np.float64) - float(np.max(scores))
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
    elif semantics in {"probability", "probabilities", "score", "scores"}:
        probabilities = scores.astype(np.float64)
    else:
        raise AdapterContractError(f"unsupported score_semantics: {semantics}")
    top_k = min(max(int(contract["top_k"]), 1), count)
    order = np.argsort(-probabilities, kind="stable")[:top_k]
    threshold = float(contract["threshold"])
    predictions = [
        {"class_id": int(index), "label": _label(contract, int(index)), "score": float(probabilities[index])}
        for index in order
        if float(probabilities[index]) >= threshold
    ]
    return {"task": "classification", "output_name": name, "predictions": predictions}


def _segmentation(spec: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    name, tensor = _output(spec, outputs)
    contract = spec["task_contract"]
    layout = str(contract["mask_layout"]).upper()
    array = np.squeeze(tensor, axis=0) if tensor.ndim >= 3 and tensor.shape[0] == 1 else tensor
    class_count = int(contract["class_count"])
    if layout in {"NCHW", "CHW"}:
        if array.ndim != 3 or array.shape[0] != class_count:
            raise AdapterContractError(f"expected CHW logits with {class_count} channels, got {array.shape}")
        mask = np.argmax(array, axis=0)
    elif layout in {"NHWC", "HWC"}:
        if array.ndim != 3 or array.shape[-1] != class_count:
            raise AdapterContractError(f"expected HWC logits with {class_count} channels, got {array.shape}")
        mask = np.argmax(array, axis=-1)
    elif layout in {"NHW", "HW", "LABELS"}:
        mask = np.squeeze(array)
        if mask.ndim != 2:
            raise AdapterContractError(f"expected a 2D label mask, got {mask.shape}")
    else:
        raise AdapterContractError(f"unsupported mask_layout: {layout}")
    mask = mask.astype(np.int32)
    ignore_label = int(contract["ignore_label"])
    invalid = ((mask < 0) | (mask >= class_count)) & (mask != ignore_label)
    if np.any(invalid):
        raise AdapterContractError("segmentation mask contains an out-of-range class id")
    labels, counts = np.unique(mask, return_counts=True)
    return {
        "task": "segmentation",
        "output_name": name,
        "mask": mask,
        "shape": list(mask.shape),
        "class_histogram": {str(int(label)): int(count) for label, count in zip(labels, counts)},
    }


def _pose(spec: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    name, tensor = _output(spec, outputs)
    contract = spec["task_contract"]
    names = list(contract["keypoint_names"])
    array = np.squeeze(tensor, axis=0) if tensor.ndim >= 3 and tensor.shape[0] == 1 else tensor
    keypoints: list[dict[str, Any]] = []
    heatmap_shape: list[int] = []
    confidence_threshold = float(contract.get("confidence_threshold", 0.0))
    if not 0.0 <= confidence_threshold <= 1.0:
        raise AdapterContractError("pose confidence_threshold must be between 0 and 1")
    coordinate_space = str(contract["coordinate_space"]).lower()
    if array.ndim == 2 and array.shape[0] == len(names) and array.shape[1] in {2, 3, 4}:
        if coordinate_space in {"heatmap", "output"}:
            raise AdapterContractError(
                "direct coordinate output cannot declare heatmap/output coordinate_space"
            )
        for index, row in enumerate(array):
            score = float(row[2]) if row.size >= 3 else 1.0
            if score < confidence_threshold:
                continue
            keypoints.append({
                "id": index,
                "name": str(names[index]),
                "x": float(row[0]),
                "y": float(row[1]),
                "score": score,
            })
    elif array.ndim == 3 and array.shape[0] == len(names):
        if coordinate_space not in {"heatmap", "output"}:
            raise AdapterContractError(
                "heatmap output must declare coordinate_space as heatmap or output"
            )
        heatmap_shape = list(array.shape[1:])
        for index, heatmap in enumerate(array):
            flat = int(np.argmax(heatmap))
            y, x = np.unravel_index(flat, heatmap.shape)
            score = float(heatmap[y, x])
            if score < confidence_threshold:
                continue
            keypoints.append({
                "id": index,
                "name": str(names[index]),
                "x": float(x),
                "y": float(y),
                "score": score,
            })
    else:
        raise AdapterContractError(f"pose output does not match coordinates or heatmaps: {array.shape}")
    name_to_id = {str(keypoint_name): index for index, keypoint_name in enumerate(names)}
    edges: list[list[int]] = []
    for edge in contract["skeleton_edges"]:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise AdapterContractError("each skeleton edge must contain two keypoint ids or names")
        converted: list[int] = []
        for endpoint in edge:
            if isinstance(endpoint, str) and endpoint in name_to_id:
                converted.append(name_to_id[endpoint])
            else:
                try:
                    converted.append(int(endpoint))
                except (TypeError, ValueError) as exc:
                    raise AdapterContractError(f"unknown skeleton endpoint: {endpoint!r}") from exc
        if any(endpoint < 0 or endpoint >= len(names) for endpoint in converted):
            raise AdapterContractError(f"skeleton edge is out of range: {edge!r}")
        edges.append(converted)
    return {
        "task": "pose",
        "output_name": name,
        "coordinate_space": coordinate_space,
        "keypoints": keypoints,
        "skeleton_edges": edges,
        "heatmap_shape": heatmap_shape,
    }


def _depth(spec: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    name, tensor = _output(spec, outputs, require_finite=False)
    contract = spec["task_contract"]
    depth = np.squeeze(tensor).astype(np.float32)
    if depth.ndim != 2:
        raise AdapterContractError(f"depth output must reduce to a 2D map, got {depth.shape}")
    if bool(contract["inverse_depth"]):
        depth = np.divide(1.0, depth, out=np.full_like(depth, np.nan), where=depth != 0)
    depth *= float(contract["scale"])
    invalid_value = contract["invalid_value"]
    valid = np.isfinite(depth)
    if invalid_value is not None:
        valid &= depth != float(invalid_value)
    if not np.any(valid):
        raise AdapterContractError("depth output contains no valid values")
    try:
        render_min = float(contract["render_min"])
        render_max = float(contract["render_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterContractError(
            "depth task_contract must declare numeric render_min and render_max"
        ) from exc
    if not np.isfinite(render_min) or not np.isfinite(render_max) or render_max <= render_min:
        raise AdapterContractError("depth render_min/render_max must be finite and ascending")
    return {
        "task": "depth",
        "output_name": name,
        "depth": depth,
        "valid_mask": valid,
        "shape": list(depth.shape),
        "units": contract["units"],
        "minimum": float(depth[valid].min()),
        "maximum": float(depth[valid].max()),
        "render_minimum": render_min,
        "render_maximum": render_max,
    }


def decode_task(spec: dict[str, Any], outputs: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    del context
    decoders: dict[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]] = {
        "classification": _classification,
        "segmentation": _segmentation,
        "pose": _pose,
        "depth": _depth,
    }
    task = str(spec["task"])
    if task not in decoders:
        raise AdapterContractError(f"no executable type adapter for task {task!r}")
    return decoders[task](spec, outputs)


def _frame(frame: Any) -> np.ndarray:
    image = np.asarray(frame).copy()
    if image.ndim != 3 or image.shape[2] < 3:
        raise AdapterContractError(f"render frame must be HWC with at least 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _resize_nearest(array: np.ndarray, height: int, width: int) -> np.ndarray:
    ys = np.minimum((np.arange(height) * array.shape[0] / height).astype(int), array.shape[0] - 1)
    xs = np.minimum((np.arange(width) * array.shape[1] / width).astype(int), array.shape[1] - 1)
    return array[np.ix_(ys, xs)]


def _line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: np.ndarray) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(int)
    ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(int)
    valid = (xs >= 0) & (ys >= 0) & (xs < image.shape[1]) & (ys < image.shape[0])
    image[ys[valid], xs[valid], :3] = color


_GLYPHS: dict[str, tuple[str, ...]] = {
    " ": ("000", "000", "000", "000", "000"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("010", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("011", "100", "101", "101", "011"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "010"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("010", "101", "101", "101", "010"),
    "P": ("110", "101", "110", "100", "100"),
    "Q": ("010", "101", "101", "111", "011"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
    ".": ("000", "000", "000", "000", "010"),
    "-": ("000", "000", "111", "000", "000"),
    "_": ("000", "000", "000", "000", "111"),
    "/": ("001", "001", "010", "100", "100"),
    ":": ("000", "010", "000", "010", "000"),
    "%": ("101", "001", "010", "100", "101"),
    "?": ("110", "001", "010", "000", "010"),
}


def _draw_text(image: np.ndarray, text: str, x: int, y: int, *, scale: int) -> None:
    cursor = x
    color = np.array([255, 255, 255], dtype=np.uint8)
    for character in text.upper():
        glyph = _GLYPHS.get(character, _GLYPHS["?"])
        for glyph_y, row in enumerate(glyph):
            for glyph_x, value in enumerate(row):
                if value != "1":
                    continue
                x0, y0 = cursor + glyph_x * scale, y + glyph_y * scale
                x1 = min(x0 + scale, image.shape[1])
                y1 = min(y0 + scale, image.shape[0])
                if x0 < image.shape[1] and y0 < image.shape[0]:
                    image[y0:y1, x0:x1, :3] = color
        cursor += 4 * scale
        if cursor >= image.shape[1]:
            break


def render_task(spec: dict[str, Any], frame: Any, decoded: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    image = _frame(frame)
    task = str(spec["task"])
    if task == "classification":
        predictions = decoded.get("predictions", [])
        width = image.shape[1]
        scale = 2 if image.shape[0] >= 120 and width >= 320 else 1
        row_height = 5 * scale + 4
        for row, prediction in enumerate(predictions):
            y0 = 2 + row * row_height
            if y0 >= image.shape[0]:
                break
            label = str(prediction.get("label", prediction.get("class_id", "?")))
            score = float(prediction["score"])
            text = f"{label} {score:.3f}"
            text_width = min(width, (len(text) * 4 - 1) * scale + 4)
            image[y0:min(y0 + 5 * scale, image.shape[0]), :text_width, :3] //= 4
            _draw_text(image, text, 2, y0, scale=scale)
            bar_y0 = y0 + 5 * scale + 1
            y1 = min(bar_y0 + 2, image.shape[0])
            x1 = min(width, max(1, int(float(prediction["score"]) * width)))
            image[bar_y0:y1, :x1, :3] = np.array([0, 220, 80], dtype=np.uint8)
    elif task == "segmentation":
        mask = _resize_nearest(np.asarray(decoded["mask"]), image.shape[0], image.shape[1])
        colors = np.stack(((mask * 53) % 256, (mask * 97) % 256, (mask * 193) % 256), axis=-1)
        valid = mask != int(spec["task_contract"]["ignore_label"])
        image[..., :3][valid] = (
            0.55 * image[..., :3][valid] + 0.45 * colors[valid]
        ).astype(np.uint8)
    elif task == "pose":
        points = decoded["keypoints"]
        by_id = {int(point["id"]): point for point in points}
        coordinate_space = str(decoded["coordinate_space"]).lower()
        normalized = coordinate_space in {"normalized", "unit", "0-1"}
        heatmap_shape = decoded.get("heatmap_shape", [])
        def pixel(point: dict[str, Any]) -> tuple[int, int]:
            x, y = float(point["x"]), float(point["y"])
            if normalized:
                x *= image.shape[1] - 1
                y *= image.shape[0] - 1
            elif coordinate_space in {"heatmap", "output"}:
                if len(heatmap_shape) != 2:
                    raise AdapterContractError("heatmap coordinate space requires heatmap_shape")
                x *= (image.shape[1] - 1) / max(int(heatmap_shape[1]) - 1, 1)
                y *= (image.shape[0] - 1) / max(int(heatmap_shape[0]) - 1, 1)
            elif coordinate_space not in {"pixel", "pixels", "image"}:
                raise AdapterContractError(f"unsupported coordinate_space: {coordinate_space}")
            return int(round(x)), int(round(y))
        for edge in decoded["skeleton_edges"]:
            if len(edge) == 2 and int(edge[0]) in by_id and int(edge[1]) in by_id:
                _line(image, pixel(by_id[int(edge[0])]), pixel(by_id[int(edge[1])]), np.array([0, 255, 255], dtype=np.uint8))
        for point in points:
            x, y = pixel(point)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[max(0, y - 2):y + 3, max(0, x - 2):x + 3, :3] = np.array([255, 60, 0], dtype=np.uint8)
    elif task == "depth":
        depth = _resize_nearest(np.asarray(decoded["depth"]), image.shape[0], image.shape[1])
        valid = _resize_nearest(np.asarray(decoded["valid_mask"]), image.shape[0], image.shape[1]).astype(bool)
        low, high = float(decoded["render_minimum"]), float(decoded["render_maximum"])
        norm = np.zeros_like(depth, dtype=np.float32) if high <= low else np.clip((depth - low) / (high - low), 0, 1)
        colors = np.stack((255 * norm, 255 * (1 - np.abs(2 * norm - 1)), 255 * (1 - norm)), axis=-1).astype(np.uint8)
        image[..., :3][valid] = colors[valid]
    else:
        raise AdapterContractError(f"no renderer for task {task!r}")
    return image


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def serialize_task(spec: dict[str, Any], decoded: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    identity = {key: context[key] for key in ("source_id", "frame_id", "roi_id", "timestamp") if key in context}
    return {"schema_version": 1, "task": spec["task"], "identity": identity, "result": _json_value(decoded)}


def input_adapter_task(spec: dict[str, Any], payload: Any, context: dict[str, Any]) -> Any:
    if spec["input_domain"] not in {"image", "tensor"}:
        raise AdapterContractError(f"input_domain {spec['input_domain']!r} requires a dedicated owner")
    if spec["input_domain"] == "tensor":
        return payload
    array = np.asarray(payload)
    if array.ndim not in {3, 4}:
        raise AdapterContractError(f"image payload must be HWC/CHW or batched, got {array.shape}")
    payload_layout = str(context.get("payload_layout", "NHWC" if array.ndim == 4 else "HWC")).upper()
    source_color = str(context.get("source_color_format", spec["color_format"])).upper()
    target_color = str(spec["color_format"]).upper()
    if source_color != target_color:
        if {source_color, target_color} == {"RGB", "BGR"}:
            channel_axis = 1 if payload_layout in {"CHW", "NCHW"} else -1
            array = np.flip(array, axis=channel_axis)
        else:
            raise AdapterContractError(f"unsupported color conversion: {source_color} to {target_color}")
    values = array.astype(np.float32)
    channel_axis = 1 if payload_layout == "NCHW" else 0 if payload_layout == "CHW" else values.ndim - 1
    channels = values.shape[channel_axis]
    mean = list(spec.get("mean", []))
    std = list(spec.get("std", []))
    if len(mean) == 1:
        mean *= channels
    if len(std) == 1:
        std *= channels
    if mean and len(mean) != channels:
        raise AdapterContractError(f"mean has {len(mean)} values for {channels} channels")
    if std and len(std) != channels:
        raise AdapterContractError(f"std has {len(std)} values for {channels} channels")
    if any(float(value) == 0.0 for value in std):
        raise AdapterContractError("std values must be nonzero")
    broadcast_shape = [1] * values.ndim
    broadcast_shape[channel_axis] = channels
    units = str(spec.get("preprocess_units", "pixel")).lower()
    scale = float(spec.get("scale", 1.0))
    if units == "normalized":
        values *= scale
        if mean:
            values -= np.asarray(mean, dtype=np.float32).reshape(broadcast_shape)
        if std:
            values /= np.asarray(std, dtype=np.float32).reshape(broadcast_shape)
    elif units == "pixel":
        if mean:
            values -= np.asarray(mean, dtype=np.float32).reshape(broadcast_shape)
        if std:
            values /= np.asarray(std, dtype=np.float32).reshape(broadcast_shape)
        else:
            values *= scale
    else:
        raise AdapterContractError(f"unsupported preprocess_units: {units}")
    target_layout = str(spec["layout"]).upper()
    if payload_layout in {"HWC", "NHWC"} and target_layout == "NCHW":
        values = np.transpose(values, (2, 0, 1)) if values.ndim == 3 else np.transpose(values, (0, 3, 1, 2))
    elif payload_layout in {"CHW", "NCHW"} and target_layout == "NHWC":
        values = np.transpose(values, (1, 2, 0)) if values.ndim == 3 else np.transpose(values, (0, 2, 3, 1))
    if values.ndim == 3 and len(spec.get("input_shape", [])) == 4:
        values = values[np.newaxis, ...]
    return np.ascontiguousarray(values)


def build_pipeline_task(spec: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    preprocess_stage = (
        "nvinfer-native-preprocess"
        if spec.get("preprocess_strategy") == "nvinfer-native"
        else "adapter-input-tensor"
    )
    return {
        "task": spec["task"],
        "input_domain": spec["input_domain"],
        "stages": [
            "source",
            "decode",
            preprocess_stage,
            "nvinfer",
            "metadata-probe",
            "decode-output",
            "render",
            "output-sink",
        ],
        "bounded_timeout_seconds": int(context.get("timeout_seconds", 300)),
        "required_gates": ["startup", "nonzero-buffers", "task-metadata", "eos", "structured-output", "visible-overlay"],
    }


def metadata_probe_task(spec: dict[str, Any], tensor_meta: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    decoded = decode_task(spec, tensor_meta, context)
    return serialize_task(spec, decoded, context)


def output_sink_task(spec: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    del spec
    available = set(context.get("available_elements", []))
    if "nvv4l2h264enc" in available:
        return {"elements": ["nvv4l2h264enc", "h264parse", "qtmux", "filesink"], "container": "mp4"}
    if {"theoraenc", "oggmux"}.issubset(available):
        return {"elements": ["theoraenc", "oggmux", "filesink"], "container": "ogg"}
    raise AdapterContractError("no supported video encoder is available; visual status is UNTESTED")


def _reference_gate_statuses(
    result: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, str]:
    evidence = summary.get("evidence")
    if not isinstance(evidence, Mapping):
        raise AdapterContractError("classification reference runner summary lacks evidence")
    try:
        structured_path = Path(str(evidence["structured_output"]))
        visual_path = Path(str(evidence["visual_validation"]))
        visual = json.loads(visual_path.read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise AdapterContractError(
            "classification reference runner evidence is incomplete or unreadable"
        ) from exc
    if not isinstance(visual, Mapping):
        raise AdapterContractError(
            "classification reference visual evidence must be a result mapping"
        )
    return {
        "startup": "PASS" if bool(result.get("startup")) else "FAIL",
        "nonzero-buffers": "PASS" if int(result.get("buffer_count", 0)) > 0 else "FAIL",
        "task-metadata": "PASS" if bool(result.get("task_metadata")) else "FAIL",
        "eos": "PASS" if bool(result.get("eos")) else "FAIL",
        "structured-output": "PASS"
        if structured_path.is_file() and structured_path.stat().st_size > 0
        else "FAIL",
        "visible-overlay": "PASS"
        if str(visual.get("visual_status", "")).upper() == "PASS"
        and str(visual.get("pipeline_visual_result", "")).upper() == "PASS"
        else "FAIL",
    }


def run_pipeline_task(spec: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    plan = build_pipeline_task(spec, context)
    runner = context.get("run_deepstream")
    if callable(runner):
        result = runner(plan, context)
    else:
        reference = str(spec.get("reference_pipeline_runner", "")).strip()
        if spec.get("task") != "classification" or not reference:
            raise AdapterContractError(
                "no shipped DeepStream reference runner for this task; "
                "context.run_deepstream callable is required"
            )
        required_context = ("manifest", "nvinfer_config", "input", "output_dir", "campaign_id")
        missing_context = [name for name in required_context if context.get(name) in (None, "")]
        if missing_context:
            raise AdapterContractError(
                "classification reference runner context is missing: " + ", ".join(missing_context)
            )
        command = [
            sys.executable, reference, "--manifest", str(context["manifest"]),
            "--nvinfer-config", str(context["nvinfer_config"]), "--input", str(context["input"]),
            "--output-dir", str(context["output_dir"]), "--campaign-id", str(context["campaign_id"]),
            "--timeout", str(min(int(context.get("timeout_seconds", 300)), 300)),
        ]
        if context.get("oracle"):
            command.extend(["--oracle", str(context["oracle"])])
        child_timeout = min(int(context.get("timeout_seconds", 300)), 300)
        parent_timeout = child_timeout + 15
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=parent_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdapterContractError(
                "classification reference runner exceeded its bounded parent "
                f"timeout of {parent_timeout} seconds"
            ) from exc
        if completed.returncode != 0:
            raise AdapterContractError(
                "classification reference runner failed: " + (completed.stderr.strip() or completed.stdout.strip())
            )
        try:
            summary = json.loads(completed.stdout)
            if not isinstance(summary, Mapping):
                raise ValueError("summary is not a mapping")
            evidence = summary.get("evidence")
            if not isinstance(evidence, Mapping):
                raise ValueError("summary lacks evidence")
            runtime_path = Path(str(evidence["deepstream_runtime"]))
            result = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise AdapterContractError(
                "classification reference runner returned unreadable runtime evidence"
            ) from exc
        if not isinstance(result, Mapping):
            raise AdapterContractError(
                "classification reference runtime evidence must be a result mapping"
            )
        result = {**result, "gate_statuses": _reference_gate_statuses(result, summary)}
    if not isinstance(result, Mapping):
        raise AdapterContractError("DeepStream runner must return a result mapping")

    gate_statuses = result.get("gate_statuses")
    if not isinstance(gate_statuses, Mapping):
        raise AdapterContractError(
            "DeepStream result must include a gate_statuses mapping for every required gate"
        )
    required_gates = tuple(str(gate) for gate in plan["required_gates"])
    missing = [gate for gate in required_gates if gate not in gate_statuses]
    if missing:
        raise AdapterContractError(
            "DeepStream result gate_statuses is missing: " + ", ".join(missing)
        )
    failed = [gate for gate in required_gates if str(gate_statuses[gate]).upper() != "PASS"]
    passed = str(result.get("status", "")).upper() == "PASS" and not failed
    return {
        **result,
        "required_gates": list(required_gates),
        "deepstream_status": "PASS" if passed else "FAIL",
    }

def validate_task(spec: dict[str, Any], decoded: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    if not oracle:
        return {"status": "UNTESTED", "reason": "semantic oracle was not supplied"}
    task = str(spec["task"])
    if task == "classification":
        actual = decoded.get("predictions", [{}])[0].get("class_id") if decoded.get("predictions") else None
        expected = oracle.get("class_id")
        passed = (
            actual is not None
            and expected is not None
            and int(actual) == int(expected)
        )
        metric = {"actual_class_id": actual, "expected_class_id": expected}
    elif task == "segmentation":
        expected = np.asarray(oracle.get("mask"))
        actual = np.asarray(decoded["mask"])
        passed = expected.shape == actual.shape and bool(np.array_equal(expected, actual))
        metric = {"pixel_agreement": float(np.mean(expected == actual)) if expected.shape == actual.shape else 0.0}
    elif task == "pose":
        expected = oracle.get("keypoints", [])
        tolerance = float(oracle.get("tolerance", 1.0))
        actual = decoded["keypoints"]
        errors = [np.hypot(float(a["x"]) - float(e["x"]), float(a["y"]) - float(e["y"])) for a, e in zip(actual, expected)]
        passed = len(expected) == len(actual) and bool(errors) and max(errors) <= tolerance
        metric = {"max_keypoint_error": float(max(errors)) if errors else None, "tolerance": tolerance}
    elif task == "depth":
        expected = np.asarray(oracle.get("depth"), dtype=np.float32)
        actual = np.asarray(decoded["depth"], dtype=np.float32)
        tolerance = float(oracle.get("max_rmse", 0.0))
        rmse = float(np.sqrt(np.mean(np.square(actual - expected)))) if expected.shape == actual.shape else float("inf")
        passed = expected.shape == actual.shape and rmse <= tolerance
        metric = {"rmse": rmse, "max_rmse": tolerance}
    else:
        raise AdapterContractError(f"no semantic validator for task {task!r}")
    return {"status": "PASS" if passed else "FAIL", "metric": metric}


def write_structured_output(
    payload: dict[str, Any],
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    path = Path(destination)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace structured output without overwrite=True: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
