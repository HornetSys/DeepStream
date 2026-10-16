#!/usr/bin/env python3
"""Render model inputs and outputs for a visual I/O smoke gate.

This is intentionally a diagnostic gate, not a semantic correctness gate. It
uses a deterministic, non-zero synthetic fixture so constant/empty tensors and
channel/layout mistakes are visible. Image-like tensors are rendered as RGB or
heatmaps; vectors are rendered as bars; flow, depth, and segmentation outputs
get task-aware reductions. Raw detection/pose/feature tensors are labelled as
such because a decoder is still required for semantic validation.

The command consumes the measured inventory produced by the model-zoo audit and
writes evidence outside the skill repository by default:

    python3 scripts/multitask/visual/visual_io_probe.py \
      /colossus/deepstream-zoo-e2e/inventory.json \
      --output-dir /colossus/deepstream-zoo-e2e/visual-io

An output ``visual-report.json`` is produced for every catalog case. A case is
PASS only when the model executes and both its input and every runtime output
have a rendered visual representation. Missing artifacts or unsupported ONNX
operators remain UNTESTED; a placeholder is never counted as a runtime output.
"""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SPECIAL_SHAPES: dict[str, dict[str, tuple[int, ...]]] = {
    "sam2-memory-attention": {
        "current_vision_feat": (1, 256, 64, 64),
        "current_vision_pos_embed": (4096, 1, 256),
        "memory_0": (1, 16, 256),
        "memory_1": (1, 7, 64, 64, 64),
        "memory_pos_embed": (1, 7 * 4096 + 64, 64),
        "cond_frame_id_diff": (),
    },
}


ONNX_DTYPE_TO_NUMPY = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
    12: np.uint32,
    13: np.uint64,
}


def _concrete_dim(dim: Any) -> int:
    if isinstance(dim, (int, np.integer)) and int(dim) > 0:
        return int(dim)
    text = str(dim).strip().lower()
    if text.isdigit() and int(text) > 0:
        return int(text)
    # A deterministic batch/sequence/profile fixture. The catalog must still
    # provide an explicit TensorRT profile before an engine build.
    return 1


def concrete_shape(shape: Iterable[Any], entry_id: str, name: str) -> tuple[int, ...]:
    special = SPECIAL_SHAPES.get(entry_id, {}).get(name)
    if special is not None:
        return special
    return tuple(_concrete_dim(dim) for dim in shape)


def numpy_dtype(dtype: Any) -> np.dtype:
    if isinstance(dtype, (int, np.integer)):
        return np.dtype(ONNX_DTYPE_TO_NUMPY.get(int(dtype), np.float32))
    text = str(dtype).lower()
    if "float16" in text or text in {"fp16", "half"}:
        return np.dtype(np.float16)
    if "float64" in text or text in {"double", "fp64"}:
        return np.dtype(np.float64)
    if "int64" in text:
        return np.dtype(np.int64)
    if "int32" in text:
        return np.dtype(np.int32)
    if "int16" in text:
        return np.dtype(np.int16)
    if "int8" in text:
        return np.dtype(np.int8)
    if "uint64" in text:
        return np.dtype(np.uint64)
    if "uint32" in text:
        return np.dtype(np.uint32)
    if "uint16" in text:
        return np.dtype(np.uint16)
    if "uint8" in text:
        return np.dtype(np.uint8)
    if "bool" in text:
        return np.dtype(np.bool_)
    return np.dtype(np.float32)


def _spatial_pattern(shape: tuple[int, ...]) -> np.ndarray:
    if not shape:
        return np.asarray(0.25, dtype=np.float32)
    if len(shape) < 2:
        values = np.linspace(0.05, 0.95, num=max(1, shape[-1]), dtype=np.float32)
        return np.broadcast_to(values, shape).copy()
    height, width = shape[-2], shape[-1]
    y = np.linspace(0.05, 0.95, num=max(1, height), dtype=np.float32).reshape(height, 1)
    x = np.linspace(0.05, 0.95, num=max(1, width), dtype=np.float32).reshape(1, width)
    base = (0.62 * x + 0.38 * y) % 1.0
    # Add channel variation only for an unambiguous CHW/NCHW/NHWC image-like
    # contract. Other ranks are feature/sequence tensors and should remain a
    # simple spatial gradient rather than accidentally treating a sequence
    # axis as a color channel.
    if len(shape) == 3 and shape[0] <= 4 and min(shape[-2:]) >= 2:
        channels = shape[0]
        channel_shift = (np.arange(channels, dtype=np.float32).reshape(channels, 1, 1) * 0.17) % 1.0
        return (base[None, ...] + channel_shift) % 1.0
    if len(shape) == 3 and shape[-1] <= 4 and min(shape[:2]) >= 2:
        channels = shape[-1]
        channel_shift = (np.arange(channels, dtype=np.float32).reshape(1, 1, channels) * 0.17) % 1.0
        return (base[..., None] + channel_shift) % 1.0
    if len(shape) == 4 and shape[1] <= 4 and min(shape[-2:]) >= 2:
        channels = shape[1]
        channel_shift = (np.arange(channels, dtype=np.float32).reshape(1, channels, 1, 1) * 0.17) % 1.0
        return (base[None, None, ...] + channel_shift) % 1.0
    if len(shape) == 4 and shape[-1] <= 4 and min(shape[1:3]) >= 2:
        channels = shape[-1]
        channel_shift = (np.arange(channels, dtype=np.float32).reshape(1, 1, 1, channels) * 0.17) % 1.0
        return (base[None, ..., None] + channel_shift) % 1.0
    leading = (1,) * (len(shape) - 2)
    return np.broadcast_to(base.reshape(leading + (height, width)), shape).copy()


def make_fixture(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if dtype.kind in "iu":
        values = np.arange(max(1, math.prod(shape)), dtype=np.int64).reshape(shape or ()) % 5
        return values.astype(dtype, copy=False)
    if dtype.kind == "b":
        return np.ones(shape or (), dtype=dtype)
    return _spatial_pattern(shape).astype(dtype, copy=False)


def fixture_for_input(entry_id: str, name: str, shape: Iterable[Any], dtype: Any) -> np.ndarray:
    concrete = concrete_shape(shape, entry_id, name)
    return make_fixture(concrete, numpy_dtype(dtype))


def fixture_hash(inputs: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(inputs):
        value = np.ascontiguousarray(inputs[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _finite_values(value: np.ndarray) -> np.ndarray:
    numeric = np.asarray(value)
    if numeric.dtype.kind not in "biufc":
        return np.asarray([], dtype=np.float32)
    values = numeric.astype(np.float32, copy=False).reshape(-1)
    return values[np.isfinite(values)]


def tensor_stats(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    finite = _finite_values(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "finite": bool(finite.size == array.size),
        "finite_ratio": round(float(finite.size / max(1, array.size)), 6),
        "min": round(float(finite.min()), 6) if finite.size else None,
        "max": round(float(finite.max()), 6) if finite.size else None,
        "mean": round(float(finite.mean()), 6) if finite.size else None,
        "std": round(float(finite.std()), 6) if finite.size else None,
    }


def _normalise(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    safe = np.where(finite, values, 0.0)
    samples = safe[finite]
    low, high = np.percentile(samples, [1, 99]) if samples.size > 8 else (samples.min(), samples.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(samples.min()), float(samples.max())
    if high <= low:
        return np.full(values.shape, 128, dtype=np.uint8)
    scaled = np.clip((safe - low) / (high - low), 0.0, 1.0) * 255.0
    return scaled.astype(np.uint8)


def _heatmap(value: np.ndarray) -> Image.Image:
    gray = _normalise(value).astype(np.float32) / 255.0
    red = np.clip(1.65 * gray, 0.0, 1.0)
    green = np.clip(1.65 * (1.0 - np.abs(2.0 * gray - 1.0)), 0.0, 1.0)
    blue = np.clip(1.65 * (1.0 - gray), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def _flow_image(value: np.ndarray) -> Image.Image:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 2:
        horizontal, vertical = array[0], array[1]
    elif array.ndim == 3 and array.shape[-1] == 2:
        horizontal, vertical = array[..., 0], array[..., 1]
    else:
        return _heatmap(array)
    magnitude = np.sqrt(horizontal * horizontal + vertical * vertical)
    angle = (np.arctan2(vertical, horizontal) + np.pi) / (2.0 * np.pi)
    brightness = _normalise(magnitude).astype(np.float32) / 255.0
    # A lightweight HSV-like flow wheel that needs no plotting dependency.
    red = np.abs(angle * 6.0 - 3.0) - 1.0
    green = 2.0 - np.abs(angle * 6.0 - 2.0)
    blue = 2.0 - np.abs(angle * 6.0 - 4.0)
    rgb = np.clip(np.stack([red, green, blue], axis=-1), 0.0, 1.0)
    rgb = 0.25 + 0.75 * rgb * brightness[..., None]
    return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")


def _first_2d(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 2:
        array = array[0] if array.shape[0] == 1 else array.mean(axis=0)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def _spatial_image(value: np.ndarray, task: str, direction: str) -> tuple[Image.Image, str]:
    array = np.asarray(value)
    if array.ndim >= 4 and array.shape[0] == 1:
        array = array[0]
    if task == "optical-flow" and array.ndim == 3 and 2 in (array.shape[0], array.shape[-1]):
        return _flow_image(array), "flow-wheel"
    if task in {"depth", "stereo-depth"}:
        return _heatmap(_first_2d(array)), "depth-heatmap"

    # NCHW image or two-channel spatial tensor.
    if array.ndim == 3 and array.shape[0] <= 4 and min(array.shape[-2:]) >= 2:
        channels = array.astype(np.float32)
        if channels.shape[0] == 1:
            return Image.fromarray(_normalise(channels[0]), mode="L").convert("RGB"), "gray-image"
        if channels.shape[0] >= 3:
            rgb = np.stack([_normalise(channels[i]) for i in range(3)], axis=-1)
            return Image.fromarray(rgb, mode="RGB"), "rgb-image"
        return _heatmap(channels.mean(axis=0)), "two-channel-heatmap"

    # NHWC image.
    if array.ndim == 3 and array.shape[-1] <= 4 and min(array.shape[:2]) >= 2:
        channels = array.astype(np.float32)
        if channels.shape[-1] == 1:
            return Image.fromarray(_normalise(channels[..., 0]), mode="L").convert("RGB"), "gray-image"
        rgb = np.stack([_normalise(channels[..., i]) for i in range(3)], axis=-1)
        return Image.fromarray(rgb, mode="RGB"), "rgb-image"

    # For class/feature tensors, show the channel winner as a categorical
    # pseudo-colour map. This makes dead channels and all-zero outputs obvious,
    # but does not claim a decoded semantic mask.
    if array.ndim == 3 and array.shape[0] > 4 and min(array.shape[-2:]) >= 2:
        classes = np.argmax(array.astype(np.float32), axis=0)
        colors = np.asarray(
            [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
             [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]],
            dtype=np.uint8,
        )
        return Image.fromarray(colors[classes % len(colors)], mode="RGB"), "channel-argmax-pseudocolor"
    return _heatmap(_first_2d(array)), "tensor-heatmap"


def _bar_image(value: np.ndarray) -> Image.Image:
    values = _finite_values(np.asarray(value)).astype(np.float32)
    if values.size == 0:
        values = np.zeros(1, dtype=np.float32)
    if values.size > 64:
        indices = np.linspace(0, values.size - 1, 64).astype(int)
        values = values[indices]
    low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    width, height = 440, 200
    image = Image.new("RGB", (width, height), (248, 249, 252))
    draw = ImageDraw.Draw(image)
    zero = int((high / (high - low)) * (height - 30)) if low < 0 < high else height - 22
    zero = max(15, min(height - 15, zero))
    draw.line((10, zero, width - 10, zero), fill=(120, 125, 135), width=1)
    bar_width = max(2, (width - 24) // len(values))
    for index, item in enumerate(values):
        x0 = 12 + index * bar_width
        x1 = max(x0 + 1, x0 + bar_width - 1)
        y = int((item - low) / (high - low) * (height - 30))
        y = height - 15 - y
        color = (30, 120, 220) if item >= 0 else (220, 75, 55)
        draw.rectangle((x0, min(y, zero), x1, max(y, zero)), fill=color)
    return image


def _render_content(value: np.ndarray, task: str, direction: str) -> tuple[Image.Image, str]:
    array = np.asarray(value)
    if array.size == 0:
        return Image.new("RGB", (440, 200), (235, 235, 235)), "empty-tensor"
    if array.ndim <= 2:
        return _bar_image(array), "vector-bars"
    return _spatial_image(array, task, direction)


def render_tile(name: str, value: np.ndarray, task: str, direction: str, width: int = 500, height: int = 300) -> tuple[Image.Image, dict[str, Any]]:
    stats = tensor_stats(value)
    content, mode = _render_content(value, task, direction)
    tile = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    title = f"{direction.upper()}  {name}"
    summary = f"shape={tuple(stats['shape'])} dtype={stats['dtype']}"
    range_text = f"min={stats['min']} max={stats['max']} finite={stats['finite_ratio']}"
    draw.text((10, 8), title, fill=(20, 25, 35), font=ImageFont.load_default())
    draw.text((10, 24), summary, fill=(60, 65, 75), font=ImageFont.load_default())
    draw.text((10, 40), range_text, fill=(60, 65, 75), font=ImageFont.load_default())
    content.thumbnail((width - 20, height - 68), Image.Resampling.BILINEAR)
    x = (width - content.width) // 2
    y = 62 + (height - 62 - content.height) // 2
    tile.paste(content, (x, y))
    return tile, {"mode": mode, "stats": stats}


def _placeholder_tile(text: str, width: int = 500, height: int = 300) -> Image.Image:
    tile = Image.new("RGB", (width, height), (255, 248, 238))
    draw = ImageDraw.Draw(tile)
    draw.text((16, 16), "OUTPUT UNTESTED", fill=(160, 75, 15), font=ImageFont.load_default())
    lines = [text[i : i + 65] for i in range(0, len(text), 65)]
    for index, line in enumerate(lines[:10]):
        draw.text((16, 45 + index * 16), line, fill=(90, 65, 45), font=ImageFont.load_default())
    return tile


def _case_canvas(entry: dict[str, Any], input_tiles: list[Image.Image], output_tiles: list[Image.Image], status: str, note: str) -> Image.Image:
    tile_width, tile_height, columns = 500, 300, 2
    input_rows = max(1, math.ceil(len(input_tiles) / columns))
    output_rows = max(1, math.ceil(len(output_tiles) / columns))
    header_height = 94
    section_height = 34
    canvas = Image.new(
        "RGB",
        (tile_width * columns, header_height + section_height * 2 + tile_height * (input_rows + output_rows)),
        (239, 242, 247),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), f"{entry['id']}  [{entry.get('task', 'unknown')}]  visual I/O: {status}", fill=(15, 25, 40), font=ImageFont.load_default())
    draw.text((14, 31), f"artifact: {entry.get('materialized_path') or entry.get('catalog_artifact') or 'missing'}", fill=(55, 65, 78), font=ImageFont.load_default())
    draw.text((14, 50), f"fixture: deterministic synthetic non-zero pattern; {note}", fill=(55, 65, 78), font=ImageFont.load_default())
    draw.text((14, 69), "Visuals are diagnostic reductions; semantic parity still requires task-specific reference/decoder gates.", fill=(55, 65, 78), font=ImageFont.load_default())
    y = header_height
    for label, tiles, rows in (("INPUTS", input_tiles, input_rows), ("OUTPUTS", output_tiles, output_rows)):
        draw.text((14, y + 8), label, fill=(25, 70, 130), font=ImageFont.load_default())
        y += section_height
        for index, tile in enumerate(tiles):
            x = (index % columns) * tile_width
            row = index // columns
            canvas.paste(tile, (x, y + row * tile_height))
        y += rows * tile_height
    return canvas


def _inventory_input_specs(entry: dict[str, Any]) -> list[dict[str, Any]]:
    inspection = entry.get("inspection") or {}
    return list(inspection.get("inputs") or [])


def _inventory_output_specs(entry: dict[str, Any]) -> list[dict[str, Any]]:
    inspection = entry.get("inspection") or {}
    return list(inspection.get("outputs") or [])


def run_case(entry: dict[str, Any], ort: Any, output_dir: Path) -> dict[str, Any]:
    case_id = str(entry["id"])
    case_dir = output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "id": case_id,
        "task": entry.get("task"),
        "path": entry.get("materialized_path"),
        "input_visual_status": "UNTESTED",
        "output_visual_status": "UNTESTED",
        "result": "UNTESTED",
        "inputs": [],
        "outputs": [],
    }
    input_tiles: list[Image.Image] = []
    output_tiles: list[Image.Image] = []
    started = time.time()
    path = Path(entry["materialized_path"]) if entry.get("materialized_path") else None
    session = None
    runtime_error = ""
    runtime_inputs: dict[str, np.ndarray] = {}
    runtime_outputs: list[np.ndarray] = []
    runtime_output_names: list[str] = []

    if path is None or not path.exists() or path.suffix.lower() != ".onnx":
        runtime_error = "missing or unsupported materialized ONNX artifact"
    else:
        try:
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            metas = session.get_inputs()
            runtime_inputs = {
                meta.name: fixture_for_input(case_id, meta.name, meta.shape, meta.type)
                for meta in metas
            }
            runtime_outputs = session.run(None, runtime_inputs)
            runtime_output_names = [meta.name for meta in session.get_outputs()]
        except Exception as exc:  # noqa: BLE001 - report first causal runtime error
            runtime_error = str(exc).splitlines()[0]

    # Always render a real synthetic input contract when graph metadata is
    # available. If ORT cannot load the graph, the JSON clearly marks it as a
    # contract-only input visual and never promotes the output to PASS.
    input_specs = []
    if runtime_inputs:
        input_specs = [
            {"name": name, "value": value, "executed": True}
            for name, value in runtime_inputs.items()
        ]
    else:
        input_specs = [
            {
                "name": spec.get("name", f"input_{index}"),
                "value": fixture_for_input(case_id, spec.get("name", f"input_{index}"), spec.get("shape", []), spec.get("dtype", 1)),
                "executed": False,
            }
            for index, spec in enumerate(_inventory_input_specs(entry))
        ]
    for spec in input_specs:
        tile, rendered = render_tile(spec["name"], spec["value"], str(entry.get("task") or "generic"), "input")
        input_tiles.append(tile)
        row["inputs"].append({
            "name": spec["name"],
            "executed": bool(spec["executed"]),
            "fixture_shape": list(spec["value"].shape),
            "fixture_dtype": str(spec["value"].dtype),
            "fixture_visual": rendered,
        })
    input_ok = (
        bool(row["inputs"])
        and all(item["executed"] is True for item in row["inputs"])
        and all(item["fixture_visual"]["stats"]["finite"] for item in row["inputs"])
    )
    row["input_visual_status"] = "PASS" if input_ok else "UNTESTED"
    row["fixture_sha256"] = fixture_hash({spec["name"]: spec["value"] for spec in input_specs}) if input_specs else None

    if runtime_outputs:
        for name, value in zip(runtime_output_names, runtime_outputs):
            tile, rendered = render_tile(name, value, str(entry.get("task") or "generic"), "output")
            output_tiles.append(tile)
            row["outputs"].append({
                "name": name,
                "executed": True,
                "runtime_visual": {"name": name, **rendered},
            })
        output_ok = len(output_tiles) == len(runtime_outputs) and all(
            item["runtime_visual"]["stats"]["finite"] for item in row["outputs"]
        )
        row["output_visual_status"] = "PASS" if output_ok else "FAIL"
        row["constant_output_warnings"] = [
            item["name"] for item in row["outputs"]
            if item.get("executed") and item["runtime_visual"]["stats"]["min"] == item["runtime_visual"]["stats"]["max"]
        ]
    else:
        reason = runtime_error or "no runtime outputs"
        output_tiles = [_placeholder_tile(reason)]
        row["output_visual_status"] = "UNTESTED"
        row["output_unavailable_reason"] = reason
        for spec in _inventory_output_specs(entry):
            row["outputs"].append({
                "name": spec.get("name"),
                "executed": False,
                "runtime_visual": None,
                "reason": reason,
            })

    if "FAIL" in {row["input_visual_status"], row["output_visual_status"]}:
        row["result"] = "FAIL"
    elif row["input_visual_status"] == "PASS" and row["output_visual_status"] == "PASS":
        row["result"] = "PASS"
    else:
        row["result"] = "UNTESTED"
    row["seconds"] = round(time.time() - started, 3)
    row["runtime_error"] = runtime_error or None
    image = _case_canvas(entry, input_tiles, output_tiles, row["result"], "runtime outputs rendered" if row["result"] == "PASS" else runtime_error or "incomplete visual evidence")
    image_path = case_dir / f"{case_id}.png"
    image.save(image_path)
    row["case_image"] = str(image_path)
    return row


def _make_montage(rows: list[dict[str, Any]], output_path: Path) -> None:
    thumb_width, thumb_height, columns = 360, 230, 3
    rows_count = max(1, math.ceil(len(rows) / columns))
    montage = Image.new("RGB", (thumb_width * columns, thumb_height * rows_count), (220, 225, 232))
    for index, row in enumerate(rows):
        path = row.get("case_image")
        if not path or not Path(path).exists():
            continue
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width - 10, thumb_height - 30), Image.Resampling.BILINEAR)
        x = (index % columns) * thumb_width + (thumb_width - image.width) // 2
        y = (index // columns) * thumb_height + 22
        montage.paste(image, (x, y))
        draw = ImageDraw.Draw(montage)
        draw.text(((index % columns) * thumb_width + 8, (index // columns) * thumb_height + 5), f"{row['id']} [{row['result']}]", fill=(25, 35, 50), font=ImageFont.load_default())
    montage.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider", help="ONNX Runtime provider; CPUExecutionProvider is the reproducible default")
    args = parser.parse_args()
    if args.provider != "CPUExecutionProvider":
        parser.error("only CPUExecutionProvider is supported by the deterministic visual fixture")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        print(f"onnxruntime is required: {exc}")
        return 2

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_case(entry, ort, args.output_dir) for entry in inventory.get("entries", [])]
    report = {
        "schema_version": 1,
        "generated_at_epoch": time.time(),
        "inventory": str(args.inventory),
        "provider": args.provider,
        "fixture_kind": "deterministic-synthetic-non-zero-visual-pattern",
        "contract": {
            "input": "every available graph input gets a rendered image or heatmap",
            "output": "every executed graph output gets a rendered image, heatmap, flow wheel, or vector bars",
            "semantic_limit": "rendered raw tensors are diagnostic; task decoder/parity is still required",
        },
        "entries": rows,
        "counts": {
            "cases": len(rows),
            "pass": sum(row["result"] == "PASS" for row in rows),
            "untested": sum(row["result"] == "UNTESTED" for row in rows),
            "fail": sum(row["result"] == "FAIL" for row in rows),
            "input_visual_pass": sum(row["input_visual_status"] == "PASS" for row in rows),
            "output_visual_pass": sum(row["output_visual_status"] == "PASS" for row in rows),
        },
    }
    report_path = args.output_dir / "visual-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _make_montage(rows, args.output_dir / "visual-montage.png")
    print(json.dumps(report["counts"], sort_keys=True))
    print(report_path)
    return 0 if rows and all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
