#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run the generated classification adapter in a real one-stream DeepStream graph.

This is the reviewed reference runner for the first non-detection execution
family. It consumes nvinfer tensor metadata, invokes the generated adapter,
adds model-derived display metadata before nvdsosd, records structured output,
and emits runtime and visual evidence. Other non-detection categories remain
ADAPTER_REQUIRED until equivalent runners are shipped and exercised.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any

PRODUCER = "deepstream-type-adapter-runner"
SCHEMA_VERSION = 1
EXIT_FAIL = 1
EXIT_INVALID = 2
EXIT_BLOCKED = 4


class RunnerError(RuntimeError):
    """A fail-closed pipeline or adapter contract error."""


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_prediction_scores(predictions: Any) -> bool:
    """Validate classification scores and report malformed adapter output."""

    if not isinstance(predictions, list):
        raise RunnerError("classification predictions must be a list")
    for index, item in enumerate(predictions):
        if not isinstance(item, Mapping):
            raise RunnerError(
                f"classification prediction {index} must be an object"
            )
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError) as exc:
            raise RunnerError(
                f"classification prediction {index} has an invalid score"
            ) from exc
        if not math.isfinite(score):
            return False
    return True


def pipeline_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "bounded_timeout_seconds": 300,
        "task": "classification",
        "pipeline": [
            "uridecodebin",
            "nvvideoconvert",
            "nvstreammux",
            "nvinfer(output-tensor-meta=1)",
            "NvDsInferTensorMeta probe",
            "generated adapter decode/serialize",
            "NvDsDisplayMeta label overlay",
            "nvdsosd",
            "rendered frame and encoded video sinks",
        ],
        "required_outputs": [
            "deepstream-runtime.json",
            "structured-output.json",
            "overlay-evidence.json",
            "rendered-frame.jpg",
            "rendered-video.mp4 or rendered-video.ogv",
            "visual-validation.json",
        ],
        "fail_closed": [
            "missing Gst or pyds",
            "missing tensor metadata",
            "adapter decode error",
            "zero buffers",
            "missing EOS",
            "missing rendered artifacts",
        ],
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RunnerError(f"{path} must contain a JSON object")
    return value


def _load_adapter(manifest_path: Path, manifest: Mapping[str, Any]) -> Any:
    contract = manifest.get("adapter_contract")
    if not isinstance(contract, Mapping):
        raise RunnerError("manifest adapter_contract must be an object")
    target = str(contract.get("decoder", ""))
    if ":" not in target:
        raise RunnerError("adapter_contract.decoder must be a file.py:symbol reference")
    file_name, symbol = target.rsplit(":", 1)
    adapter_path = Path(file_name)
    if not adapter_path.is_absolute():
        adapter_path = manifest_path.parent / adapter_path
    adapter_path = adapter_path.resolve()
    if not adapter_path.is_file():
        raise RunnerError(f"generated adapter is missing: {adapter_path}")
    spec = importlib.util.spec_from_file_location("generated_deepstream_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RunnerError(f"cannot import generated adapter: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(adapter_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    if symbol != "decode":
        raise RunnerError("adapter_contract.decoder must name the generated decode hook")
    if not callable(module.decode) or not callable(module.serialize):
        raise RunnerError("generated adapter must provide callable decode and serialize hooks")
    return module


def _require_validate_hook(adapter: Any) -> Any:
    """Return the generated semantic validator only when an oracle is requested."""

    try:
        validate = adapter.validate
    except AttributeError as exc:
        raise RunnerError(
            "generated adapter must provide a callable validate hook when --oracle is set"
        ) from exc
    if not callable(validate):
        raise RunnerError(
            "generated adapter must provide a callable validate hook when --oracle is set"
        )
    return validate


def _pipeline_description(encoder: str, *, mux_width: int = 1280, mux_height: int = 720) -> str:
    if encoder == "h264":
        video_chain = (
            "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
            "nvv4l2h264enc ! h264parse ! qtmux ! filesink name=video_sink"
        )
    else:
        video_chain = (
            "nvvideoconvert ! video/x-raw,format=I420 ! "
            "theoraenc ! oggmux ! filesink name=video_sink"
        )
    return " ".join(
        [
            "uridecodebin name=source ! queue ! nvvideoconvert !",
            "video/x-raw(memory:NVMM),format=NV12 ! mux.sink_0",
            f"nvstreammux name=mux batch-size=1 width={mux_width} height={mux_height} ! tee name=raw",
            "raw. ! queue ! nvvideoconvert ! video/x-raw,format=I420 !",
            "jpegenc ! multifilesink name=source_frame_sink max-files=1",
            "raw. ! queue ! nvinfer name=infer ! nvvideoconvert !",
            "video/x-raw(memory:NVMM),format=RGBA ! nvdsosd name=osd process-mode=0 display-text=1 ! tee name=rendered",
            "rendered. ! queue ! nvvideoconvert ! video/x-raw,format=I420 !",
            "jpegenc ! multifilesink name=rendered_frame_sink max-files=1",
            "rendered. ! queue !",
            video_chain,
        ]
    )


def _choose_encoder(Gst: Any, requested: str = "auto") -> tuple[str, str]:
    if requested in {"auto", "h264"} and Gst.ElementFactory.find("nvv4l2h264enc"):
        return "h264", "rendered-video.mp4"
    if requested in {"auto", "theora"} and Gst.ElementFactory.find("theoraenc") and Gst.ElementFactory.find("oggmux"):
        return "theora", "rendered-video.ogv"
    raise RunnerError(
        f"requested encoder {requested} is unavailable; use auto, h264, or theora"
    )


def _iter_meta(node: Any, cast: Any):
    while node is not None:
        try:
            yield cast(node.data)
            node = node.next
        except StopIteration:
            break


def _layer_array(pyds: Any, tensor_meta: Any, index: int) -> tuple[str, Any]:
    import numpy as np

    layer = pyds.get_nvds_LayerInfo(tensor_meta, index)
    dimensions = tuple(
        int(layer.inferDims.d[position])
        for position in range(int(layer.inferDims.numDims))
    )
    if not dimensions or any(value <= 0 for value in dimensions):
        raise RunnerError(f"tensor layer has invalid dimensions: {dimensions}")
    data_type = int(layer.dataType)
    dtype_map = {
        0: (ctypes.c_float, np.float32),
        1: (ctypes.c_uint16, np.float16),
        2: (ctypes.c_int8, np.int8),
        3: (ctypes.c_int32, np.int32),
        # NvDsInferDataType is not TensorRT's DataType enum: value 4 is INT64.
        4: (ctypes.c_int64, np.int64),
        5: (ctypes.c_uint8, np.uint8),
    }
    if data_type not in dtype_map:
        raise RunnerError(f"unsupported NvDsInfer tensor data type: {data_type}")
    ctype, dtype = dtype_map[data_type]
    try:
        pointer_value = pyds.get_ptr(tensor_meta.out_buf_ptrs_host[index])
    except (AttributeError, IndexError, TypeError):
        pointer_value = pyds.get_ptr(layer.buffer)
    pointer = ctypes.cast(pointer_value, ctypes.POINTER(ctype))
    array = np.ctypeslib.as_array(pointer, shape=(int(layer.inferDims.numElements),))
    if dtype == np.float16:
        array = array.view(np.float16)
    else:
        array = array.astype(dtype, copy=False)
    name = str(layer.layerName or f"output_{index}")
    return name, array.reshape(dimensions).copy()


def _tensor_outputs(pyds: Any, frame_meta: Any) -> tuple[dict[str, Any], list[str]]:
    outputs: dict[str, Any] = {}
    tensor_names: list[str] = []

    def consume(user_list: Any) -> None:
        for user_meta in _iter_meta(user_list, pyds.NvDsUserMeta.cast):
            if (
                user_meta.base_meta.meta_type
                != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
            ):
                continue
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            for index in range(int(tensor_meta.num_output_layers)):
                name, array = _layer_array(pyds, tensor_meta, index)
                outputs[name] = array
                tensor_names.append(name)

    consume(frame_meta.frame_user_meta_list)
    for object_meta in _iter_meta(frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast):
        consume(object_meta.obj_user_meta_list)
    return outputs, tensor_names


def _add_classification_overlay(
    pyds: Any,
    batch_meta: Any,
    frame_meta: Any,
    decoded: Mapping[str, Any],
) -> None:
    predictions = decoded.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise RunnerError("classification adapter returned no predictions")
    labels = []
    for prediction in predictions[:5]:
        if not isinstance(prediction, Mapping):
            continue
        label = prediction.get("label", prediction.get("class_id", "?"))
        score = float(prediction.get("score", 0.0))
        labels.append(f"{label} {score:.3f}")
    if not labels:
        raise RunnerError("classification predictions are not renderable")

    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    display_meta.num_rects = 1
    display_meta.num_labels = 1
    panel = display_meta.rect_params[0]
    panel.left = 8
    panel.top = 8
    panel.width = 640
    panel.height = 48
    panel.border_width = 2
    panel.has_bg_color = 1
    panel.bg_color.set(0.0, 0.15, 0.55, 0.85)
    panel.border_color.set(1.0, 1.0, 1.0, 1.0)
    text = display_meta.text_params[0]
    text.display_text = " | ".join(labels)
    text.x_offset = 16
    text.y_offset = 20
    text.font_params.font_name = "DejaVu Sans"
    text.font_params.font_size = 18
    text.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    text.set_bg_clr = 1
    text.text_bg_clr.set(0.0, 0.0, 0.0, 0.75)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def _common(
    *,
    campaign_id: str,
    task: str,
    source_hash: str,
    input_hash: str,
    producer: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "producer": producer,
        "task": task,
        "source_hash": source_hash,
        "input_fixture_sha256": input_hash,
    }


def _parse_pipeline(Gst: Any, GLib: Any, description: str) -> Any:
    """Convert GStreamer construction failures into the runner's BLOCKED contract."""

    try:
        return Gst.parse_launch(description)
    except GLib.GError as exc:
        raise RunnerError(f"DeepStream pipeline construction failed: {exc}") from exc


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GLib", "2.0")
        import pyds
        from gi.repository import GLib, Gst

        if importlib.util.find_spec("numpy") is None:
            raise ImportError("numpy is unavailable")
    except ImportError as exc:
        raise RunnerError(f"DeepStream Python dependencies are unavailable: {exc}") from exc

    manifest_path = args.manifest.resolve()
    manifest = _load_json(manifest_path)
    task = str(manifest.get("task", ""))
    if task != "classification":
        raise RunnerError(
            "the shipped reference runner currently proves classification only; "
            f"{task or 'undeclared'} remains ADAPTER_REQUIRED"
        )
    if str(manifest.get("capability_status", "")) not in {
        "ADAPTER_REQUIRED",
        "READY",
    }:
        raise RunnerError("manifest does not describe an executable classification route")
    adapter_contract = manifest.get("adapter_contract")
    if not isinstance(adapter_contract, Mapping):
        raise RunnerError("manifest adapter_contract must be an object")
    if adapter_contract.get("preprocess_strategy") != "nvinfer-native":
        raise RunnerError(
            "classification reference runner requires nvinfer-native preprocessing; "
            "adapter-input-tensor contracts remain ADAPTER_REQUIRED"
        )

    input_path = args.input.resolve()
    config_path = args.nvinfer_config.resolve()
    for path, label in ((input_path, "input media"), (config_path, "nvinfer config")):
        if not path.is_file() or path.stat().st_size == 0:
            raise RunnerError(f"{label} is missing or empty: {path}")
    config_text = config_path.read_text(encoding="utf-8")
    if "output-tensor-meta=1" not in config_text:
        raise RunnerError(
            "nvinfer config must enable output-tensor-meta=1 for adapter decoding"
        )
    if "input-tensor-from-meta=1" in config_text:
        raise RunnerError(
            "reference runner does not synthesize input tensor metadata; "
            "use nvinfer-native preprocessing"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "source-frame.jpg",
        "rendered-frame.jpg",
        "structured-output.json",
        "overlay-evidence.json",
        "deepstream-runtime.json",
        "semantic-validation.json",
        "visual-validation.json",
    )
    occupied = [name for name in artifact_names if (output_dir / name).exists()]
    if occupied:
        raise RunnerError(
            "output directory contains prior evidence; choose a fresh directory: "
            + ", ".join(occupied)
        )

    adapter = _load_adapter(manifest_path, manifest)
    validate_hook = _require_validate_hook(adapter) if args.oracle is not None else None
    source_hash = str(manifest.get("source_hash", "")).strip().lower()
    if len(source_hash) != 64:
        source_path = Path(str(manifest.get("source", "")))
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        if not source_path.is_file():
            raise RunnerError(
                "model source is unavailable for runtime evidence hashing"
            )
        source_hash = _digest(source_path)
    input_hash = _digest(input_path)
    runner_hash = _digest(Path(__file__).resolve())

    Gst.init(None)
    encoder, video_name = _choose_encoder(Gst, args.encoder)
    video_path = output_dir / video_name
    source_frame_path = output_dir / "source-frame.jpg"
    rendered_frame_path = output_dir / "rendered-frame.jpg"
    pipeline = _parse_pipeline(
        Gst, GLib,
        _pipeline_description(encoder, mux_width=args.mux_width, mux_height=args.mux_height),
    )
    pipeline.get_by_name("source").set_property("uri", input_path.as_uri())
    pipeline.get_by_name("infer").set_property("config-file-path", str(config_path))
    pipeline.get_by_name("source_frame_sink").set_property(
        "location", str(source_frame_path)
    )
    pipeline.get_by_name("rendered_frame_sink").set_property(
        "location", str(rendered_frame_path)
    )
    pipeline.get_by_name("video_sink").set_property("location", str(video_path))

    state: dict[str, Any] = {
        "buffer_count": 0,
        "metadata_count": 0,
        "tensor_names": set(),
        "last_decoded": None,
        "last_structured": None,
        "last_frame_id": None,
        "errors": [],
    }

    def probe(_pad: Any, info: Any, _data: Any) -> Any:
        buffer = info.get_buffer()
        if buffer is None:
            state["errors"].append("nvinfer probe received no GstBuffer")
            return Gst.PadProbeReturn.OK
        state["buffer_count"] += 1
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            state["errors"].append("nvinfer output has no NvDsBatchMeta")
            return Gst.PadProbeReturn.OK
        for frame_meta in _iter_meta(batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast):
            try:
                outputs, names = _tensor_outputs(pyds, frame_meta)
                if not outputs:
                    continue
                context = {
                    "source_id": int(frame_meta.source_id),
                    "frame_id": int(frame_meta.frame_num),
                    "roi_id": "full-frame",
                    "timestamp": int(frame_meta.buf_pts),
                }
                decoded = adapter.decode(outputs, context)
                structured = adapter.serialize(decoded, context)
                structured.update(
                    _common(
                        campaign_id=args.campaign_id,
                        task=task,
                        source_hash=source_hash,
                        input_hash=input_hash,
                        producer=PRODUCER,
                    )
                )
                _add_classification_overlay(pyds, batch_meta, frame_meta, decoded)
                state["metadata_count"] += 1
                state["tensor_names"].update(names)
                state["last_decoded"] = decoded
                state["last_structured"] = structured
                state["last_frame_id"] = context["frame_id"]
            except Exception as exc:  # DeepStream callbacks must preserve the bus.
                state["errors"].append(str(exc))
        return Gst.PadProbeReturn.OK

    infer_src = pipeline.get_by_name("infer").get_static_pad("src")
    if infer_src is None:
        raise RunnerError("nvinfer has no static src pad")
    infer_src.add_probe(Gst.PadProbeType.BUFFER, probe, None)

    bus = pipeline.get_bus()
    started_at = time.time()
    change = pipeline.set_state(Gst.State.PLAYING)
    startup = change != Gst.StateChangeReturn.FAILURE
    eos = False
    bus_error = ""
    try:
        deadline_ns = int(args.timeout * Gst.SECOND)
        message = bus.timed_pop_filtered(
            deadline_ns,
            Gst.MessageType.ERROR | Gst.MessageType.EOS,
        )
        if message is None:
            bus_error = f"pipeline timed out after {args.timeout} seconds"
        elif message.type == Gst.MessageType.EOS:
            eos = True
        else:
            error, debug = message.parse_error()
            bus_error = f"{error}: {debug or ''}".strip()
    finally:
        pipeline.set_state(Gst.State.NULL)
    ended_at = time.time()

    if bus_error:
        state["errors"].append(bus_error)
    if not startup:
        state["errors"].append("pipeline did not enter PLAYING")
    if not eos:
        state["errors"].append("pipeline did not reach EOS")
    if state["buffer_count"] <= 0:
        state["errors"].append("pipeline produced zero nvinfer buffers")
    if state["metadata_count"] <= 0 or not state["tensor_names"]:
        state["errors"].append("pipeline produced no decodable tensor metadata")
    if state["last_structured"] is None:
        state["errors"].append("pipeline produced no structured classification output")
    for path in (source_frame_path, rendered_frame_path, video_path):
        if not path.is_file() or path.stat().st_size == 0:
            state["errors"].append(f"pipeline did not create {path.name}")
    if state["errors"]:
        raise RunnerError("; ".join(dict.fromkeys(state["errors"])))

    finite_scores = _finite_prediction_scores(state["last_decoded"]["predictions"])
    if not finite_scores:
        raise RunnerError("classification predictions contain non-finite scores")
    structured_path = output_dir / "structured-output.json"
    structured_path.write_text(
        json.dumps(state["last_structured"], indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    structured_hash = _digest(structured_path)
    rendered_hash = _digest(rendered_frame_path)

    overlay = {
        **_common(
            campaign_id=args.campaign_id,
            task=task,
            source_hash=source_hash,
            input_hash=input_hash,
            producer="adapter-renderer",
        ),
        "frame_id": state["last_frame_id"],
        "roi_id": "full-frame",
        "renderer": "NvDsDisplayMeta classification label panel",
        "overlay_fields": ["class_id", "label", "score"],
        "structured_output_sha256": structured_hash,
        "source_frame_sha256": _digest(source_frame_path),
        "rendered_frame_sha256": rendered_hash,
    }
    overlay_path = output_dir / "overlay-evidence.json"
    overlay_path.write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runtime_passed = finite_scores and not state["errors"]
    runtime = {
        **_common(
            campaign_id=args.campaign_id,
            task=task,
            source_hash=source_hash,
            input_hash=input_hash,
            producer=PRODUCER,
        ),
        "producer_sha256": runner_hash,
        "status": "PASS" if runtime_passed else "FAIL",
        "startup": startup,
        "buffer_count": state["buffer_count"],
        "eos": eos,
        "exit_code": 0 if not state["errors"] else EXIT_FAIL,
        "elapsed_seconds": round(ended_at - started_at, 6),
        "task_metadata": {
            "count": state["metadata_count"],
            "tensor_names": sorted(state["tensor_names"]),
        },
        "numeric_assertion": {
            "name": "finite classification predictions",
            "passed": runtime_passed,
            "finite": finite_scores,
        },
        "structured_output_sha256": structured_hash,
        "rendered_output_sha256": rendered_hash,
        "video_sha256": _digest(video_path),
        "encoder": encoder,
    }
    runtime_path = output_dir / "deepstream-runtime.json"
    runtime_path.write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    semantic_path = output_dir / "semantic-validation.json"
    if args.oracle is not None:
        oracle_path = args.oracle.resolve()
        oracle = _load_json(oracle_path)
        if validate_hook is None:
            raise RunnerError("semantic validation hook was not initialized")
        semantic = validate_hook(state["last_decoded"], dict(oracle))
        semantic.update(
            _common(
                campaign_id=args.campaign_id,
                task=task,
                source_hash=source_hash,
                input_hash=input_hash,
                producer="adapter-semantic-validator",
            )
        )
        semantic["oracle_sha256"] = _digest(oracle_path)
        semantic["structured_output_sha256"] = structured_hash
        semantic_path.write_text(
            json.dumps(semantic, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    visual_dir = Path(__file__).resolve().parents[1] / "visual"
    sys.path.insert(0, str(visual_dir))
    try:
        from validate_rendered_output import validate_rendered_output
    finally:
        sys.path.pop(0)
    visual = validate_rendered_output(
        manifest_path,
        frame=rendered_frame_path,
        structured_output=structured_path,
        video=video_path,
        source_frame=source_frame_path,
        overlay_evidence=overlay_path,
        deepstream_status="PASS",
    )
    visual.update(
        _common(
            campaign_id=args.campaign_id,
            task=task,
            source_hash=source_hash,
            input_hash=input_hash,
            producer="validate_rendered_output.py",
        )
    )
    visual["structured_output_sha256"] = structured_hash
    visual["rendered_output_sha256"] = rendered_hash
    visual_path = output_dir / "visual-validation.json"
    visual_path.write_text(
        json.dumps(visual, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if visual.get("pipeline_visual_result") != "PASS":
        raise RunnerError("rendered-output validation did not pass")

    return {
        "status": "PASS",
        "task": task,
        "campaign_id": args.campaign_id,
        "encoder": encoder,
        "evidence": {
            "input_fixture": str(input_path),
            "deepstream_runtime": str(runtime_path),
            "structured_output": str(structured_path),
            "semantic_validation": str(semantic_path) if semantic_path.exists() else "",
            "overlay_evidence": str(overlay_path),
            "rendered_output": str(rendered_frame_path),
            "visual_validation": str(visual_path),
            "video": str(video_path),
            "source_frame": str(source_frame_path),
        },
        "next": (
            "Add backend_parity evidence with the same campaign_id/source/input bindings; "
            "semantic evidence is also required before readiness promotion."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--nvinfer-config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--mux-width", type=int, default=1280, help="decoded frame width presented to nvinfer")
    parser.add_argument("--mux-height", type=int, default=720, help="decoded frame height presented to nvinfer")
    parser.add_argument("--encoder", choices=("auto", "h264", "theora"), default="auto",
                        help="select one output encoder; use theora for a bounded hardware-encoder fallback")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.describe:
        print(json.dumps(pipeline_contract(), indent=2, sort_keys=True))
        return 0
    required_arguments = (
        ("manifest", args.manifest),
        ("nvinfer_config", args.nvinfer_config),
        ("input", args.input),
        ("output_dir", args.output_dir),
        ("campaign_id", args.campaign_id),
    )
    missing = [name for name, value in required_arguments if value in (None, "")]
    if missing:
        print(
            json.dumps(
                {"status": "BLOCKED", "errors": ["missing arguments: " + ", ".join(missing)]},
                indent=2,
            ),
            file=sys.stderr,
        )
        return EXIT_INVALID
    if args.timeout <= 0 or args.timeout > 300:
        print(
            json.dumps(
                {"status": "BLOCKED", "errors": ["timeout must be between 1 and 300 seconds"]},
                indent=2,
            ),
            file=sys.stderr,
        )
        return EXIT_INVALID
    if not (16 <= args.mux_width <= 4096 and 16 <= args.mux_height <= 4096):
        print(
            json.dumps({"status": "BLOCKED", "errors": ["mux width and height must be between 16 and 4096 pixels"]}, indent=2),
            file=sys.stderr,
        )
        return EXIT_INVALID
    try:
        result = run(args)
    except (ImportError, OSError, ValueError, RunnerError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"status": "BLOCKED", "errors": [str(exc)]}, indent=2),
            file=sys.stderr,
        )
        return EXIT_BLOCKED
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
