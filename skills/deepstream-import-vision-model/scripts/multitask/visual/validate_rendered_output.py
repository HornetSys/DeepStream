#!/usr/bin/env python3
"""Gate task output rendered on a decoded DeepStream frame.

When the caller supplies the DeepStream gate status, also emit the orthogonal
pipeline-plus-visual verdict used by Devtest. This verdict is PASS only when
the pipeline completed successfully and this validator accepted model-aware
rendered output. It does not rewrite backend-parity or semantic-oracle gates.
"""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageChops

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(SCRIPT_DIR))
from model_pipeline import ManifestError, manifest_from_dict, pipeline_graph_errors  # noqa: E402

SEMANTIC_KEYS = {
    "classification": ("class_id", "label", "score", "top_k", "logits"),
    "segmentation": ("mask", "class_map", "segmentation", "instances", "confidence"),
    "pose": ("keypoints", "skeleton", "joints", "confidence"),
    "depth": ("depth", "depth_map", "inverse_depth"),
    "embedding": ("embedding", "vector", "features"),
    "optical-flow": ("flow", "optical_flow", "motion"),
    "multimodal": ("outputs", "results"),
    "generic": ("outputs", "results"),
    "detection": ("detections", "objects", "boxes"),
}
DIAGNOSTIC_RENDERER_TOKENS = ("none", "diagnostic", "raw tensor", "tensor-only", "tensor bars only")


def _normalise_gate_status(value: str | None) -> str:
    status = str(value or "UNTESTED").strip().upper().replace("-", "_")
    if status.startswith("PASS") or status == "RENDERED_PASS":
        return "PASS"
    if status.startswith("FAIL") or status == "RUNTIME_ONLY":
        return "FAIL"
    if status.startswith("BLOCKED"):
        return "BLOCKED"
    if status.startswith("ADAPTER_REQUIRED"):
        return "ADAPTER_REQUIRED"
    return "UNTESTED"


def pipeline_visual_verdict(deepstream_status: str | None, visual_status: str | None) -> str:
    """Return the Devtest runtime-visual verdict without changing other gates."""

    deepstream = _normalise_gate_status(deepstream_status)
    visual = _normalise_gate_status(visual_status)
    if deepstream == "PASS" and visual == "PASS":
        return "PASS"
    if "FAIL" in {deepstream, visual}:
        return "FAIL"
    if "BLOCKED" in {deepstream, visual}:
        return "BLOCKED"
    if "ADAPTER_REQUIRED" in {deepstream, visual}:
        return "ADAPTER_REQUIRED"
    return "UNTESTED"


def _attach_pipeline_visual_verdict(
    report: dict[str, Any], deepstream_status: str | None
) -> dict[str, Any]:
    visual_status = report.get("visual_status", report.get("status", "UNTESTED"))
    report["deepstream_status"] = _normalise_gate_status(deepstream_status)
    report["pipeline_visual_result"] = pipeline_visual_verdict(
        deepstream_status, str(visual_status)
    )
    report["pipeline_visual_rule"] = (
        "PASS only when deepstream_status=PASS and visual_status=PASS; "
        "backend and semantic results remain orthogonal"
    )
    return report


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(child) for child in value.values())
    if isinstance(value, list):
        return all(_all_finite(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _semantic_values(value: Any, keys: tuple[str, ...]) -> list[Any]:
    matches: list[Any] = []
    if isinstance(value, Mapping):
        for key in keys:
            candidate = value.get(key)
            if candidate not in (None, "", [], {}):
                matches.append(candidate)
        for child in value.values():
            matches.extend(_semantic_values(child, keys))
    if isinstance(value, list):
        for child in value:
            matches.extend(_semantic_values(child, keys))
    return matches


def _contains_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_number(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_number(child) for child in value)
    return False


def _image_check(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "FAIL", "detail": f"missing decoded frame: {path}"}
    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            extrema = rgb.getextrema()
            width, height = rgb.size
    except (OSError, ValueError) as exc:
        return {"status": "FAIL", "detail": f"invalid decoded frame: {exc}"}
    blank = all(low == high == 0 for low, high in extrema)
    return {
        "status": "FAIL" if blank or width < 2 or height < 2 else "PASS",
        "detail": "decoded frame is blank" if blank else f"decoded frame {width}x{height}",
        "path": str(path),
        "width": width,
        "height": height,
        "blank": blank,
    }


def _difference_check(source: Path | None, rendered: Path) -> dict[str, Any]:
    if source is None:
        return {"status": "UNTESTED", "detail": "no source frame supplied for overlay-difference check"}
    if not source.exists():
        return {"status": "FAIL", "detail": f"missing source frame: {source}"}
    try:
        with Image.open(source) as source_image, Image.open(rendered) as rendered_image:
            left = source_image.convert("RGB").resize((640, 360))
            right = rendered_image.convert("RGB").resize((640, 360))
            difference = ImageChops.difference(left, right)
            channels = difference.split()
            thresholded = [channel.point(lambda value: 255 if value > 12 else 0) for channel in channels]
            changed_mask = ImageChops.lighter(ImageChops.lighter(thresholded[0], thresholded[1]), thresholded[2])
            changed_pixels = changed_mask.histogram()[255]
            pixel_count = left.width * left.height
            changed_ratio = changed_pixels / pixel_count
            mean_delta = sum(value * count for value, count in enumerate(difference.histogram())) / (pixel_count * 3)
            changed = changed_ratio >= 0.0005 and mean_delta >= 0.05
    except (OSError, ValueError) as exc:
        return {"status": "FAIL", "detail": f"could not compare source/rendered frames: {exc}"}
    return {
        "status": "PASS" if changed else "FAIL",
        "detail": "rendered frame has a measurable overlay difference" if changed else "no measurable overlay difference",
        "changed_pixel_ratio": round(changed_ratio, 6),
        "mean_absolute_delta": round(mean_delta, 6),
    }


def _structured_check(path: Path, task: str) -> dict[str, Any]:
    if not path.exists():
        return {"status": "FAIL", "detail": f"missing structured output: {path}"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "detail": f"invalid structured output JSON: {exc}"}
    keys = SEMANTIC_KEYS.get(task, SEMANTIC_KEYS["generic"])
    semantic_values = _semantic_values(value, keys)
    semantic = bool(semantic_values)
    numeric = any(_contains_number(candidate) for candidate in semantic_values)
    finite = _all_finite(value)
    passed = semantic and numeric and finite
    detail = (
        f"structured {task} output contains a finite semantic field"
        if passed
        else (
            "structured output contains non-finite values"
            if semantic and not finite
            else (
                "semantic output contains no finite numeric values"
                if semantic and not numeric
                else f"structured output has no non-empty {keys} field"
            )
        )
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
        "path": str(path),
        "task": task,
        "semantic_keys": list(keys),
    }


def _all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def _contains_identity(value: Any, names: set[str], expected: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in names and str(child) == str(expected):
                return True
            if _contains_identity(child, names, expected):
                return True
    elif isinstance(value, list):
        return any(_contains_identity(child, names, expected) for child in value)
    return False


def _overlay_provenance_check(
    path: Path | None,
    structured_path: Path | None,
    rendered_path: Path | None,
    task: str,
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "FAIL",
            "detail": "--overlay-evidence is required to bind rendered pixels to structured output",
        }
    if structured_path is None or not structured_path.exists():
        return {"status": "FAIL", "detail": "structured output is unavailable for overlay binding"}
    if rendered_path is None or not rendered_path.exists():
        return {"status": "FAIL", "detail": "rendered frame is unavailable for overlay binding"}
    if not path.exists():
        return {"status": "FAIL", "detail": f"missing overlay evidence: {path}"}
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
        structured_bytes = structured_path.read_bytes()
        rendered_bytes = rendered_path.read_bytes()
        structured = json.loads(structured_bytes)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "detail": f"invalid overlay evidence: {exc}"}
    if not isinstance(evidence, Mapping):
        return {"status": "FAIL", "detail": "overlay evidence must be a JSON object"}
    for field in (
        "frame_id",
        "structured_output_sha256",
        "rendered_frame_sha256",
        "renderer",
        "overlay_fields",
    ):
        if evidence.get(field) in (None, "", [], {}):
            return {"status": "FAIL", "detail": f"overlay evidence is missing {field}"}
    structured_digest = hashlib.sha256(structured_bytes).hexdigest()
    rendered_digest = hashlib.sha256(rendered_bytes).hexdigest()
    if str(evidence["structured_output_sha256"]).lower() != structured_digest:
        return {"status": "FAIL", "detail": "overlay evidence hash does not match structured output"}
    if str(evidence["rendered_frame_sha256"]).lower() != rendered_digest:
        return {"status": "FAIL", "detail": "overlay evidence hash does not match rendered frame"}
    fields = evidence["overlay_fields"]
    if not isinstance(fields, list) or not all(isinstance(field, str) and field for field in fields):
        return {"status": "FAIL", "detail": "overlay_fields must be a non-empty string list"}
    structured_keys = _all_keys(structured)
    missing_fields = sorted(set(fields) - structured_keys)
    if missing_fields:
        return {
            "status": "FAIL",
            "detail": "overlay fields are absent from structured output: " + ", ".join(missing_fields),
        }
    semantic_fields = set(SEMANTIC_KEYS.get(task, SEMANTIC_KEYS["generic"]))
    if not semantic_fields.intersection(fields):
        return {
            "status": "FAIL",
            "detail": "overlay_fields contain no task-semantic field",
        }
    if not _contains_identity(structured, {"frame_id", "source_frame_id"}, evidence["frame_id"]):
        return {"status": "FAIL", "detail": "frame_id is not present in structured output"}
    if "roi_id" in evidence and not _contains_identity(
        structured, {"roi_id", "object_id"}, evidence["roi_id"]
    ):
        return {"status": "FAIL", "detail": "roi_id is not present in structured output"}
    return {
        "status": "PASS",
        "detail": "renderer evidence binds the changed frame/ROI overlay to the structured task record",
        "path": str(path),
        "frame_id": evidence["frame_id"],
        "roi_id": evidence.get("roi_id"),
        "renderer": evidence["renderer"],
        "overlay_fields": fields,
        "structured_output_sha256": structured_digest,
        "rendered_frame_sha256": rendered_digest,
    }


def _decode_video_frame(Gst: Any, path: Path) -> tuple[bool, str]:
    """Decode one video buffer without requiring an unrelated audio decoder."""

    pipeline = Gst.Pipeline.new("video-evidence-check")
    source = Gst.ElementFactory.make("uridecodebin", "video-source")
    sink = Gst.ElementFactory.make("fakesink", "video-sink")
    if not pipeline or not source or not sink:
        return False, "GStreamer video decoder or sink is unavailable"
    source.set_property("uri", path.resolve().as_uri())
    sink.set_property("sync", False)
    sink.set_property("signal-handoffs", True)
    decoded = False
    linked = False

    def on_pad_added(_source: Any, pad: Any) -> None:
        nonlocal linked
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            return
        media_type = caps.get_structure(0).get_name()
        if not media_type.startswith("video/x-raw"):
            return
        linked = pad.link(sink.get_static_pad("sink")) == Gst.PadLinkReturn.OK

    def on_handoff(*_args: Any) -> None:
        nonlocal decoded
        decoded = True

    source.connect("pad-added", on_pad_added)
    sink.connect("handoff", on_handoff)
    pipeline.add(source)
    pipeline.add(sink)
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            return False, "GStreamer could not start video decoding"
        bus = pipeline.get_bus()
        deadline = time.monotonic() + 30
        while not decoded and time.monotonic() < deadline:
            message = bus.timed_pop_filtered(
                Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS
            )
            if message and message.type == Gst.MessageType.ERROR:
                error, _debug = message.parse_error()
                return False, f"GStreamer video decode failed: {error}"
            if message and message.type == Gst.MessageType.EOS:
                break
        if decoded and linked:
            return True, "decoded video frame observed"
        return False, "no decoded video frame observed"
    finally:
        pipeline.set_state(Gst.State.NULL)


def _gstreamer_video_check(path: Path) -> dict[str, Any]:
    """Inspect and decode video without gating on unrelated audio streams."""

    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstPbutils", "1.0")
        from gi.repository import Gst, GstPbutils

        Gst.init(None)
        discoverer = GstPbutils.Discoverer.new(30 * Gst.SECOND)
        try:
            info = discoverer.discover_uri(path.resolve().as_uri())
        except Exception as exc:
            # A missing audio plugin can abort container discovery even when
            # the video stream itself is decodable. Require a real frame.
            decoded, detail = _decode_video_frame(Gst, path)
            return {
                "status": "PASS" if decoded else "UNTESTED",
                "detail": detail if decoded else f"video discovery unavailable ({exc}); {detail}",
                "path": str(path),
                "inspector": "gstreamer-video-decode",
            }
        streams = info.get_video_streams()
        if not streams:
            return {"status": "FAIL", "detail": "GStreamer found no usable video stream"}
        stream = streams[0]
        width, height = int(stream.get_width()), int(stream.get_height())
        if width < 2 or height < 2:
            return {"status": "FAIL", "detail": "video has invalid dimensions"}
        decoded, detail = _decode_video_frame(Gst, path)
        return {
            "status": "PASS" if decoded else "FAIL",
            "detail": detail,
            "path": str(path),
            "inspector": "gstreamer-discoverer",
            "stream": {"width": width, "height": height},
            "container_discovery": str(info.get_result().value_nick),
        }
    except Exception as exc:
        return {"status": "UNTESTED", "detail": f"GStreamer discovery unavailable: {exc}"}


def _video_check(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "UNTESTED", "detail": "no video evidence supplied"}
    if not path.exists():
        return {"status": "FAIL", "detail": f"missing video evidence: {path}"}
    gstreamer_result = _gstreamer_video_check(path)
    if gstreamer_result["status"] != "UNTESTED":
        return gstreamer_result
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return gstreamer_result
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "FAIL", "detail": f"ffprobe failed: {exc}"}
    if result.returncode != 0:
        return {"status": "FAIL", "detail": result.stderr[-1000:] or "ffprobe returned non-zero"}
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except (ValueError, TypeError):
        streams = []
    passed = bool(streams and int(streams[0].get("width", 0)) > 1 and int(streams[0].get("height", 0)) > 1)
    return {"status": "PASS" if passed else "FAIL", "detail": "video contains a decoded video stream" if passed else "video has no usable video stream", "path": str(path), "stream": streams[0] if streams else None}


def validate_rendered_output(
    manifest_path: str | Path,
    *,
    frame: str | Path | None,
    structured_output: str | Path | None,
    video: str | Path | None = None,
    display_screenshot: str | Path | None = None,
    source_frame: str | Path | None = None,
    overlay_evidence: str | Path | None = None,
    require_display: bool = False,
    deepstream_status: str | None = None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        manifest = manifest_from_dict(
            json.loads(Path(manifest_path).read_text(encoding="utf-8")),
            manifest_path=manifest_path,
        )
        graph_errors = pipeline_graph_errors(
            manifest.pipeline_graph,
            required_upstream_stages=manifest.upstream_stages,
        )
    except (OSError, ValueError, ManifestError) as exc:
        return _attach_pipeline_visual_verdict(
            {
                "status": "BLOCKED",
                "visual_status": "BLOCKED",
                "passed": False,
                "checks": {"manifest": {"status": "FAIL", "detail": str(exc)}},
            },
            deepstream_status,
        )
    if graph_errors:
        return _attach_pipeline_visual_verdict(
            {
                "status": "BLOCKED",
                "visual_status": "BLOCKED",
                "passed": False,
                "checks": {"pipeline_graph": {"status": "FAIL", "detail": "; ".join(graph_errors)}},
            },
            deepstream_status,
        )
    graph = manifest.pipeline_graph
    declared_status = str(graph.get("status", "PLANNED")).upper()
    renderer = str(graph.get("renderer", "")).lower()
    if declared_status == "BLOCKED":
        return _attach_pipeline_visual_verdict(
            {
                "status": "BLOCKED",
                "visual_status": "BLOCKED",
                "route_status": declared_status,
                "passed": False,
                "checks": {
                    "pipeline_graph": {
                        "status": "FAIL",
                        "detail": "graph is blocked and has no executable runtime path",
                    }
                },
            },
            deepstream_status,
        )
    if any(token in renderer for token in DIAGNOSTIC_RENDERER_TOKENS):
        return _attach_pipeline_visual_verdict(
            {
                "status": "RUNTIME_ONLY",
                "visual_status": "FAIL",
                "route_status": declared_status,
                "passed": False,
                "checks": {
                    "renderer": {
                        "status": "FAIL",
                        "detail": "renderer is diagnostic/raw-tensor only",
                    }
                },
            },
            deepstream_status,
        )

    frame_path = Path(frame) if frame else None
    structured_path = Path(structured_output) if structured_output else None
    checks["decoded_frame"] = _image_check(frame_path) if frame_path else {"status": "FAIL", "detail": "--frame is required"}
    checks["structured_output"] = _structured_check(structured_path, manifest.task) if structured_path else {"status": "FAIL", "detail": "--structured-output is required"}
    checks["overlay_difference"] = _difference_check(Path(source_frame) if source_frame else None, frame_path) if frame_path else {"status": "FAIL", "detail": "cannot compare without --frame"}
    checks["overlay_provenance"] = _overlay_provenance_check(
        Path(overlay_evidence) if overlay_evidence else None,
        structured_path,
        frame_path,
        manifest.task,
    )
    checks["video"] = _video_check(Path(video) if video else None)
    if require_display:
        checks["display"] = _image_check(Path(display_screenshot)) if display_screenshot else {"status": "FAIL", "detail": "--display-screenshot is required when --require-display is set"}
    elif display_screenshot:
        checks["display"] = _image_check(Path(display_screenshot))
    required = {
        "decoded_frame",
        "structured_output",
        "overlay_difference",
        "overlay_provenance",
        "video",
    }
    if require_display:
        required.add("display")
    passed = all(checks[name]["status"] == "PASS" for name in required)
    status = "RENDERED_PASS" if passed else "RUNTIME_ONLY"
    return _attach_pipeline_visual_verdict({
        "status": status,
        "visual_status": "PASS" if passed else "FAIL",
        "route_status": declared_status,
        "passed": passed,
        "task": manifest.task,
        "task_variant": manifest.task_variant or None,
        "pipeline_graph": graph,
        "checks": checks,
        "contract": {
            "semantic_render_required": True,
            "raw_tensor_visuals_are_insufficient": True,
            "overlay_difference_is_optional_without_source_frame": False,
            "video_evidence_required": True,
            "overlay_provenance_required": True,
        },
    }, deepstream_status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--frame", type=Path, required=True, help="decoded frame with task overlay")
    parser.add_argument("--structured-output", type=Path, required=True, help="JSON task output asserted against the model task")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--display-screenshot", type=Path)
    parser.add_argument("--source-frame", type=Path, required=True, help="undecorated source frame used to verify a visible task overlay")
    parser.add_argument(
        "--overlay-evidence",
        type=Path,
        required=True,
        help="renderer-written JSON binding frame/ROI identity and semantic fields to structured-output and rendered-frame SHA-256 values",
    )
    parser.add_argument("--require-display", action="store_true")
    parser.add_argument(
        "--deepstream-status",
        help="DeepStream gate status after EOS/buffer/metadata validation; enables pipeline_visual_result",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_rendered_output(
        args.manifest,
        frame=args.frame,
        structured_output=args.structured_output,
        video=args.video,
        display_screenshot=args.display_screenshot,
        source_frame=args.source_frame,
        overlay_evidence=args.overlay_evidence,
        require_display=args.require_display,
        deepstream_status=args.deepstream_status,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.deepstream_status is not None:
        return 0 if report["pipeline_visual_result"] == "PASS" else 1
    return 0 if report["status"] == "RENDERED_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
