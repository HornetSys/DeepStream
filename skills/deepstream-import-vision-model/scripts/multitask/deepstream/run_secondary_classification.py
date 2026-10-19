#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run a detector-owned ROI to secondary-classifier DeepStream application.

The two nvinfer configs own model-specific preprocessing and parsing. This
runner owns only the reusable graph, object/classifier metadata, and overlay.
It does not claim semantic accuracy without a separately reviewed oracle.
"""

from __future__ import annotations

import argparse
import configparser
import json
import re
import subprocess
import sys
from pathlib import Path


MAX_TIMEOUT = 300
SAFE_LABEL = re.compile(r"[\x20-\x7e]{1,96}\Z")


def _property(path: Path, key: str) -> str:
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    if not parser.read(path) or not parser.has_section("property"):
        raise ValueError(f"nvinfer config has no [property] section: {path}")
    return parser.get("property", key, fallback="").strip()


def _labels(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    labels = [item.strip() for item in re.split(r"[;\n]", raw) if item.strip()]
    if not labels or any(not SAFE_LABEL.fullmatch(label) for label in labels):
        raise ValueError(f"label map contains missing or unsafe labels: {path}")
    return labels


def _classifier_text_position(roi: dict[str, float]) -> tuple[int, int]:
    """Keep the classifier label inside its ROI, below the detector title."""
    return max(int(roi["left"]) + 2, 0), max(int(roi["top"]) + 2, 0)


def _validated_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    video, primary, secondary = (Path(value).resolve() for value in
                                 (args.video, args.primary_config, args.secondary_config))
    detector_labels, classifier_labels = (Path(value).resolve() for value in
                                           (args.detector_labels, args.classifier_labels))
    for path in (video, primary, secondary, detector_labels, classifier_labels):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required regular nonempty file is absent: {path}")
    primary_id = _property(primary, "gie-unique-id")
    secondary_id = _property(secondary, "gie-unique-id")
    if _property(primary, "process-mode") != "1" or _property(secondary, "process-mode") != "2":
        raise ValueError("configs must declare primary detector and secondary classifier process modes")
    if _property(primary, "network-type") != "0" or _property(secondary, "network-type") != "1":
        raise ValueError("configs must declare detector and classifier network types")
    if not primary_id.isdecimal() or not secondary_id.isdecimal() or primary_id == secondary_id:
        raise ValueError("primary and secondary GIE IDs must be distinct positive integers")
    if _property(secondary, "operate-on-gie-id") != primary_id:
        raise ValueError("secondary classifier must operate on the selected detector GIE")
    if not _property(secondary, "operate-on-class-ids"):
        raise ValueError("secondary classifier must declare detector class IDs for its ROIs")
    _labels(detector_labels)
    _labels(classifier_labels)
    return video, primary, secondary, detector_labels, classifier_labels


def _graph(video: Path, primary: Path, secondary: Path, output: Path) -> dict:
    nodes = [
        {"type": "uridecodebin", "name": "source", "properties": {"uri": video.as_uri()}},
        {"type": "nvstreammux", "name": "streammux", "properties":
         {"batch-size": 1, "width": 1280, "height": 720, "batched-push-timeout": 40000}},
        {"type": "nvinfer", "name": "pgie", "properties": {"config-file-path": str(primary)}},
        {"type": "queue", "name": "queue_sgie"},
        {"type": "nvinfer", "name": "sgie", "properties": {"config-file-path": str(secondary)}},
        {"type": "queue", "name": "queue_osd"},
        {"type": "nvvideoconvert", "name": "convert_for_osd"},
        {"type": "nvdsosd", "name": "osd"},
        {"type": "nvvideoconvert", "name": "convert_for_encoder"},
        {"type": "videoconvert", "name": "software_convert"},
        {"type": "capsfilter", "name": "software_caps", "properties":
         {"caps": "video/x-raw,format=I420"}},
        {"type": "theoraenc", "name": "encoder"},
        {"type": "oggmux", "name": "muxer"},
        {"type": "filesink", "name": "sink", "properties": {"location": str(output)}},
    ]
    names = [node["name"] for node in nodes]
    return {"deepstream": {"nodes": nodes,
                           "edges": dict(zip(names[:-1], names[1:]))}}


def _run_worker(args: argparse.Namespace) -> int:
    import yaml
    from pyservicemaker import BatchMetadataOperator, Pipeline, Probe, osd

    video, primary, secondary, detector_labels_path, classifier_labels_path = _validated_paths(args)
    detector_labels = _labels(detector_labels_path)
    classifier_labels = _labels(classifier_labels_path)
    secondary_id = int(_property(secondary, "gie-unique-id"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "pipeline.yaml"
    video_path = output_dir / "rendered-output.ogv"
    structured_path = output_dir / "structured-output.json"
    for path in (graph_path, video_path, structured_path):
        if path.exists():
            raise ValueError(f"choose a fresh output directory; file exists: {path}")
    with graph_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(_graph(video, primary, secondary, video_path),
                       stream, sort_keys=False)

    class Evidence(BatchMetadataOperator):
        def __init__(self) -> None:
            super().__init__()
            self.frames = 0
            self.detector_objects = 0
            self.records: list[dict] = []

        def handle_metadata(self, batch_meta) -> None:
            for frame in batch_meta.frame_items:
                self.frames += 1
                for obj in frame.object_items:
                    self.detector_objects += 1
                    class_id = int(obj.class_id)
                    detector_label = (detector_labels[class_id]
                                      if 0 <= class_id < len(detector_labels) else "unknown")
                    roi = {key: float(getattr(obj.rect_params, key))
                           for key in ("left", "top", "width", "height")}
                    for classifier in obj.classifier_items:
                        if int(classifier.unique_component_id) != secondary_id:
                            continue
                        for index in range(int(classifier.n_labels)):
                            label = str(classifier.get_n_label(index)).strip()
                            if not SAFE_LABEL.fullmatch(label):
                                continue
                            record = {
                                "source_id": int(frame.source_id),
                                "frame_number": int(frame.frame_number),
                                "detector_class_id": class_id,
                                "detector_label": detector_label,
                                "detector_confidence": float(obj.confidence),
                                "roi": roi,
                                "classifier_component_id": secondary_id,
                                "class_id": (classifier_labels.index(label)
                                             if label in classifier_labels else -1),
                                "label": label,
                            }
                            self.records.append(record)
                            text = osd.Text()
                            text.display_text = label.encode("ascii")
                            text.x_offset, text.y_offset = _classifier_text_position(roi)
                            text.font.name = osd.FontFamily.Serif
                            text.font.size = 14
                            text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
                            text.set_bg_color = True
                            text.bg_color = osd.Color(0.0, 0.0, 0.0, 0.85)
                            display = batch_meta.acquire_display_meta()
                            display.add_text(text)
                            frame.append(display)

    evidence = Evidence()
    failure = None
    try:
        Pipeline("secondary-classification", str(graph_path)).attach(
            "queue_osd", Probe("secondary_classification_evidence", evidence)
        ).start().wait()
    except Exception as exc:
        failure = str(exc)
    video_ok = video_path.is_file() and video_path.stat().st_size > 0
    payload = {
        "schema_version": 1, "task": "classification", "producer": "secondary-classification-runner",
        "status": "PASS" if failure is None and evidence.frames > 0 and evidence.records and video_ok else "FAIL",
        "frames": evidence.frames, "detector_object_count": evidence.detector_objects,
        "classifier_record_count": len(evidence.records), "records": evidence.records,
        "eos": failure is None, "pipeline_error": failure,
        "video": str(video_path), "graph": str(graph_path),
    }
    with structured_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"},
                     indent=2))
    return 0 if payload["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--primary-config", required=True)
    parser.add_argument("--secondary-config", required=True)
    parser.add_argument("--detector-labels", required=True)
    parser.add_argument("--classifier-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=MAX_TIMEOUT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout <= MAX_TIMEOUT:
        parser.error("timeout must be between 1 and 300 seconds")
    try:
        _validated_paths(args)
        if args.worker:
            return _run_worker(args)
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        result = subprocess.run(command, text=True, capture_output=True,
                                timeout=args.timeout, check=False)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        return result.returncode
    except (ImportError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
