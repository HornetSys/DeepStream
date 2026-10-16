#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Render a custom config_infer_primary.txt (or SGIE) from spec values.

This is a library module imported by render.py, but also runnable standalone.

Usage:

    python3 render_infer.py --network-type detector --batch-size 4 \\
        --onnx-file my.onnx --num-detected-classes 4 --out my_infer.txt

Arguments:
    Network type, batching/model options, and the required ``--out`` path.

Output:
    Writes one nvinfer configuration file and reports its path on stderr.

Exit codes:
    0 on success, 1 for a handled write failure, and 2 for invalid
    command-line arguments.
"""
from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent.parent / "assets" / "templates"


def _load_template(name: str) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    parser.read(TEMPLATES / name, encoding="utf-8")
    return {s: dict(parser.items(s)) for s in parser.sections()}


def build_infer(spec: dict) -> dict[str, dict[str, str]]:
    """spec is the `infer_config` block from the main spec, *plus* a few
    derived fields (gie_unique_id, batch_size) merged in by render.py.
    Returns a {group: {key: val}} ready to serialise.
    """
    nt = spec.get("network_type", "detector")
    template_name = (
        "infer_primary_classifier.txt" if nt == "classifier" else "infer_primary_detector.txt"
    )
    cfg = _load_template(template_name)
    p = cfg["property"]

    # Required-ish fields with sensible defaults
    p["gpu-id"] = str(spec.get("gpu_id", 0))
    p["batch-size"] = str(spec.get("batch_size", 1))
    p["gie-unique-id"] = str(spec.get("gie_unique_id", 1))
    p["network-mode"] = str(spec.get("network_mode", 2))  # 2=FP16
    if "net_scale_factor" in spec:
        p["net-scale-factor"] = str(spec["net_scale_factor"])
    if "model_color_format" in spec:
        p["model-color-format"] = str(spec["model_color_format"])

    # Model files
    if spec.get("onnx_file"):
        p["onnx-file"] = str(spec["onnx_file"])
    else:
        p.pop("onnx-file", None)
    if spec.get("model_engine_file"):
        p["model-engine-file"] = str(spec["model_engine_file"])
    else:
        p.pop("model-engine-file", None)
    if spec.get("labelfile_path"):
        p["labelfile-path"] = str(spec["labelfile_path"])
    else:
        p.pop("labelfile-path", None)
    if spec.get("int8_calib_file"):
        p["int8-calib-file"] = str(spec["int8_calib_file"])
    else:
        p.pop("int8-calib-file", None)

    # Detector-specific
    if nt == "detector":
        p["num-detected-classes"] = str(spec.get("num_detected_classes", 4))
        p["interval"] = str(spec.get("interval", 0))
        p["process-mode"] = str(spec.get("process_mode", 1))
        p["cluster-mode"] = str(spec.get("cluster_mode", 2))
        if "maintain_aspect_ratio" in spec:
            p["maintain-aspect-ratio"] = str(spec["maintain_aspect_ratio"])
        cls = cfg.setdefault("class-attrs-all", {})
        cls["pre-cluster-threshold"] = str(spec.get("pre_cluster_threshold", 0.2))
        cls["nms-iou-threshold"] = str(spec.get("nms_iou_threshold", 0.5))
        cls["topk"] = str(spec.get("topk", 20))
    else:  # classifier
        p["process-mode"] = str(spec.get("process_mode", 2))
        p["classifier-threshold"] = str(spec.get("classifier_threshold", 0.51))
        p["classifier-type"] = str(spec.get("classifier_type", "custom"))
        p["operate-on-gie-id"] = str(spec.get("operate_on_gie_id", 1))
        p["operate-on-class-ids"] = str(spec.get("operate_on_class_ids", "0"))
        if "input_object_min_width" in spec:
            p["input-object-min-width"] = str(spec["input_object_min_width"])
        if "input_object_min_height" in spec:
            p["input-object-min-height"] = str(spec["input_object_min_height"])

    # Drop any remaining `auto` sentinels.
    for grp in cfg.values():
        for k in list(grp.keys()):
            if grp[k] == "auto":
                grp.pop(k)
    return cfg


def to_txt(cfg: dict, header: str = "") -> str:
    lines = []
    if header:
        lines.append(header.rstrip())
        lines.append("")
    for group, kv in cfg.items():
        lines.append(f"[{group}]")
        for k, v in kv.items():
            lines.append(f"{k}={v}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


HEADER = (
    "################################################################################\n"
    "# Custom nvinfer config generated by deepstream-generate-config/\n"
    "# Referenced by the main deepstream-app config via [primary-gie].config-file.\n"
    "# Re-render via render.py instead of hand-editing.\n"
    "################################################################################"
)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Render a custom config_infer_primary.txt.")
    ap.add_argument("--network-type", choices=["detector", "classifier"], default="detector")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gie-unique-id", type=int, default=1)
    ap.add_argument("--network-mode", type=int, default=2)
    ap.add_argument("--num-detected-classes", type=int, default=4)
    ap.add_argument("--onnx-file", default="")
    ap.add_argument("--model-engine-file", default="")
    ap.add_argument("--labelfile-path", default="")
    ap.add_argument("--int8-calib-file", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    spec = {
        "network_type": args.network_type,
        "batch_size": args.batch_size,
        "gie_unique_id": args.gie_unique_id,
        "network_mode": args.network_mode,
        "num_detected_classes": args.num_detected_classes,
        "onnx_file": args.onnx_file,
        "model_engine_file": args.model_engine_file,
        "labelfile_path": args.labelfile_path,
        "int8_calib_file": args.int8_calib_file,
    }
    cfg = build_infer(spec)
    out_path = Path(args.out).expanduser()
    try:
        out_path.write_text(to_txt(cfg, HEADER), encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        sys.stderr.write(f"[render_infer] unable to write {out_path}: {exc}\n")
        return 1
    sys.stderr.write(f"[render_infer] wrote {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
