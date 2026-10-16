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

"""Render a deepstream-app / deepstream-test5 config file from a requirements spec.

Usage:
    render.py --spec <path-to-spec.yaml> [--out <output-path>]
    render.py --quickstart [--out <output-path>]

The spec format is documented in scripts/spec_schema.yaml.
The script applies a profile overlay (latency / perf / display / custom),
stitches fragment templates, resolves `auto` keys, and writes either a
`.txt` (GKeyFile / [section] style) or `.yml` config consumable by
`deepstream-app -c` or `deepstream-test5-app -c`.

Default output location (when omitted in spec):
    ./generated_config.<ext> in the current working directory

The stock deepstream-app samples directory under /opt/nvidia/deepstream/... is an
optional convention only. It is root-owned; do not write there from this script.
On PermissionError, pass --out to a writable path.

Arguments:
    Exactly one of ``--spec`` or ``--quickstart``; optional ``--out`` and
    ``--print-resolved-spec``.

Output:
    Writes the generated application config and any explicitly requested
    inference or tracker sidecar configs. Diagnostics go to standard error.

Exit codes:
    0 on success, 1 for a handled write failure, and 2 for invalid input or an
    unexpected command-line/specification error.
"""
from __future__ import annotations

import argparse
import configparser
import math
import shlex
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "render.py requires PyYAML. Install the pinned dependency with:\n"
        "    python3 -m pip install -r scripts/requirements.txt\n"
    )
    sys.exit(2)


SKILL_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = SKILL_ROOT / "assets" / "templates"

# Sibling helpers for opt-in custom infer / tracker generation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_infer
import render_tracker

GROUP_ORDER = [
    "application",
    "tiled-display",
    "source-list",
    "source-attr-all",
    # source0..N inserted dynamically here
    "streammux",
    "pre-process",
    "primary-gie",
    "tracker",
    # secondary-gie0..M inserted dynamically here
    "osd",
    # sink0..K inserted dynamically here
    "message-converter",
    # message-consumer0..L inserted dynamically here
    "nvds-analytics",
    "tests",
]


ALLOWED_PROFILES = frozenset({"latency", "perf", "display", "custom"})
ALLOWED_SOURCE_KINDS = frozenset({
    "file",
    "rtsp",
    "v4l2",
    "csi",
    "multi_uri",
    "source_list",
})
ALLOWED_SOURCE_KEYS = frozenset({
    "kind",
    "uris",
    "num_sources",
    "live",
    "use_nvmultiurisrcbin",
    "simulate_fps_interval_ms",
    "v4l2_dev_node",
    "camera_width",
    "camera_height",
    "camera_fps_n",
    "camera_fps_d",
    "csi_sensor_id",
})
CORE_SPEC_KEYS = frozenset({
    "deployment",
    "profile",
    "sources",
    "streammux",
    "primary_gie",
    "tracker",
    "secondary_gies",
    "osd",
    "tiled_display",
    "sinks",
    "tests",
})
MAPPING_SPEC_KEYS = frozenset({
    "deployment",
    "sources",
    "streammux",
    "primary_gie",
    "tracker",
    "osd",
    "tiled_display",
    "tests",
})
SEQUENCE_SPEC_KEYS = frozenset({"secondary_gies", "sinks"})
PGIE_PRESETS = frozenset({"trafficcamnet", "peoplenet", "custom"})
SGIE_PRESETS = frozenset({"vehicletypes", "vehiclemake", "custom"})


class SpecValidationError(ValueError):
    """Raised for an invalid user spec before rendering starts."""


def _spec_enabled(value) -> bool:
    """Interpret common YAML/CLI false values consistently."""
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def validate_profile(spec: dict) -> bool:
    profile = spec.get("profile", "perf")
    if profile in ALLOWED_PROFILES:
        return True
    sys.stderr.write(
        f"[render] invalid profile '{profile}'. "
        f"Use one of: {', '.join(sorted(ALLOWED_PROFILES))}.\n"
    )
    return False


# ---------------------------------------------------------------------------
# Preset registries
# ---------------------------------------------------------------------------
DS_ROOT_DEFAULT = "/opt/nvidia/deepstream/deepstream"
# Stock deepstream-app samples dir. Generated configs land here by default so
# they sit alongside the reference samples (source4_..., source30_..., etc.)
# and any sibling config_infer_primary / tracker .yml files generated for the
# same run end up co-located.
DEFAULT_OUTPUT_DIR = f"{DS_ROOT_DEFAULT}/samples/configs/deepstream-app"


def pgie_preset(name: str, ds_root: str) -> dict:
    """Returns (config_file, model_engine_file_template) for a PGIE preset.

    `model_engine_file_template` is a string containing `{N}` which gets
    substituted with the resolved batch-size.
    """
    presets = {
        "trafficcamnet": {
            "config-file": f"{ds_root}/samples/configs/deepstream-app/config_infer_primary.txt",
            "model-engine-file": f"{ds_root}/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b{{N}}_gpu0_fp16.engine",
            "labelfile-path": f"{ds_root}/samples/models/Primary_Detector/labels.txt",
        },
        "peoplenet": {
            "config-file": f"{ds_root}/samples/configs/tao_pretrained_models/config_infer_primary_peoplenet.txt",
            "model-engine-file": f"{ds_root}/samples/models/tao_pretrained_models/peoplenet/resnet34_peoplenet_int8.onnx_b{{N}}_gpu0_int8.engine",
            "labelfile-path": f"{ds_root}/samples/models/tao_pretrained_models/peoplenet/labels.txt",
        },
    }
    return presets.get(name, {})


def sgie_preset(name: str, ds_root: str) -> dict:
    presets = {
        "vehicletypes": {
            "config-file": f"{ds_root}/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.txt",
            "model-engine-file": f"{ds_root}/samples/models/Secondary_VehicleTypes/resnet18_vehicletypenet_pruned.onnx_b16_gpu0_fp16.engine",
            "labelfile-path": f"{ds_root}/samples/models/Secondary_VehicleTypes/labels.txt",
            "batch-size": 16,
            "gie-unique-id": 4,
            "operate-on-gie-id": 1,
            "operate-on-class-ids": "0;",
        },
        "vehiclemake": {
            "config-file": f"{ds_root}/samples/configs/deepstream-app/config_infer_secondary_vehiclemake.txt",
            "model-engine-file": f"{ds_root}/samples/models/Secondary_VehicleMake/resnet18_vehiclemakenet_pruned.onnx_b16_gpu0_fp16.engine",
            "labelfile-path": f"{ds_root}/samples/models/Secondary_VehicleMake/labels.txt",
            "batch-size": 16,
            "gie-unique-id": 5,
            "operate-on-gie-id": 1,
            "operate-on-class-ids": "0;",
        },
    }
    return presets.get(name, {})


def tracker_preset(name: str, ds_root: str) -> str:
    presets = {
        "NvDCF_perf": "config_tracker_NvDCF_perf.yml",
        "NvDCF_accuracy": "config_tracker_NvDCF_accuracy.yml",
        "NvSORT": "config_tracker_NvSORT.yml",
        "IOU": "config_tracker_IOU.yml",
        "NvDeepSORT": "config_tracker_NvDeepSORT.yml",
        "MaskTracker": "config_tracker_MaskTracker.yml",
    }
    fname = presets.get(name, presets["NvDCF_perf"])
    return f"{ds_root}/samples/configs/deepstream-app/{fname}"


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------
def load_template(name: str) -> dict:
    """Loads a single INI/GKeyFile-style template into {group: {key: val}}."""
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    parser.read(TEMPLATES / name, encoding="utf-8")
    out: dict[str, dict] = {}
    for section in parser.sections():
        out[section] = dict(parser.items(section))
    return out


def deep_merge(dst: dict, src: dict) -> dict:
    """Merge src into dst. Src values overwrite dst values."""
    for group, kv in src.items():
        if group not in dst:
            dst[group] = {}
        dst[group].update(kv)
    return dst


# ---------------------------------------------------------------------------
# Spec normalisation helpers
# ---------------------------------------------------------------------------
def spec_get(spec: dict, dotted: str, default=None):
    cur = spec
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def substitute_ds_root(value, ds_root: str):
    if isinstance(value, str):
        return value.replace("${DS_ROOT}", ds_root)
    return value


def normalise(spec: dict) -> dict:
    """Fills implicit defaults so the rest of the code can assume the shape."""
    spec.setdefault("deployment", {})
    d = spec["deployment"]
    d.setdefault("app", "classic")
    d.setdefault("format", "txt")
    d.setdefault("platform", "dgpu")
    d.setdefault("gpu_id", 0)
    d.setdefault("ds_root", DS_ROOT_DEFAULT)
    d.setdefault("output_path", f"./generated_config.{d['format']}")

    spec.setdefault("profile", "perf")

    spec.setdefault("sources", {})
    s = spec["sources"]
    s.setdefault("kind", "multi_uri")
    s.setdefault("uris", [])
    s.setdefault("num_sources", max(1, len(s.get("uris") or []) or 1))
    s.setdefault("live", "auto")
    s.setdefault("use_nvmultiurisrcbin", s["kind"] == "source_list")
    s.setdefault("simulate_fps_interval_ms", 0)

    spec.setdefault("streammux", {})
    sm = spec["streammux"]
    sm.setdefault("batch_size", "auto")
    sm.setdefault("batched_push_timeout", "auto")
    sm.setdefault("width", 1920)
    sm.setdefault("height", 1080)
    sm.setdefault("overrides", {})

    spec.setdefault("primary_gie", {"enable": True, "preset": "trafficcamnet", "interval": 0, "batch_size": "auto"})
    spec.setdefault("tracker", {"enable": False, "preset": "NvDCF_perf", "tracker_width": 960, "tracker_height": 544})
    spec.setdefault("secondary_gies", [])
    spec.setdefault("osd", {"enable": True})
    spec.setdefault("tiled_display", {"enable": "auto", "rows": "auto", "columns": "auto", "width": 1280, "height": 720})
    spec.setdefault("sinks", [])
    spec.setdefault("tests", {"file_loop": "auto"})

    # Auto-promote test5 if any sink has consumer enabled.
    if d["app"] == "classic":
        for sink in spec["sinks"]:
            if sink.get("type") == "msgbroker" and sink.get("consumer", {}).get("enable"):
                d["app"] = "test5"
                sys.stderr.write("[render] auto-promoted deployment.app -> test5 (msgbroker consumer enabled)\n")
                break

    # Resolve live=auto
    if s["live"] == "auto":
        s["live"] = s["kind"] in {"rtsp", "v4l2", "csi"}

    return spec


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def build_sources(spec: dict) -> dict[str, dict]:
    """Returns {group_name: {key: value}} for one or more source* groups."""
    s = spec["sources"]
    kind = s["kind"]
    num = int(s["num_sources"])
    uris = s.get("uris") or []
    ds_root = spec["deployment"]["ds_root"]

    out: dict[str, dict] = {}

    if kind == "source_list":
        base = load_template("source-list-multiuri.txt")
        base["source-list"]["num-source-bins"] = str(num)
        materialized_uris = uris * num if len(uris) == 1 else uris
        if uris:
            base["source-list"]["list"] = ";".join(substitute_ds_root(u, ds_root) for u in materialized_uris)
        base["source-list"]["max-batch-size"] = str(num)
        base["source-list"]["use-nvmultiurisrcbin"] = "1" if s["use_nvmultiurisrcbin"] else "0"
        # sensor-id-list / sensor-name-list semicolon-counts MUST equal
        # num-source-bins or nvmultiurisrcbin aborts on init.
        base["source-list"]["sensor-id-list"] = ";".join(
            f"UniqueSensorId{i+1}" for i in range(num)
        )
        base["source-list"]["sensor-name-list"] = ";".join(
            f"UniqueSensorName{i+1}" for i in range(num)
        )
        # simulate-fps-interval-ms paces FILE URIs only; emitted into
        # [source-attr-all]. 0 means "no pacing" (works for live/rtsp).
        base["source-attr-all"]["simulate-fps-interval-ms"] = str(
            int(s.get("simulate_fps_interval_ms", 0) or 0)
        )
        return base

    if kind == "v4l2":
        base = load_template("source-camera-v4l2.txt")
        base["source0"]["camera-v4l2-dev-node"] = str(s.get("v4l2_dev_node", 0))
        base["source0"]["camera-width"] = str(s.get("camera_width", 1920))
        base["source0"]["camera-height"] = str(s.get("camera_height", 1080))
        base["source0"]["camera-fps-n"] = str(s.get("camera_fps_n", 30))
        base["source0"]["camera-fps-d"] = str(s.get("camera_fps_d", 1))
        return base

    if kind == "csi":
        base = load_template("source-csi.txt")
        base["source0"]["camera-csi-sensor-id"] = str(s.get("csi_sensor_id", 0))
        base["source0"]["camera-width"] = str(s.get("camera_width", 1920))
        base["source0"]["camera-height"] = str(s.get("camera_height", 1080))
        base["source0"]["camera-fps-n"] = str(s.get("camera_fps_n", 30))
        base["source0"]["camera-fps-d"] = str(s.get("camera_fps_d", 1))
        return base

    if kind == "rtsp":
        # One [source<i>] per URI.
        for i, uri in enumerate(uris):
            frag = load_template("source-rtsp.txt")
            entry = frag["source0"]
            entry["uri"] = substitute_ds_root(uri, ds_root)
            entry["num-sources"] = "1"
            out[f"source{i}"] = entry
        return out

    if kind == "file":
        # One [source<i>] per URI (type=2 single URI).
        for i, uri in enumerate(uris):
            frag = load_template("source-uri-file.txt")
            entry = frag["source0"]
            entry["uri"] = substitute_ds_root(uri, ds_root)
            entry["type"] = "2"
            entry["num-sources"] = "1"
            out[f"source{i}"] = entry
        return out

    # multi_uri (default): one [source0] type=3 replicating
    base = load_template("source-uri-file.txt")
    if uris:
        base["source0"]["uri"] = substitute_ds_root(uris[0], ds_root)
    base["source0"]["num-sources"] = str(num)
    return base


def build_streammux(spec: dict) -> dict[str, dict]:
    profile = spec["profile"]
    mapping = {
        "latency": "streammux-realtime.txt",
        "perf": "streammux-perf.txt",
        "display": "streammux-display.txt",
        "custom": "streammux-perf.txt",  # neutral; user must override
    }
    sect = load_template(mapping[profile])
    sm_spec = spec["streammux"]
    width = sm_spec.get("width", 1920)
    height = sm_spec.get("height", 1080)
    sect["streammux"]["width"] = str(width)
    sect["streammux"]["height"] = str(height)
    sect["streammux"]["gpu-id"] = str(spec["deployment"]["gpu_id"])

    num = int(spec["sources"]["num_sources"])
    bs = sm_spec.get("batch_size", "auto")
    if bs == "auto":
        bs = num
    sect["streammux"]["batch-size"] = str(int(bs))

    bpt = sm_spec.get("batched_push_timeout", "auto")
    if bpt == "auto":
        bpt = 33333 if profile == "latency" else 40000
    sect["streammux"]["batched-push-timeout"] = str(int(bpt))

    live = spec["sources"]["live"]
    if profile == "perf" and not live:
        sect["streammux"]["live-source"] = "0"
    elif profile == "perf" and live:
        sect["streammux"]["live-source"] = "1"
    elif profile == "display":
        sect["streammux"]["live-source"] = "1" if live else "0"

    # Overrides last.
    for k, v in sm_spec.get("overrides", {}).items():
        sect["streammux"][k] = str(v)
    return sect


def build_pgie(spec: dict) -> dict[str, dict]:
    g = spec["primary_gie"]
    if not _spec_enabled(g.get("enable", True)):
        return {}
    ds_root = spec["deployment"]["ds_root"]
    bs = g.get("batch_size", "auto")
    if bs == "auto":
        sm_bs = spec["streammux"].get("batch_size", "auto")
        bs = int(sm_bs) if sm_bs != "auto" else int(spec["sources"]["num_sources"])
    bs = int(bs)

    sect = {
        "enable": "1",
        "gpu-id": str(spec["deployment"]["gpu_id"]),
        "batch-size": str(bs),
        "interval": str(g.get("interval", 0)),
        "gie-unique-id": "1",
        "bbox-border-color0": "1;0;0;1",
        "bbox-border-color1": "0;1;1;1",
        "bbox-border-color2": "0;0;1;1",
        "bbox-border-color3": "0;1;0;1",
        "nvbuf-memory-type": "0",
    }
    preset = g.get("preset", "trafficcamnet")
    if preset == "custom":
        if g.get("config_file"):
            sect["config-file"] = substitute_ds_root(g["config_file"], ds_root)
        if g.get("model_engine_file"):
            sect["model-engine-file"] = substitute_ds_root(g["model_engine_file"], ds_root)
    else:
        p = pgie_preset(preset, ds_root)
        if "config-file" in p:
            sect["config-file"] = p["config-file"]
        if "model-engine-file" in p:
            sect["model-engine-file"] = p["model-engine-file"].format(N=bs)
        if "labelfile-path" in p:
            sect["labelfile-path"] = p["labelfile-path"]
    return {"primary-gie": sect}


def build_tracker(spec: dict) -> dict[str, dict]:
    t = spec["tracker"]
    if not t.get("enable", False):
        return {}
    ds_root = spec["deployment"]["ds_root"]
    sect = {
        "enable": "1",
        "tracker-width": str(t.get("tracker_width", 960)),
        "tracker-height": str(t.get("tracker_height", 544)),
        "ll-lib-file": f"{ds_root}/lib/libnvds_nvmultiobjecttracker.so",
        "ll-config-file": tracker_preset(t.get("preset", "NvDCF_perf"), ds_root),
        "gpu-id": str(spec["deployment"]["gpu_id"]),
        "display-tracking-id": "1",
    }
    return {"tracker": sect}


def build_sgies(spec: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    ds_root = spec["deployment"]["ds_root"]
    for i, sg in enumerate(spec.get("secondary_gies", []) or []):
        if isinstance(sg, str):
            name, custom = sg, {}
        else:
            name = sg.get("preset", "custom")
            custom = sg
        if name == "custom":
            sect = {
                "enable": "1",
                "gpu-id": str(spec["deployment"]["gpu_id"]),
                "batch-size": str(custom.get("batch_size", 16)),
                "gie-unique-id": str(custom.get("gie_unique_id", 4 + i)),
                "operate-on-gie-id": str(custom.get("operate_on_gie_id", 1)),
                "operate-on-class-ids": str(custom.get("operate_on_class_ids", "0;")),
            }
            if custom.get("config_file"):
                sect["config-file"] = substitute_ds_root(custom["config_file"], ds_root)
            if custom.get("model_engine_file"):
                sect["model-engine-file"] = substitute_ds_root(custom["model_engine_file"], ds_root)
        else:
            p = sgie_preset(name, ds_root)
            sect = {
                "enable": "1",
                "gpu-id": str(spec["deployment"]["gpu_id"]),
                "batch-size": str(p.get("batch-size", 16)),
                "gie-unique-id": str(p.get("gie-unique-id", 4 + i)),
                "operate-on-gie-id": str(p.get("operate-on-gie-id", 1)),
                "operate-on-class-ids": str(p.get("operate-on-class-ids", "0;")),
                "config-file": p.get("config-file", ""),
                "model-engine-file": p.get("model-engine-file", ""),
                "labelfile-path": p.get("labelfile-path", ""),
            }
        out[f"secondary-gie{i}"] = sect
    return out


def build_osd_and_tiled(spec: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    profile = spec["profile"]
    num = int(spec["sources"]["num_sources"])

    td_spec = spec["tiled_display"]
    enable = td_spec.get("enable", "auto")
    if enable == "auto":
        enable = profile == "display"

    # Map the user-facing value to the integer that deepstream-app expects.
    # deepstream-app's tiled-display.enable has THREE values, not two:
    #   0 = no tile. Demuxer is AUTO-CREATED when sinks declare distinct
    #       source-ids; each sink receives one per-source stream. This is
    #       the canonical "N sources -> N independent sinks" demux pattern.
    #       No link-to-demux property required.
    #   1 = nvmultistreamtiler -- grid all sources into one output buffer
    #       (display use case).
    #   2 = parallel demux + tile. Tiler runs as usual AND in parallel one
    #       sink (with link-to-demux=1) taps a single demuxed source pad.
    #       Only ONE link-to-demux=1 sink is supported (any others ignored).
    #       Rare; mostly used when you want to record a single source while
    #       still rendering the tile.
    #
    # Spec values:
    #   enable: true | 1            -> integer 1 (tile)
    #   enable: false | 0           -> integer 0 (per-source demux if sinks have source-id)
    #   enable: demux               -> integer 0 (alias; emphasises N->N demux)
    #   enable: parallel_demux | 2  -> integer 2 (advanced: tile + one demux tap)
    sval = str(enable).lower() if isinstance(enable, str) else enable
    if sval == "demux":
        enable_int = 0
    elif sval in ("parallel_demux", 2, "2"):
        enable_int = 2
    elif sval in (True, 1, "1", "true"):
        enable_int = 1
    else:
        enable_int = 0

    rows = td_spec.get("rows", "auto")
    cols = td_spec.get("columns", "auto")
    if rows == "auto":
        rows = max(1, math.ceil(math.sqrt(num)))
    if cols == "auto":
        cols = max(1, math.ceil(num / rows))

    out["tiled-display"] = {
        "enable": str(enable_int),
        "rows": str(int(rows)),
        "columns": str(int(cols)),
        "width": str(td_spec.get("width", 1280)),
        "height": str(td_spec.get("height", 720)),
        "gpu-id": str(spec["deployment"]["gpu_id"]),
        "nvbuf-memory-type": "4" if spec["deployment"]["platform"] == "jetson" else "0",
    }

    # Per-source demux validation (the N->N pattern, enable_int==0 with
    # multiple distinct source_ids in sinks).
    is_demux_request = (
        str(td_spec.get("enable", "")).lower() == "demux"
        or any(
            sink.get("source_id") not in (None, 0)
            for sink in spec.get("sinks", [])
        )
    )
    if enable_int == 0 and is_demux_request:
        source_ids = [int(sink.get("source_id", 0)) for sink in spec.get("sinks") or []]
        n_sources = int(spec["sources"]["num_sources"])
        missing = [i for i in range(n_sources) if i not in source_ids]
        if missing:
            sys.stderr.write(
                f"[render] WARN: demux requested but no sink declared "
                f"source_id={missing}. Source(s) {missing} will be dropped at "
                f"the demuxer (no consumer). Add a sink with source_id=<i> for "
                f"each source you want to keep.\n"
            )
        sys.stderr.write(
            "[render] tiled-display.enable=0 + per-sink source_id => "
            "deepstream-app auto-creates nvstreamdemux. Each [sink<N>] "
            "binds to one demuxed source by source-id. No link-to-demux "
            "property needed (that is for enable=2 'parallel demux' only).\n"
        )
    elif enable_int == 2:
        sys.stderr.write(
            "[render] tiled-display.enable=2 (parallel demux + tile): "
            "exactly ONE sink must set link_to_demux=true; any extras "
            "are silently dropped by deepstream-app. Use enable=demux "
            "(=0) for the N-sources-to-N-sinks pattern instead.\n"
        )

    osd = spec["osd"]
    if not osd.get("enable", True):
        out.setdefault("osd", {})["enable"] = "0"
    return out


def build_sinks(spec: dict) -> dict[str, dict]:
    """Returns sink0..N groups plus an optional message-consumer0."""
    sinks_spec = spec.get("sinks") or []
    profile = spec["profile"]
    platform = spec["deployment"]["platform"]
    ds_root = spec["deployment"]["ds_root"]

    if not sinks_spec:
        # Profile default sink
        if profile == "latency" or profile == "perf":
            sinks_spec = [{"type": "fakesink"}]
        elif profile == "display":
            sinks_spec = [{"type": "egl"}]
        else:
            sinks_spec = [{"type": "fakesink"}]

    out: dict[str, dict] = {}
    consumer_idx = 0
    for i, sink in enumerate(sinks_spec):
        stype = sink.get("type", "fakesink")
        if stype == "fakesink":
            if profile == "latency":
                frag = load_template("sink-fakesink-latency.txt")
            else:
                frag = load_template("sink-fakesink-perf.txt")
            sect = frag["sink0"]
        elif stype == "egl":
            if platform == "jetson":
                frag = load_template("sink-nv3dsink-jetson.txt")
            else:
                frag = load_template("sink-egl.txt")
            sect = frag["sink0"]
        elif stype == "nv3dsink":
            frag = load_template("sink-nv3dsink-jetson.txt")
            sect = frag["sink0"]
        elif stype == "file":
            frag = load_template("sink-file-mp4.txt")
            sect = frag["sink0"]
            for k, v in [
                ("container", sink.get("container")),
                ("codec", sink.get("codec")),
                ("enc-type", sink.get("enc_type")),
                ("bitrate", sink.get("bitrate")),
                ("profile", sink.get("profile")),
                ("output-file", sink.get("output_file")),
            ]:
                if v is not None:
                    sect[k] = str(v)
        elif stype == "rtsp_out":
            frag = load_template("sink-rtsp-out.txt")
            sect = frag["sink0"]
            for k, v in [
                ("codec", sink.get("codec")),
                ("enc-type", sink.get("enc_type")),
                ("bitrate", sink.get("bitrate")),
                ("profile", sink.get("profile")),
                ("rtsp-port", sink.get("rtsp_port")),
                ("udp-port", sink.get("udp_port")),
            ]:
                if v is not None:
                    sect[k] = str(v)
        elif stype == "msgbroker":
            frag = load_template("sink-msgbroker.txt")
            sect = frag["sink0"]
            for k, v in [
                ("msg-conv-config", sink.get("msg_conv_config")),
                ("msg-conv-payload-type", sink.get("payload_type")),
                ("msg-broker-proto-lib", sink.get("proto_lib")),
                ("msg-broker-conn-str", sink.get("conn_str")),
                ("topic", sink.get("topic")),
                ("msg-broker-config", sink.get("broker_config")),
                ("disable-msgconv", sink.get("disable_msgconv")),
            ]:
                if v is not None and v != "":
                    sect[k] = substitute_ds_root(str(v), ds_root)
            # Consumer
            cons = sink.get("consumer") or {}
            if cons.get("enable"):
                csec = {
                    "enable": "1",
                    "proto-lib": substitute_ds_root(
                        cons.get("proto_lib", "${DS_ROOT}/lib/libnvds_kafka_proto.so"),
                        ds_root,
                    ),
                    "conn-str": cons.get("conn_str", "<host>;<port>"),
                    "config-file": substitute_ds_root(
                        cons.get("config_file", "${DS_ROOT}/samples/configs/deepstream-app-test5/cfg_kafka.txt"),
                        ds_root,
                    ),
                    "subscribe-topic-list": ";".join(cons.get("subscribe_topics", []) or ["topic1"]),
                }
                out[f"message-consumer{consumer_idx}"] = csec
                consumer_idx += 1
        else:
            sys.stderr.write(f"[render] unknown sink type {stype!r}; skipping\n")
            continue

        # Common per-sink overrides (applied to any sink type)
        if "sync" in sink and sink["sync"] != "auto":
            sect["sync"] = "1" if sink["sync"] else "0"
        sect["source-id"] = str(sink.get("source_id", 0))
        # link_to_demux=1 is the "parallel demux + tile" tap. Required only
        # when tiled_display.enable=2 (parallel_demux). Harmless otherwise
        # because deepstream-app reads it as 0 by default.
        if sink.get("link_to_demux"):
            sect["link-to-demux"] = "1"
        sect["gpu-id"] = str(spec["deployment"]["gpu_id"])
        if spec["deployment"]["platform"] == "jetson":
            sect["nvbuf-memory-type"] = "4"
        if sink.get("enable") is False:
            sect["enable"] = "0"
        out[f"sink{i}"] = sect

    return out


def build_tests(spec: dict) -> dict[str, dict]:
    fl = spec["tests"].get("file_loop", "auto")
    if fl == "auto":
        fl = 1 if spec["profile"] in {"latency", "perf"} else 0
    return {"tests": {"file-loop": str(int(bool(fl)))}}


# ---------------------------------------------------------------------------
# Source-list specific overlays
# ---------------------------------------------------------------------------
def apply_source_list_profile(spec: dict, cfg: dict) -> None:
    if spec["sources"]["kind"] != "source_list":
        return
    if spec["profile"] == "latency":
        cfg.setdefault("source-list", {})["low-latency-mode"] = "1"
    n = int(spec["sources"]["num_sources"])
    cfg.setdefault("source-list", {})["max-batch-size"] = str(n)

    # Auto-pace file URIs to simulate a live camera. Without this the
    # decoder races ahead and queues frames, making the measured latency
    # smaller than what a real camera would actually produce.
    # Only applies when the user hasn't already set a value, when at least
    # one URI is file:// (live sources self-pace), and only under the
    # latency profile (perf/display don't care about honest latency math).
    if spec["profile"] == "latency":
        existing = int(spec["sources"].get("simulate_fps_interval_ms", 0) or 0)
        uris = spec["sources"].get("uris") or []
        has_file_uri = any(str(u).startswith("file://") for u in uris)
        if existing == 0 and has_file_uri:
            cfg.setdefault("source-attr-all", {})["simulate-fps-interval-ms"] = "33"
            sys.stderr.write(
                "[render] latency profile + file source -> auto-set "
                "simulate-fps-interval-ms=33 in [source-attr-all] (paces "
                "the file decoder to ~30 fps so end-to-end latency "
                "reflects what a real camera would produce). "
                "Override via sources.simulate_fps_interval_ms.\n"
            )


# ---------------------------------------------------------------------------
# RTSP source overlay (for latency profile we want TCP + 100 latency etc.)
# ---------------------------------------------------------------------------
def apply_source_overrides(spec: dict, cfg: dict) -> None:
    if spec["sources"]["kind"] == "rtsp" and spec["profile"] == "latency":
        for k in list(cfg.keys()):
            if k.startswith("source") and not k.startswith("source-"):
                cfg[k].setdefault("latency", "100")
                cfg[k].setdefault("select-rtp-protocol", "4")


# ---------------------------------------------------------------------------
# Custom-groups bridge -- populated by deepstream-app-add-group.
#
# Descriptors live in <skill_root>/custom_groups/*.yaml and are written by
# `deepstream-app-add-group/scripts/register_with_renderer.py`
# after a successful parser extension. Each descriptor lets this generator
# emit a `[<group-name>]` block natively, so a user spec can reference the
# extended parser group as `<spec_key>: { enable: true, ... }`.
#
# Schema (descriptor file):
#   schema_version: 1
#   group_name: my-component
#   spec_key:   my_component
#   placement_hint: pre_mux | post_infer | per_instance
#   sub_config_file: false
#   keys:
#     - { name: enable,     spec_key: enable,     type: bool,   default: 0 }
#     - { name: server-url, spec_key: server_url, type: string, required: true }
# ---------------------------------------------------------------------------
CUSTOM_GROUPS_DIR = SKILL_ROOT / "custom_groups"


def _load_custom_group_descriptors() -> list[dict]:
    if not CUSTOM_GROUPS_DIR.is_dir():
        return []
    descriptors: list[dict] = []
    for path in sorted(CUSTOM_GROUPS_DIR.glob("*.yaml")):
        try:
            with path.open() as f:
                d = yaml.safe_load(f) or {}
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            sys.stderr.write(f"[render] WARN: failed to load custom group {path}: {exc}\n")
            continue
        if not isinstance(d, dict):
            sys.stderr.write(f"[render] WARN: custom group {path} must contain a mapping\n")
            continue
        if d.get("schema_version") != 1:
            sys.stderr.write(
                f"[render] WARN: ignoring custom group {path}: unsupported "
                f"schema_version {d.get('schema_version')!r}\n"
            )
            continue
        if not d.get("group_name") or not d.get("spec_key"):
            sys.stderr.write(f"[render] WARN: custom group {path} missing group_name/spec_key\n")
            continue
        d["_source_path"] = str(path)
        descriptors.append(d)
    return descriptors


def _format_custom_value(value, key_type: str) -> str:
    """Format a Python value into a deepstream-app .txt-friendly string."""
    if key_type == "bool":
        return "1" if value in (1, True, "1", "true", "True", "yes") else "0"
    return str(value)


def build_custom_groups(spec: dict, descriptors: list[dict]) -> dict[str, dict]:
    """Emit `{group-name: {key: value, ...}}` for every registered custom
    group that the user spec references. Validates unknown spec keys."""
    out: dict[str, dict] = {}
    for desc in descriptors:
        spec_key = desc["spec_key"]
        if spec_key not in spec:
            continue
        user_block = spec[spec_key] or {}
        if not isinstance(user_block, dict):
            sys.stderr.write(
                f"[render] WARN: spec.{spec_key} should be a mapping; got "
                f"{type(user_block).__name__}. Skipping custom group "
                f"[{desc['group_name']}].\n"
            )
            continue

        keys_by_spec_key = {k["spec_key"]: k for k in desc.get("keys") or []}
        unknown = set(user_block.keys()) - set(keys_by_spec_key.keys())
        for u in sorted(unknown):
            sys.stderr.write(
                f"[render] WARN: spec.{spec_key}.{u} is not a recognised key "
                f"for custom group [{desc['group_name']}]. Recognised keys: "
                f"{sorted(keys_by_spec_key)}\n"
            )

        rendered: dict[str, str] = {}
        for k in desc.get("keys") or []:
            spec_field = k["spec_key"]
            name = k["name"]
            if spec_field in user_block:
                rendered[name] = _format_custom_value(user_block[spec_field], k.get("type", "string"))
            elif "default" in k:
                rendered[name] = _format_custom_value(k["default"], k.get("type", "string"))
            elif k.get("required"):
                sys.stderr.write(
                    f"[render] ERROR: spec.{spec_key}.{spec_field} is required "
                    f"by custom group [{desc['group_name']}] but missing.\n"
                )
        if rendered:
            out[desc["group_name"]] = rendered
    return out


_CUSTOM_GROUPS_CACHE: list[dict] | None = None


def _custom_descriptors() -> list[dict]:
    global _CUSTOM_GROUPS_CACHE
    if _CUSTOM_GROUPS_CACHE is None:
        _CUSTOM_GROUPS_CACHE = _load_custom_group_descriptors()
    return _CUSTOM_GROUPS_CACHE


def _has_nonempty_string(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_inference_block(
    label: str,
    block: dict,
    *,
    allowed_presets: frozenset[str],
    default_preset: str,
    enabled: bool = True,
) -> None:
    """Validate model inputs for one enabled primary or secondary GIE spec."""
    preset = block.get("preset", default_preset)
    if not isinstance(preset, str) or preset not in allowed_presets:
        allowed = ", ".join(sorted(allowed_presets))
        raise SpecValidationError(
            f"spec.{label}.preset must be one of: {allowed}; got {preset!r}"
        )
    if not enabled or preset != "custom":
        return

    if block.get("generate_infer_config", False):
        infer = block.get("infer_config") or {}
        for key in ("onnx_file", "model_engine_file"):
            if key in infer and not isinstance(infer[key], str):
                raise SpecValidationError(
                    f"spec.{label}.infer_config.{key} must be a string; "
                    f"got {type(infer[key]).__name__}"
                )
        if not any(
            _has_nonempty_string(infer.get(key))
            for key in ("onnx_file", "model_engine_file")
        ):
            raise SpecValidationError(
                f"spec.{label}.infer_config requires a non-empty onnx_file or "
                "model_engine_file when generate_infer_config is true"
            )
        return

    missing = [
        key
        for key in ("config_file", "model_engine_file")
        if not _has_nonempty_string(block.get(key))
    ]
    if missing:
        raise SpecValidationError(
            f"spec.{label} with preset 'custom' requires non-empty "
            + " and ".join(missing)
        )


def validate_spec_shape(spec) -> None:
    """Reject unknown keys and unsafe shapes before applying defaults."""
    if not isinstance(spec, dict):
        raise SpecValidationError(
            f"spec root must be a mapping; got {type(spec).__name__}"
        )

    custom_keys = {
        desc["spec_key"] for desc in _custom_descriptors() if desc.get("spec_key")
    }
    allowed_keys = CORE_SPEC_KEYS | custom_keys
    unknown = sorted(
        repr(key) for key in spec if not isinstance(key, str) or key not in allowed_keys
    )
    if unknown:
        raise SpecValidationError(
            "unknown top-level spec key(s): " + ", ".join(unknown)
        )

    for key in sorted(MAPPING_SPEC_KEYS | custom_keys):
        if key in spec and not isinstance(spec[key], dict):
            raise SpecValidationError(
                f"spec.{key} must be a mapping; got {type(spec[key]).__name__}"
            )
    for key in sorted(SEQUENCE_SPEC_KEYS):
        if key in spec and not isinstance(spec[key], list):
            raise SpecValidationError(
                f"spec.{key} must be a list; got {type(spec[key]).__name__}"
            )
    if "profile" in spec and not isinstance(spec["profile"], str):
        raise SpecValidationError(
            f"spec.profile must be a string; got {type(spec['profile']).__name__}"
        )

    sources = spec.get("sources", {})
    unknown_source_keys = sorted(
        repr(key)
        for key in sources
        if not isinstance(key, str) or key not in ALLOWED_SOURCE_KEYS
    )
    if unknown_source_keys:
        raise SpecValidationError(
            "unknown spec.sources key(s): " + ", ".join(unknown_source_keys)
        )
    if "kind" in sources and (
        not isinstance(sources["kind"], str)
        or sources["kind"] not in ALLOWED_SOURCE_KINDS
    ):
        allowed = ", ".join(sorted(ALLOWED_SOURCE_KINDS))
        raise SpecValidationError(
            f"spec.sources.kind must be one of: {allowed}; got {sources['kind']!r}"
        )
    if "num_sources" in sources and (
        isinstance(sources["num_sources"], bool)
        or not isinstance(sources["num_sources"], int)
        or sources["num_sources"] < 1
    ):
        raise SpecValidationError(
            "spec.sources.num_sources must be a positive integer; got "
            f"{sources['num_sources']!r}"
        )
    if "uris" in sources and not isinstance(sources["uris"], list):
        raise SpecValidationError(
            f"spec.sources.uris must be a list; got {type(sources['uris']).__name__}"
        )
    for index, uri in enumerate(sources.get("uris", [])):
        if not isinstance(uri, str):
            raise SpecValidationError(
                f"spec.sources.uris[{index}] must be a string; "
                f"got {type(uri).__name__}"
            )
    kind = sources.get("kind", "multi_uri")
    uris = sources.get("uris", [])
    num_sources = sources.get("num_sources", max(1, len(uris)))
    if kind in {"file", "rtsp"} and len(uris) != num_sources:
        raise SpecValidationError(
            f"spec.sources.num_sources ({num_sources}) must equal "
            f"len(spec.sources.uris) ({len(uris)}) for kind {kind!r}"
        )
    if kind == "source_list" and len(uris) not in {1, num_sources}:
        raise SpecValidationError(
            "spec.sources.uris must contain one reusable URI or exactly "
            f"num_sources ({num_sources}) entries for kind 'source_list'; got {len(uris)}"
        )
    if kind in {"v4l2", "csi"} and num_sources != 1:
        raise SpecValidationError(
            f"spec.sources.num_sources must be 1 for kind {kind!r}; "
            f"got {num_sources}"
        )
    if kind == "multi_uri" and len(uris) > 1:
        raise SpecValidationError(
            "spec.sources.kind 'multi_uri' accepts at most one URI because "
            f"it replicates that URI; got {len(uris)}"
        )
    if kind in {"v4l2", "csi"} and uris:
        raise SpecValidationError(
            f"spec.sources.kind {kind!r} does not accept spec.sources.uris"
        )
    streammux = spec.get("streammux", {})
    for key in ("batch_size", "batched_push_timeout"):
        value = streammux.get(key, "auto")
        if value != "auto" and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise SpecValidationError(
                f"spec.streammux.{key} must be an integer or 'auto'; got {value!r}"
            )
    if "overrides" in streammux and not isinstance(streammux["overrides"], dict):
        raise SpecValidationError(
            "spec.streammux.overrides must be a mapping; got "
            f"{type(streammux['overrides']).__name__}"
        )
    tracker = spec.get("tracker", {})
    if "tracker_config" in tracker and not isinstance(tracker["tracker_config"], dict):
        raise SpecValidationError(
            "spec.tracker.tracker_config must be a mapping; got "
            f"{type(tracker['tracker_config']).__name__}"
        )

    for index, sink in enumerate(spec.get("sinks", [])):
        if not isinstance(sink, dict):
            raise SpecValidationError(
                f"spec.sinks[{index}] must be a mapping; got {type(sink).__name__}"
            )
        if "consumer" in sink and not isinstance(sink["consumer"], dict):
            raise SpecValidationError(
                f"spec.sinks[{index}].consumer must be a mapping; "
                f"got {type(sink['consumer']).__name__}"
            )
    for index, sgie in enumerate(spec.get("secondary_gies", [])):
        if not isinstance(sgie, (dict, str)):
            raise SpecValidationError(
                "spec.secondary_gies"
                f"[{index}] must be a mapping or preset string; got "
                f"{type(sgie).__name__}"
            )
        block = {"preset": sgie} if isinstance(sgie, str) else sgie
        if "infer_config" in block and not isinstance(block["infer_config"], dict):
            raise SpecValidationError(
                f"spec.secondary_gies[{index}].infer_config must be a mapping; "
                f"got {type(block['infer_config']).__name__}"
            )
        _validate_inference_block(
            f"secondary_gies[{index}]",
            block,
            allowed_presets=SGIE_PRESETS,
            default_preset="custom",
        )

    primary = spec.get("primary_gie")
    if primary is not None:
        if "infer_config" in primary and not isinstance(primary["infer_config"], dict):
            raise SpecValidationError(
                "spec.primary_gie.infer_config must be a mapping; got "
                f"{type(primary['infer_config']).__name__}"
            )
        _validate_inference_block(
            "primary_gie",
            primary,
            allowed_presets=PGIE_PRESETS,
            default_preset="trafficcamnet",
            enabled=_spec_enabled(primary.get("enable", True)),
        )



# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------
def render(spec: dict) -> dict:
    """Builds the merged config dict from a normalised spec."""
    cfg: dict[str, dict] = {}

    if spec["deployment"]["app"] == "test5":
        deep_merge(cfg, load_template("base_test5.txt"))
    else:
        deep_merge(cfg, load_template("base_classic.txt"))

    # Sources -- inserted before streammux
    deep_merge(cfg, build_sources(spec))

    # Streammux + tiled-display defaults
    deep_merge(cfg, build_streammux(spec))

    # PGIE + tracker + SGIEs
    deep_merge(cfg, build_pgie(spec))
    deep_merge(cfg, build_tracker(spec))
    deep_merge(cfg, build_sgies(spec))

    # OSD + tiled-display overlay
    deep_merge(cfg, build_osd_and_tiled(spec))

    # Sinks + consumers
    deep_merge(cfg, build_sinks(spec))

    # Tests
    deep_merge(cfg, build_tests(spec))

    # Custom groups registered by deepstream-app-add-group (if any).
    deep_merge(cfg, build_custom_groups(spec, _custom_descriptors()))

    # Conditional overlays
    apply_source_list_profile(spec, cfg)
    apply_source_overrides(spec, cfg)

    # Platform-wide nvbuf-memory-type bump on Jetson
    if spec["deployment"]["platform"] == "jetson":
        for group in cfg.values():
            if "nvbuf-memory-type" in group and group["nvbuf-memory-type"] == "0":
                group["nvbuf-memory-type"] = "4"

    # gpu-id propagation
    gpu = str(spec["deployment"]["gpu_id"])
    for group in cfg.values():
        if "gpu-id" in group:
            group["gpu-id"] = gpu

    # ${DS_ROOT} substitution on every string value
    ds_root = spec["deployment"]["ds_root"]
    for group in cfg.values():
        for k, v in list(group.items()):
            group[k] = substitute_ds_root(v, ds_root)

    return cfg


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def ordered_groups(cfg: dict) -> list[str]:
    """Returns group names in canonical order, dynamic groups slotted in."""
    out: list[str] = []
    used: set[str] = set()

    def add(name: str):
        if name in cfg and name not in used:
            out.append(name)
            used.add(name)

    def add_prefix(prefix: str):
        keys = sorted([g for g in cfg if g.startswith(prefix) and not g.startswith(prefix + "-")],
                      key=lambda x: int(x[len(prefix):] or "0") if x[len(prefix):].isdigit() else 0)
        # The above excludes "source-list", "source-attr-all" (they have hyphen).
        for k in keys:
            if k not in used:
                out.append(k)
                used.add(k)

    # Bucket custom groups by placement hint so they land in sensible spots.
    custom_by_hint: dict[str, list[str]] = {"pre_mux": [], "post_infer": [], "per_instance": []}
    for desc in _custom_descriptors():
        if desc["group_name"] in cfg:
            hint = desc.get("placement_hint", "post_infer")
            custom_by_hint.setdefault(hint, []).append(desc["group_name"])

    def add_custom(hint: str) -> None:
        for name in custom_by_hint.get(hint, []):
            add(name)

    add("application")
    add("tiled-display")
    add("source-list")
    add("source-attr-all")
    add_prefix("source")
    add_custom("pre_mux")
    add("streammux")
    add("pre-process")
    add("primary-gie")
    add("tracker")
    add_prefix("secondary-gie")
    add_custom("post_infer")
    add("osd")
    add_prefix("sink")
    add_custom("per_instance")
    add("message-converter")
    add_prefix("message-consumer")
    add("nvds-analytics")
    add("tests")

    for k in cfg:
        if k not in used:
            out.append(k)
            used.add(k)
    return out


def to_txt(cfg: dict, header: str) -> str:
    lines = [header.rstrip(), ""]
    for group in ordered_groups(cfg):
        lines.append(f"[{group}]")
        for k, v in cfg[group].items():
            lines.append(f"{k}={v}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_yml(cfg: dict, header: str) -> str:
    out_lines = [header.rstrip(), ""]
    for group in ordered_groups(cfg):
        out_lines.append(f"{group}:")
        for k, v in cfg[group].items():
            sv = str(v)
            # Quote strings that look risky for the YAML parser.
            if any(c in sv for c in [":", "#", ";"]) or sv == "" or sv.strip() != sv:
                out_lines.append(f"  {k}: '{sv}'")
            else:
                out_lines.append(f"  {k}: {sv}")
        out_lines.append("")
    return "\n".join(out_lines).rstrip() + "\n"


HEADER = """\
################################################################################
# Generated by deepstream-generate-config/scripts/render.py
# Run: deepstream-app -c <this file>   (or deepstream-test5-app -c)
# Edit the spec and re-render; don't hand-edit this file because
# the next render will overwrite it.
################################################################################
"""


# ---------------------------------------------------------------------------
# Opt-in detailed config generation (config_infer_primary.txt / tracker.yml)
# ---------------------------------------------------------------------------
def _sidecar_output_path(out_dir: Path, requested_name: object) -> Path:
    """Resolve a sidecar name while keeping it inside the main output directory."""
    if not isinstance(requested_name, str) or not requested_name.strip():
        raise SpecValidationError("generated sidecar path must be a non-empty relative string")
    requested = Path(requested_name).expanduser()
    if requested.is_absolute():
        raise SpecValidationError(
            f"generated sidecar path must be relative to the main output directory: {requested_name!r}"
        )
    root = out_dir.resolve()
    candidate = (root / requested).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SpecValidationError(
            f"generated sidecar path escapes the main output directory: {requested_name!r}"
        ) from exc
    return candidate


def write_detailed_configs(spec: dict, cfg: dict, out_dir: Path) -> list[Path]:
    """Materialises any opt-in custom infer / tracker config files next to
    the main config and rewrites references in `cfg` to point at them.
    Returns the list of files written for logging.
    """
    written: list[Path] = []
    ds_root = spec["deployment"]["ds_root"]

    # PGIE custom infer config
    pg_spec = spec.get("primary_gie", {}) or {}
    if pg_spec.get("enable", True) and pg_spec.get("generate_infer_config"):
        infer = dict(pg_spec.get("infer_config") or {})
        infer.setdefault("network_type", "detector")
        infer.setdefault("gie_unique_id", 1)
        # batch_size: prefer infer block, else PGIE batch in cfg, else streammux.
        if "batch_size" not in infer:
            bs = cfg.get("primary-gie", {}).get("batch-size") or cfg.get("streammux", {}).get("batch-size", 1)
            infer["batch_size"] = int(bs)
        infer["gpu_id"] = spec["deployment"]["gpu_id"]
        # Substitute ${DS_ROOT}.
        for k, v in list(infer.items()):
            if isinstance(v, str):
                infer[k] = v.replace("${DS_ROOT}", ds_root)
        out_name = pg_spec.get("infer_config_path", "config_infer_primary_custom.txt")
        out_path = _sidecar_output_path(out_dir, out_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_infer.to_txt(render_infer.build_infer(infer), render_infer.HEADER), encoding="utf-8")
        cfg.setdefault("primary-gie", {})["config-file"] = str(out_path)
        # If the user gave model paths in infer_config, sync them onto primary-gie too.
        if infer.get("model_engine_file"):
            cfg["primary-gie"]["model-engine-file"] = infer["model_engine_file"]
        if infer.get("labelfile_path"):
            cfg["primary-gie"]["labelfile-path"] = infer["labelfile_path"]
        written.append(out_path)

    # SGIE custom infer configs
    for i, sg_spec in enumerate(spec.get("secondary_gies") or []):
        if not isinstance(sg_spec, dict):
            continue
        if not sg_spec.get("generate_infer_config"):
            continue
        infer = dict(sg_spec.get("infer_config") or {})
        infer.setdefault("network_type", "classifier")
        infer.setdefault("gie_unique_id", sg_spec.get("gie_unique_id", 4 + i))
        infer.setdefault("operate_on_gie_id", sg_spec.get("operate_on_gie_id", 1))
        infer.setdefault("operate_on_class_ids", sg_spec.get("operate_on_class_ids", "0"))
        if "batch_size" not in infer:
            infer["batch_size"] = int(sg_spec.get("batch_size", 16))
        infer["gpu_id"] = spec["deployment"]["gpu_id"]
        for k, v in list(infer.items()):
            if isinstance(v, str):
                infer[k] = v.replace("${DS_ROOT}", ds_root)
        out_name = sg_spec.get("infer_config_path", f"config_infer_secondary_custom_{i}.txt")
        out_path = _sidecar_output_path(out_dir, out_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_infer.to_txt(render_infer.build_infer(infer), render_infer.HEADER), encoding="utf-8")
        cfg.setdefault(f"secondary-gie{i}", {})["config-file"] = str(out_path)
        written.append(out_path)

    # Tracker custom YAML
    tk_spec = spec.get("tracker", {}) or {}
    if tk_spec.get("enable") and tk_spec.get("generate_tracker_config"):
        algo = (tk_spec.get("preset") or "NvDCF_perf").split("_")[0]  # NvDCF_perf -> NvDCF
        if algo == "NvDeepSORT":
            sys.stderr.write("[render] NvDeepSORT custom tracker generation not yet supported; emitting NvDCF base.\n")
            algo = "NvDCF"
        if algo == "MaskTracker":
            sys.stderr.write("[render] MaskTracker custom tracker generation not yet supported; emitting NvDCF base.\n")
            algo = "NvDCF"
        overrides = tk_spec.get("tracker_config") or {}
        text = render_tracker.build_tracker(algo, overrides)
        out_name = tk_spec.get("tracker_config_path", "config_tracker_custom.yml")
        out_path = _sidecar_output_path(out_dir, out_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        cfg.setdefault("tracker", {})["ll-config-file"] = str(out_path)
        written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
QUICKSTART_SPEC_PATH = TEMPLATES / "quickstart_spec.yaml"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Render a deepstream-app config from a spec.")
    ap.add_argument("--spec", default=None, help="Path to the YAML/JSON spec file.")
    ap.add_argument("--quickstart", action="store_true",
                    help="Use the bundled default spec (assets/templates/quickstart_spec.yaml). "
                         "Produces a known-good config that runs against a stock DeepStream install.")
    ap.add_argument("--out", default=None, help="Override deployment.output_path.")
    ap.add_argument("--print-resolved-spec", action="store_true", help="Dump resolved spec to stderr.")
    args = ap.parse_args(argv)

    if not args.spec and not args.quickstart:
        sys.stderr.write("[render] one of --spec or --quickstart is required.\n")
        return 2

    spec_path = QUICKSTART_SPEC_PATH if args.quickstart else Path(args.spec).expanduser().resolve()
    if not spec_path.is_file():
        sys.stderr.write(f"[render] spec not found: {spec_path}\n")
        return 2
    if args.quickstart:
        sys.stderr.write(f"[render] using bundled quickstart spec ({spec_path})\n")
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)
    except (OSError, UnicodeError, yaml.YAMLError) as ex:
        sys.stderr.write(f"[render] unable to read spec {spec_path}: {ex}\n")
        return 2
    if spec is None:
        spec = {}
    try:
        validate_spec_shape(spec)
    except SpecValidationError as ex:
        sys.stderr.write(f"[render] invalid spec: {ex}\n")
        return 2
    spec = normalise(spec)

    if not validate_profile(spec):
        return 2

    if args.out:
        spec["deployment"]["output_path"] = args.out
    elif args.quickstart:
        spec["deployment"]["output_path"] = "./quickstart_generated.txt"
    out_path = Path(spec["deployment"]["output_path"]).expanduser().resolve()

    # Honor extension if user explicitly passed --out.
    suffix = out_path.suffix.lower()
    if suffix in {".txt", ".yml", ".yaml"}:
        spec["deployment"]["format"] = "yml" if suffix in {".yml", ".yaml"} else "txt"

    cfg = render(spec)

    try:
        # Opt-in: also emit custom config_infer_primary.txt and tracker .yml.
        extras = write_detailed_configs(spec, cfg, out_path.parent)
        for p in extras:
            sys.stderr.write(f"[render] wrote {p}\n")

        fmt = spec["deployment"]["format"]
        text = to_yml(cfg, HEADER) if fmt == "yml" else to_txt(cfg, HEADER)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    except SpecValidationError as ex:
        sys.stderr.write(f"[render] invalid generated output path: {ex}\n")
        return 2
    except PermissionError as ex:
        # Strip any existing --out (and its value) so the suggested commands
        # below don't end up with two --out flags.
        argv_no_out: list[str] = []
        skip_next = False
        for tok in argv:
            if skip_next:
                skip_next = False
                continue
            if tok == "--out":
                skip_next = True
                continue
            if tok.startswith("--out="):
                continue
            argv_no_out.append(tok)
        script = Path(__file__).resolve()
        retry_command = shlex.join(
            ["python3", str(script), *argv_no_out, "--out", "./generated_config.txt"]
        )
        in_default_dir = str(out_path).startswith(DEFAULT_OUTPUT_DIR)

        sys.stderr.write(
            f"\n[render] permission denied: {ex}\n"
            f"[render] target was: {out_path}\n\n"
        )
        if in_default_dir:
            sys.stderr.write(
                f"The requested path is under the stock deepstream-app samples folder\n"
                f"  {DEFAULT_OUTPUT_DIR}\n"
                f"which is usually root-owned. Use a writable path instead:\n\n"
                f"  {retry_command}\n\n"
            )
        else:
            sys.stderr.write(
                f"You requested an unwritable path. Use a writable destination:\n\n"
                f"  {retry_command}\n\n"
            )
        return 1
    except (OSError, UnicodeError) as ex:
        sys.stderr.write(f"[render] unable to write generated config: {ex}\n")
        return 1

    sys.stderr.write(f"[render] wrote {out_path}\n")
    app = "deepstream-test5-app" if spec["deployment"]["app"] == "test5" else "deepstream-app"
    sys.stderr.write(f"[render] run: {shlex.join([app, '-c', str(out_path)])}\n")

    if args.print_resolved_spec:
        sys.stderr.write("[render] resolved spec:\n")
        yaml.safe_dump(spec, sys.stderr, default_flow_style=False, sort_keys=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
