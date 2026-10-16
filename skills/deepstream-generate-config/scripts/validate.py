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

"""Validate a generated deepstream-app / deepstream-test5 config file.

Usage:
    validate.py <config-file>

Arguments:
    Path to one generated DeepStream INI or YAML configuration file.

Output:
    Writes validation diagnostics to standard error without modifying the
    supplied configuration.

Exit codes:
    0 on success (warnings may still be printed), 1 for validation/read errors,
    and 2 for invalid command-line arguments. Re-render with a corrected spec;
    do not hand-edit generated output because the next render overwrites it.
"""
from __future__ import annotations

import argparse
import configparser
import re
import sys
from pathlib import Path

ERRORS: list[str] = []
WARNINGS: list[str] = []


def err(msg: str) -> None:
    ERRORS.append(msg)


def warn(msg: str) -> None:
    WARNINGS.append(msg)


def load_txt(path: Path) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    parser.read(path, encoding="utf-8")
    out: dict[str, dict[str, str]] = {}
    for s in parser.sections():
        out[s] = dict(parser.items(s))
    return out


def load_yml(path: Path) -> dict[str, dict[str, str]]:
    try:
        import yaml
    except ImportError:
        err(
            "PyYAML required to validate .yml configs; install it with "
            "python3 -m pip install -r scripts/requirements.txt."
        )
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        err(f"YAML parse error in {path}: {exc}")
        return {}
    except (OSError, UnicodeError) as exc:
        err(f"Unable to read YAML config {path}: {exc}")
        return {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        err(f"YAML root must be a mapping, got {type(data).__name__}.")
        return {}
    out: dict[str, dict[str, str]] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = {
                kk: "" if vv is None else str(vv)
                for kk, vv in v.items()
            }
    return out


def load(path: Path) -> dict[str, dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        return load_yml(path)
    return load_txt(path)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def _as_int(s: str | None) -> int | None:
    try:
        return int(s) if s is not None else None
    except (TypeError, ValueError):
        return None


def _config_enabled(value, *, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def check_batch_sizes(cfg) -> None:
    sm = cfg.get("streammux", {})
    sm_bs = _as_int(sm.get("batch-size"))
    if sm_bs is None:
        err("streammux.batch-size missing or non-integer.")
        return
    pgie = cfg.get("primary-gie", {})
    pgie_bs = _as_int(pgie.get("batch-size"))
    if pgie and _config_enabled(pgie.get("enable"), default=True):
        if pgie_bs is None:
            err("primary-gie enabled but primary-gie.batch-size is missing.")
        elif pgie_bs != sm_bs:
            err(f"primary-gie.batch-size ({pgie_bs}) must equal streammux.batch-size ({sm_bs}).")

    for group_name, kv in cfg.items():
        if not group_name.startswith("secondary-gie"):
            continue
        if not _config_enabled(kv.get("enable"), default=True):
            continue
        sgie_bs = _as_int(kv.get("batch-size"))
        if sgie_bs is None:
            warn(f"{group_name}.batch-size missing.")
        elif sgie_bs < sm_bs:
            warn(f"{group_name}.batch-size ({sgie_bs}) is smaller than streammux.batch-size ({sm_bs}); may starve.")


def check_source_count_alignment(cfg) -> None:
    """Require streammux batching to match the materialized source count."""
    source_list = cfg.get("source-list")
    if source_list:
        source_count = _as_int(source_list.get("num-source-bins"))
    else:
        source_groups = [
            values
            for name, values in cfg.items()
            if re.fullmatch(r"source\d+", name)
        ]
        source_count = sum(
            (_as_int(values.get("num-sources")) or 1)
            for values in source_groups
        )

    streammux_batch = _as_int(cfg.get("streammux", {}).get("batch-size"))
    if (
        source_count is not None
        and source_count > 0
        and streammux_batch is not None
        and source_count != streammux_batch
    ):
        err(
            f"materialized source count ({source_count}) must equal "
            f"streammux.batch-size ({streammux_batch})."
        )


def check_engine_suffix(cfg) -> None:
    """Warn when model-engine-file batch suffix _bN_ != batch-size."""
    for gname, kv in cfg.items():
        if not (gname == "primary-gie" or gname.startswith("secondary-gie")):
            continue
        mef = kv.get("model-engine-file", "")
        bs = _as_int(kv.get("batch-size"))
        if not mef or bs is None:
            continue
        m = re.search(r"_b(\d+)_", mef)
        if not m:
            warn(f"{gname}: model-engine-file has no _bN_ suffix — cannot verify batch ({mef}).")
            continue
        suffix_bs = int(m.group(1))
        if suffix_bs != bs:
            warn(f"{gname}: model-engine-file _b{suffix_bs}_ != batch-size {bs}; DeepStream will silently rebuild the engine.")


def check_inference_config_files(cfg) -> None:
    """Every enabled nvinfer group needs a non-empty config-file."""
    for group_name, values in cfg.items():
        if not (
            group_name == "primary-gie"
            or group_name.startswith("secondary-gie")
        ):
            continue
        if not _config_enabled(values.get("enable"), default=True):
            continue
        if not values.get("config-file", "").strip():
            err(
                f"{group_name}: enabled inference group requires a non-empty "
                "'config-file'."
            )



def check_streammux_realtime(cfg) -> None:
    sm = cfg.get("streammux", {})
    bpt = _as_int(sm.get("batched-push-timeout"))
    live = sm.get("live-source", "0")
    if live == "1" and bpt is not None and bpt > 40000:
        warn(f"streammux.live-source=1 with batched-push-timeout={bpt} > 40000 adds latency. Consider 33333.")
    if live == "0" and bpt is not None and bpt < 20000:
        warn(f"streammux.live-source=0 with batched-push-timeout={bpt} < 20000 may starve perf; consider 40000.")
    if live == "1":
        for k, expected in (("cache-buffer", "0"), ("align-first-buffer", "0"), ("sort-batch", "0")):
            if sm.get(k, expected) != expected:
                warn(f"streammux.{k}={sm.get(k)} with live-source=1 — latency profile expects {expected}.")


def check_sink_keys(cfg) -> None:
    for gname, kv in cfg.items():
        if not gname.startswith("sink"):
            continue
        if kv.get("enable", "1") == "0":
            continue
        stype = _as_int(kv.get("type"))
        if stype is None:
            err(f"{gname}: missing or non-integer type.")
            continue
        if stype == 4:  # RTSPStreaming
            for k in ("codec", "bitrate", "rtsp-port", "udp-port"):
                if k not in kv:
                    err(f"{gname} (RTSPStreaming): missing required key '{k}'.")
            if kv.get("sync", "0") == "0":
                warn(f"{gname}: RTSPStreaming with sync=0 streams faster than real time; consumer-side jitter likely.")
        elif stype == 3:  # File
            for k in ("codec", "bitrate", "output-file"):
                if k not in kv:
                    err(f"{gname} (File): missing required key '{k}'.")
        elif stype == 6:  # MsgConvBroker
            for k in ("msg-conv-config", "msg-broker-proto-lib", "msg-broker-conn-str", "topic"):
                if k not in kv:
                    err(f"{gname} (MsgConvBroker): missing required key '{k}'.")
            if kv.get("msg-broker-conn-str", "").startswith("<"):
                err(f"{gname}: msg-broker-conn-str looks like a placeholder ({kv.get('msg-broker-conn-str')!r}).")
            if kv.get("topic", "").startswith("<"):
                err(f"{gname}: topic looks like a placeholder ({kv.get('topic')!r}).")


def check_source_list(cfg) -> None:
    sl = cfg.get("source-list")
    if not sl:
        return
    n = _as_int(sl.get("num-source-bins"))
    mbs = _as_int(sl.get("max-batch-size"))
    if n is None:
        err("source-list.num-source-bins missing or non-integer.")
    if mbs is None:
        warn("source-list.max-batch-size missing; nvstreammux defaults may not match num-source-bins.")
    elif n is not None and mbs != n:
        err(f"source-list.max-batch-size ({mbs}) must equal num-source-bins ({n}).")
    if sl.get("use-nvmultiurisrcbin", "0") not in ("0", "1"):
        warn("source-list.use-nvmultiurisrcbin should be 0 or 1.")

    uri_entries = [value for value in sl.get("list", "").split(";") if value]
    if n is not None and len(uri_entries) != n:
        err(
            f"source-list.list has {len(uri_entries)} entries but "
            f"num-source-bins={n}; the counts must match."
        )

    # Mis-placed key: deepstream-app's [source-list] parser does NOT accept
    # 'simulate-fps-interval-ms'. The key lives in [source-attr-all].
    # Symptom at runtime: WARN parse_source_list: Unknown key
    # 'simulate-fps-interval-ms' for group [source-list], followed by other
    # parser errors as the keep-going parser misaligns subsequent keys.
    if "simulate-fps-interval-ms" in sl:
        err(
            "source-list.simulate-fps-interval-ms is not a valid key for "
            "[source-list]; deepstream-app's parser only accepts it in "
            "[source-attr-all]. Move the key (render.py does this "
            "automatically when sources.simulate_fps_interval_ms is set)."
        )

    # use-nvmultiurisrcbin=1 hard-requirements (each is a runtime crash if
    # missing). Source: gst-nvmultiurisrcbin / NvDsServer init paths.
    if sl.get("use-nvmultiurisrcbin", "0") == "1":
        # sensor-id-list / sensor-name-list count MUST match num-source-bins
        # or nvmultiurisrcbin aborts with "Mismatch in Sensor IDs provided
        # and num-source-bins" (parse_source_list:188).
        for k in ("sensor-id-list", "sensor-name-list"):
            v = sl.get(k)
            if v is None or v == "":
                err(
                    f"source-list.{k} is required when "
                    f"use-nvmultiurisrcbin=1; missing value triggers "
                    f"'Mismatch in Sensor IDs provided and num-source-bins'."
                )
                continue
            count = len([x for x in v.split(";") if x])
            if n is not None and count != n:
                err(
                    f"source-list.{k} has {count} entries but "
                    f"num-source-bins={n}; nvmultiurisrcbin aborts on "
                    f"mismatched counts."
                )

        # http-ip / http-port are read unconditionally to start the REST
        # endpoint. Missing values become std::string(nullptr) and abort
        # with 'basic_string: construction from null is not valid' before
        # the first frame is decoded.
        for k in ("http-ip", "http-port"):
            v = sl.get(k)
            if v is None or v == "":
                err(
                    f"source-list.{k} is required when "
                    f"use-nvmultiurisrcbin=1; missing value crashes "
                    f"nvmultiurisrcbin with "
                    f"\"basic_string: construction from null is not "
                    f"valid\" during init."
                )


def check_demux(cfg) -> None:
    """Validate the two demux paths: enable=0 (auto-demux) and enable=2 (parallel).

    enable=0 + sinks with distinct source-id: deepstream-app auto-creates
        nvstreamdemux; every source MUST have at least one sink consuming
        it (otherwise the source's demuxer pad has no consumer and frames
        are dropped at the demuxer).

    enable=2 (parallel demux + tile): EXACTLY one sink must have
        link-to-demux=1. Zero such sinks => deepstream-app errors with
        'sink for demux (use link-to-demux-only property) is not provided
        in the config file' at create_pipeline time. More than one sink
        with link-to-demux=1 => deepstream-app silently keeps the first
        and drops the rest (known limitation).
    """
    td = cfg.get("tiled-display", {})
    td_enable = td.get("enable", "0")

    sinks = [(g, kv) for g, kv in cfg.items()
             if re.fullmatch(r"sink\d+", g) and kv.get("enable", "1") != "0"]

    # Count distinct source-ids requested by enabled sinks.
    sink_source_ids = []
    for _, kv in sinks:
        sid = _as_int(kv.get("source-id"))
        if sid is not None:
            sink_source_ids.append(sid)
    distinct_sink_source_ids = set(sink_source_ids)

    # Count of sources (for the demux source-coverage check).
    source_groups = [g for g in cfg if re.fullmatch(r"source\d+", g)]
    if source_groups:
        n_sources = sum((_as_int(cfg[g].get("num-sources")) or 1) for g in source_groups)
    elif "source-list" in cfg:
        n_sources = _as_int(cfg["source-list"].get("num-source-bins")) or 0
    else:
        n_sources = 0

    if td_enable == "0" and len(distinct_sink_source_ids) > 1 and n_sources > 0:
        # Auto-demux pattern. Verify every source has a sink consumer.
        missing = [i for i in range(n_sources) if i not in distinct_sink_source_ids]
        if missing:
            warn(
                f"tiled-display.enable=0 with per-sink source-ids "
                f"{sorted(distinct_sink_source_ids)} but no sink for "
                f"source-id={missing}. Those sources will be dropped "
                f"at the auto-created nvstreamdemux. Add a sink with "
                f"source-id=<i> for each source you want to keep."
            )

    if td_enable == "2":
        link_sinks = [g for g, kv in sinks if kv.get("link-to-demux", "0") == "1"]
        if len(link_sinks) == 0:
            err(
                "tiled-display.enable=2 (parallel demux + tile) requires "
                "exactly one sink with link-to-demux=1. None found. "
                "deepstream-app fails create_pipeline with 'sink for demux "
                "(use link-to-demux-only property) is not provided in the "
                "config file'. For N sources -> N independent sinks use "
                "tiled-display.enable=0 (alias: enable: demux) instead -- "
                "demux is auto-created and link-to-demux is not needed."
            )
        elif len(link_sinks) > 1:
            warn(
                f"tiled-display.enable=2 with {len(link_sinks)} sinks "
                f"having link-to-demux=1 ({link_sinks}); deepstream-app "
                f"silently keeps the first and drops the rest. For N "
                f"sources -> N sinks use tiled-display.enable=0 instead."
            )

    # link-to-demux=1 outside enable=2 is a no-op; warn so users don't
    # think it does anything when they meant enable=0 auto-demux.
    if td_enable != "2":
        stray = [g for g, kv in sinks if kv.get("link-to-demux", "0") == "1"]
        if stray:
            warn(
                f"link-to-demux=1 on {stray} but tiled-display.enable="
                f"{td_enable!r} (not 2); deepstream-app ignores the flag. "
                f"For per-source demux, use tiled-display.enable=0 + "
                f"sink source-id; link-to-demux is only honoured under "
                f"enable=2 (parallel demux + tile)."
            )


def check_jetson_hints(cfg) -> None:
    """Best-effort: if any sink/group has nvbuf-memory-type=4, treat as Jetson and flag conflicts."""
    is_jetson_hint = any(g.get("nvbuf-memory-type") == "4" for g in cfg.values() if isinstance(g, dict))
    if not is_jetson_hint:
        return
    for gname, kv in cfg.items():
        if gname.startswith("sink") and kv.get("type") == "2":
            err(f"{gname}: type=2 (EglSink) is dGPU-only; on Jetson use type=7 (nv3dsink) or type=5 (nvdrmvideosink).")


def check_tiled_display(cfg) -> None:
    td = cfg.get("tiled-display", {})
    if td.get("enable", "0") != "1":
        return
    rows = _as_int(td.get("rows"))
    cols = _as_int(td.get("columns"))
    if rows is None or cols is None or rows < 1 or cols < 1:
        err(f"tiled-display enabled but rows/columns invalid (rows={rows}, columns={cols}).")
    # Count sources
    source_groups = [g for g in cfg if re.fullmatch(r"source\d+", g)]
    src_total = 0
    for g in source_groups:
        src_total += _as_int(cfg[g].get("num-sources")) or 1
    if not source_groups and "source-list" in cfg:
        src_total = _as_int(cfg["source-list"].get("num-source-bins")) or 0
    if src_total and rows and cols and rows * cols < src_total:
        warn(f"tiled-display grid {rows}x{cols}={rows*cols} < total streams {src_total}; some streams will be hidden.")


def check_msgbroker_consumer_app(cfg, path: Path) -> None:
    has_consumer = any(g.startswith("message-consumer") and cfg[g].get("enable", "0") == "1" for g in cfg)
    if has_consumer:
        # heuristic: classic deepstream-app ignores message-consumer; warn.
        if "message-converter" not in cfg and "message-consumer0" not in cfg:
            return
        # The actual app variant isn't recorded in the config; rely on filename.
        if "test5" not in path.stem.lower():
            warn("message-consumer enabled — run this config with deepstream-test5-app (classic deepstream-app ignores it).")


def check_relative_paths(cfg, cfg_dir: Path) -> None:
    """Best-effort existence check for file paths in the config."""
    checked: set[str] = set()
    for gname, kv in cfg.items():
        if not isinstance(kv, dict):
            continue
        for k, v in kv.items():
            if not isinstance(v, str):
                continue
            if not any(k.endswith(suf) for suf in ("-file", "-config", "config-file", "labelfile-path", "ll-lib-file", "ll-config-file")):
                continue
            if k == "output-file":
                continue
            if v.startswith(("<", "file://", "rtsp://")):
                continue
            if v in checked:
                continue
            checked.add(v)
            p = Path(v)
            if not p.is_absolute():
                p = (cfg_dir / v).resolve()
            if not p.exists() and "<" not in v:
                warn(f"{gname}.{k}: file not found on this machine ({p}). Ignore if running on a different host.")


def _report_findings() -> bool:
    if WARNINGS:
        sys.stderr.write("warnings:\n")
        for warning in WARNINGS:
            sys.stderr.write(f"  - {warning}\n")
    if ERRORS:
        sys.stderr.write("errors:\n")
        for error in ERRORS:
            sys.stderr.write(f"  - {error}\n")
        sys.stderr.write(
            f"FAILED: {len(ERRORS)} error(s), {len(WARNINGS)} warning(s)\n"
        )
        return True
    return False



def main(argv: list[str]) -> int:
    ERRORS.clear()
    WARNINGS.clear()
    parser = argparse.ArgumentParser(
        description="Validate a deepstream-app or deepstream-test5 config."
    )
    parser.add_argument("config_file", help="Path to a .txt, .yml, or .yaml config.")
    args = parser.parse_args(argv)

    path = Path(args.config_file).expanduser().resolve()
    if not path.is_file():
        sys.stderr.write(f"config file not found: {path}\n")
        return 2
    try:
        cfg = load(path)
    except (OSError, UnicodeError, configparser.Error) as exc:
        sys.stderr.write(f"unable to read config {path}: {exc}\n")
        return 1
    if not cfg:
        if ERRORS:
            _report_findings()
        else:
            sys.stderr.write(
                "no [groups] found — is this a deepstream-app config?\n"
            )
        return 1

    check_batch_sizes(cfg)
    check_source_count_alignment(cfg)
    check_inference_config_files(cfg)
    check_engine_suffix(cfg)
    check_streammux_realtime(cfg)
    check_sink_keys(cfg)
    check_source_list(cfg)
    check_jetson_hints(cfg)
    check_tiled_display(cfg)
    check_demux(cfg)
    check_msgbroker_consumer_app(cfg, path)
    check_relative_paths(cfg, path.parent)

    if _report_findings():
        return 1
    sys.stderr.write(f"OK ({len(WARNINGS)} warning(s))\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
