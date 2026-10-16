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

"""Audit/validate DeepStream deepstream-app / deepstream-test5 config files.

Default mode is offline/static parsing and design checks. Opt-in --probe-* flags
may run local subprocess tools, open RTSP/HTTP connections, and (with
--probe-pipeline --confirm-pipeline-probe) execute a bounded deepstream-app /
deepstream-test5-app smoke test. NTP checks live in scripts/ntp_probe.py.

The checks are intentionally conservative: errors are reserved for settings that
are likely to fail or route metadata incorrectly, while warnings call out
use-case mismatches and missing explicit design policy.

No third-party dependency is required for .txt configs. YAML requires PyYAML.
"""

from __future__ import annotations

import argparse
import configparser
import dataclasses
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse


@dataclasses.dataclass
class Finding:
    severity: str
    area: str
    message: str
    suggestion: str | None = None


FINDINGS: list[Finding] = []

USECASE_CHOICES = (
    "auto",
    "filesrc",
    "rtsp",
    "latency",
    "perf",
    "display",
    "msgbroker",
    "test5",
    "dynamic-rtsp",
    "recording",
    "rtsp-out",
    "jetson",
    "dgpu",
)


def fatal_markers(output: str) -> list[str]:
    """Return a bounded list of fatal/error lines from an app smoke run."""
    patterns = (
        r"\bERROR\b",
        r"FAILED",
        r"Failed to",
        r"No such file",
        r"not found",
        r"Could not create",
        r"Segmentation fault",
        r"core dumped",
    )
    markers = [
        line.strip()
        for line in output.splitlines()
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns)
    ]
    return markers[:8]


def redact_sensitive_text(text: str | None) -> str | None:
    if text is None:
        return None
    return re.sub(
        r"\b(rtsps?)://([^:/@\s]+):([^@\s]+)@",
        r"\1://\2:***@",
        text,
        flags=re.IGNORECASE,
    )


def add(severity: str, area: str, message: str, suggestion: str | None = None) -> None:
    FINDINGS.append(
        Finding(
            severity,
            area,
            redact_sensitive_text(message) or "",
            redact_sensitive_text(suggestion),
        )
    )


def err(area: str, message: str, suggestion: str | None = None) -> None:
    add("ERROR", area, message, suggestion)


def warn(area: str, message: str, suggestion: str | None = None) -> None:
    add("WARN", area, message, suggestion)


def info(area: str, message: str, suggestion: str | None = None) -> None:
    add("INFO", area, message, suggestion)


def as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def as_bool_str(value: Any, default: str = "0") -> str:
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in ("true", "yes", "on"):
        return "1"
    if s in ("false", "no", "off"):
        return "0"
    return str(value).strip()


def enabled(kv: dict[str, str]) -> bool:
    return as_bool_str(kv.get("enable"), "1") != "0"


def split_semicolon_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def normalize_usecases(values: list[str] | None) -> set[str]:
    if not values:
        return {"auto"}
    out: set[str] = set()
    for raw in values:
        for item in str(raw).split(","):
            item = item.strip().lower()
            if not item:
                continue
            if item not in USECASE_CHOICES:
                valid = ", ".join(USECASE_CHOICES)
                raise argparse.ArgumentTypeError(f"unknown usecase {item!r}; valid choices: {valid}")
            out.add(item)
    return out or {"auto"}


def load_txt(path: Path) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    parser.read(path, encoding="utf-8")
    return {section: dict(parser.items(section)) for section in parser.sections()}


def load_yml(path: Path) -> dict[str, dict[str, str]]:
    try:
        import yaml  # type: ignore
    except ImportError:
        err("parse", "PyYAML is required to validate .yml/.yaml configs (pip install pyyaml).")
        return {}

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = ""
        if mark is not None:
            location = f" at line {mark.line + 1}, column {mark.column + 1}"
        err(
            "parse",
            f"Malformed YAML{location}.",
            "Fix the YAML syntax, then re-run validation.",
        )
        return {}
    except (OSError, UnicodeError) as exc:
        err(
            "parse",
            f"Unable to read YAML config ({type(exc).__name__}).",
            "Verify that the file is readable UTF-8 text, then re-run validation.",
        )
        return {}
    out: dict[str, dict[str, str]] = {}
    if not isinstance(data, dict):
        err("parse", "YAML root must be a mapping of DeepStream config groups.")
        return out
    for group, value in data.items():
        if isinstance(value, dict):
            out[str(group)] = {str(k): str(v) for k, v in value.items()}
    return out


def load_config(path: Path) -> dict[str, dict[str, str]]:
    if path.suffix.lower() in (".yml", ".yaml"):
        return load_yml(path)
    return load_txt(path)


def source_groups(cfg: dict[str, dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    groups = []
    for name, kv in cfg.items():
        if re.fullmatch(r"source\d+", name) and enabled(kv):
            groups.append((name, kv))
    return groups


def sink_groups(cfg: dict[str, dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    groups = []
    for name, kv in cfg.items():
        if re.fullmatch(r"sink\d+", name) and enabled(kv):
            groups.append((name, kv))
    return groups


def gie_groups(cfg: dict[str, dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    groups = []
    for name, kv in cfg.items():
        if (name == "primary-gie" or re.fullmatch(r"secondary-gie\d*", name)) and enabled(kv):
            groups.append((name, kv))
    return groups


def active_source_count(cfg: dict[str, dict[str, str]]) -> int:
    groups = source_groups(cfg)
    if groups:
        total = 0
        for _, kv in groups:
            total += as_int(kv.get("num-sources")) or 1
        return total
    sl = cfg.get("source-list")
    if sl:
        return as_int(sl.get("num-source-bins")) or len(split_semicolon_list(sl.get("list")))
    return 0


def has_rtsp_source(cfg: dict[str, dict[str, str]]) -> bool:
    for _, kv in source_groups(cfg):
        uri = kv.get("uri", "")
        if as_int(kv.get("type")) == 4 or uri.startswith(("rtsp://", "rtsps://")):
            return True
    sl = cfg.get("source-list") or {}
    if any(x.startswith(("rtsp://", "rtsps://")) for x in split_semicolon_list(sl.get("list"))):
        return True
    sat = cfg.get("source-attr-all") or {}
    return as_int(sat.get("type")) == 4


def has_file_source(cfg: dict[str, dict[str, str]]) -> bool:
    for _, kv in source_groups(cfg):
        stype = as_int(kv.get("type"))
        uri = kv.get("uri", "")
        if stype == 2 and not uri.startswith(("rtsp://", "rtsps://")):
            return True
        if uri.startswith("file://"):
            return True
    sl = cfg.get("source-list") or {}
    uris = split_semicolon_list(sl.get("list"))
    if any(u.startswith("file://") or not u.startswith(("rtsp://", "rtsps://")) for u in uris):
        return bool(uris)
    sat = cfg.get("source-attr-all") or {}
    return as_int(sat.get("type")) == 2


def has_live_source(cfg: dict[str, dict[str, str]]) -> bool:
    for _, kv in source_groups(cfg):
        stype = as_int(kv.get("type"))
        if stype in (1, 4) or as_bool_str(kv.get("live-source"), "0") == "1":
            return True
    sl = cfg.get("source-list") or {}
    if sl.get("use-nvmultiurisrcbin") == "1":
        return True
    return has_rtsp_source(cfg)


def has_sink_type(cfg: dict[str, dict[str, str]], sink_type: int) -> bool:
    return any(as_int(kv.get("type")) == sink_type for _, kv in sink_groups(cfg))


def has_display_sink(cfg: dict[str, dict[str, str]]) -> bool:
    return any(as_int(kv.get("type")) in (2, 5, 7) for _, kv in sink_groups(cfg))


def source_list_uses_nvmultiurisrcbin(cfg: dict[str, dict[str, str]]) -> bool:
    sl = cfg.get("source-list") or {}
    return as_bool_str(sl.get("use-nvmultiurisrcbin"), "0") == "1"


def iter_source_uris(cfg: dict[str, dict[str, str]]) -> list[tuple[str, str]]:
    uris: list[tuple[str, str]] = []
    for name, kv in source_groups(cfg):
        uri = kv.get("uri")
        if uri:
            uris.append((name, uri))
    for idx, uri in enumerate(split_semicolon_list((cfg.get("source-list") or {}).get("list"))):
        uris.append((f"source-list[{idx}]", uri))
    return uris


def resolve_file_uri(uri: str, cfg_dir: Path) -> Path | None:
    if uri.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return None
    raw = uri
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        raw = parsed.path
    p = Path(raw)
    if not p.is_absolute():
        p = (cfg_dir / p).resolve()
    return p


def sink_type_names(cfg: dict[str, dict[str, str]]) -> set[int]:
    return {stype for _, kv in sink_groups(cfg) if (stype := as_int(kv.get("type"))) is not None}


def check_source_contract(cfg: dict[str, dict[str, str]]) -> None:
    for name, kv in source_groups(cfg):
        stype = as_int(kv.get("type"))
        uri = kv.get("uri", "")
        if stype is None:
            err("source", f"{name}: missing or non-integer type.")
            continue
        if stype == 2 and not uri:
            err("source", f"{name}: type=2 URI source requires uri.")
        if stype == 4:
            if not uri:
                err("source", f"{name}: type=4 RTSP source requires uri.")
            elif not uri.startswith(("rtsp://", "rtsps://")):
                warn("source", f"{name}: type=4 usually expects rtsp:// or rtsps:// URI, got {uri!r}.")
        if stype == 1:
            for key in ("camera-width", "camera-height", "camera-fps-n", "camera-fps-d", "camera-v4l2-dev-node"):
                if key not in kv:
                    warn("source", f"{name}: camera source missing {key}.")
        n = as_int(kv.get("num-sources"))
        if n is not None and n < 1:
            err("source", f"{name}: num-sources must be >= 1.")


def check_rtsp_policy(cfg: dict[str, dict[str, str]]) -> None:
    if not has_rtsp_source(cfg):
        return
    sm = cfg.get("streammux", {})
    attach = sm.get("attach-sys-ts-as-ntp")
    if attach is None:
        warn(
            "timestamp",
            "RTSP sources found but streammux.attach-sys-ts-as-ntp is not explicitly set. "
            "Parser default is 1 (system time). Use 1 for latency/local RTSP; use 0 for "
            "camera event-time only when RTCP sender reports/NTP are trustworthy.",
            "Add `attach-sys-ts-as-ntp=1` under [streammux] for local latency tests, or `attach-sys-ts-as-ntp=0` for production camera event-time after verifying RTCP/NTP.",
        )
    elif as_bool_str(attach) == "1":
        info("timestamp", "attach-sys-ts-as-ntp=1: frame_meta->ntp_timestamp uses system time.")
    elif as_bool_str(attach) == "0":
        info("timestamp", "attach-sys-ts-as-ntp=0: frame_meta->ntp_timestamp depends on RTSP sender NTP/RTCP.")
    else:
        warn("timestamp", f"streammux.attach-sys-ts-as-ntp should be 0 or 1, got {attach!r}.")

    if "sync-inputs-ntp" in sm and as_bool_str(sm.get("attach-sys-ts-as-ntp"), "1") == "1":
        warn("timestamp", "sync-inputs-ntp is configured while attach-sys-ts-as-ntp=1; NTP sync only helps if RTSP sender NTP is used.")
    if "frame-duration" in sm:
        warn("timestamp", "streammux.frame-duration is overridden; verify this is compensating for known jitter or missing frame-rate metadata.")
    if "max-latency" in sm and "sync-inputs" not in sm and "sync-inputs-ntp" not in sm:
        warn("timestamp", "streammux.max-latency is set without sync-inputs/sync-inputs-ntp; verify it has the intended effect.")


def check_rtsp_stability(cfg: dict[str, dict[str, str]]) -> None:
    for name, kv in source_groups(cfg):
        is_rtsp = as_int(kv.get("type")) == 4 or kv.get("uri", "").startswith(("rtsp://", "rtsps://"))
        if not is_rtsp:
            continue
        for key in ("latency", "rtsp-reconnect-interval-sec", "init-rtsp-reconnect-interval-sec", "rtsp-reconnect-attempts"):
            if key not in kv:
                warn("rtsp", f"{name}: RTSP source missing {key}.")
        proto = as_int(kv.get("select-rtp-protocol"))
        if proto is None:
            warn("rtsp", f"{name}: select-rtp-protocol not set; TCP (4) is safer for CI/unstable networks.", f"Add `select-rtp-protocol=4` to {name} unless UDP is required.")
        elif proto != 4:
            warn("rtsp", f"{name}: select-rtp-protocol={proto}; TCP (4) is usually safer unless UDP is intentional.")
        latency = as_int(kv.get("latency"))
        if latency is not None:
            if latency < 50:
                warn("rtsp", f"{name}: latency={latency} is very low; expect drops/jitter on lossy networks.")
            elif latency > 500:
                warn("rtsp", f"{name}: latency={latency} adds end-to-end delay; verify this is intentional.")


def check_streammux(cfg: dict[str, dict[str, str]]) -> None:
    sm = cfg.get("streammux")
    if not sm:
        err("streammux", "Missing [streammux] group.")
        return
    src_count = active_source_count(cfg)
    bs = as_int(sm.get("batch-size"))
    if bs is None:
        err("streammux", "streammux.batch-size missing or non-integer.")
    elif src_count and bs != src_count:
        warn("streammux", f"streammux.batch-size={bs} but active source count appears to be {src_count}.")

    if has_live_source(cfg) and as_bool_str(sm.get("live-source"), "0") != "1":
        warn("streammux", "Live/RTSP sources found but streammux.live-source is not 1.")

    bpt = as_int(sm.get("batched-push-timeout"))
    if as_bool_str(sm.get("live-source"), "0") == "1":
        if bpt is not None and bpt > 40000:
            warn("streammux", f"live-source=1 with batched-push-timeout={bpt} may add latency.")
        for key, expected in (("cache-buffer", "0"), ("align-first-buffer", "0"), ("sort-batch", "0")):
            if key in sm and as_bool_str(sm.get(key)) != expected:
                warn("streammux", f"live-source=1 with {key}={sm.get(key)}; latency configs usually use {expected}.")

    pool = as_int(sm.get("buffer-pool-size"))
    if pool is not None and src_count and pool < min(4, src_count):
        warn("streammux", f"buffer-pool-size={pool} may be small for {src_count} source(s).")

    width = as_int(sm.get("width"))
    height = as_int(sm.get("height"))
    if width is not None and width <= 0 or height is not None and height <= 0:
        err("streammux", f"Invalid streammux dimensions width={width}, height={height}.")


def check_batch_and_gie(cfg: dict[str, dict[str, str]]) -> None:
    sm_bs = as_int((cfg.get("streammux") or {}).get("batch-size"))
    enabled_gies = gie_groups(cfg)
    ids: dict[int, str] = {}
    for name, kv in enabled_gies:
        if "config-file" not in kv and "model-engine-file" not in kv and "onnx-file" not in kv:
            warn("gie", f"{name}: enabled but no config-file/model-engine-file/onnx-file found.")
        gid = as_int(kv.get("gie-unique-id"))
        if gid is not None:
            if gid in ids:
                err("gie", f"{name}: gie-unique-id={gid} duplicates {ids[gid]}.")
            ids[gid] = name
        bs = as_int(kv.get("batch-size"))
        if name == "primary-gie" and sm_bs is not None:
            if bs is None:
                err("gie", "primary-gie enabled but batch-size is missing.")
            elif bs != sm_bs:
                err("gie", f"primary-gie.batch-size ({bs}) must equal streammux.batch-size ({sm_bs}).")
        mef = kv.get("model-engine-file", "")
        if mef and bs is not None:
            m = re.search(r"_b(\d+)_", mef)
            if m and int(m.group(1)) != bs:
                warn("gie", f"{name}: model-engine-file _b{m.group(1)}_ != batch-size {bs}; engine may rebuild silently.")

    enabled_ids = set(ids)
    for name, kv in enabled_gies:
        if name.startswith("secondary-gie"):
            op = as_int(kv.get("operate-on-gie-id"))
            if op is None:
                warn("gie", f"{name}: missing operate-on-gie-id.")
            elif op not in enabled_ids:
                err("gie", f"{name}.operate-on-gie-id={op} does not reference an enabled GIE unique ID.")
            if "operate-on-class-ids" not in kv:
                warn("gie", f"{name}: operate-on-class-ids missing; verify classifier operates on intended classes.")


def check_preprocess(cfg: dict[str, dict[str, str]]) -> None:
    pre = cfg.get("pre-process")
    pre_enabled = bool(pre and enabled(pre))
    tensor_consumers = [name for name, kv in gie_groups(cfg) if as_bool_str(kv.get("input-tensor-meta"), "0") == "1"]
    if pre_enabled and "config-file" not in pre:
        err("preprocess", "pre-process enabled but config-file is missing.")
    if pre_enabled and not tensor_consumers:
        warn("preprocess", "pre-process enabled but no enabled GIE has input-tensor-meta=1.")
    if tensor_consumers and not pre_enabled:
        warn("preprocess", f"{tensor_consumers} use input-tensor-meta=1 but [pre-process] is not enabled.")


def check_source_list(cfg: dict[str, dict[str, str]]) -> None:
    sl = cfg.get("source-list")
    if not sl:
        return
    n = as_int(sl.get("num-source-bins"))
    uris = split_semicolon_list(sl.get("list"))
    if n is None:
        err("source-list", "num-source-bins missing or non-integer.")
    elif uris and len(uris) != n:
        warn("source-list", f"list has {len(uris)} URI(s) but num-source-bins={n}.")
    if not uris:
        warn("source-list", "source-list.list is missing or empty.", "Add semicolon-separated source URIs to [source-list].")
    for idx, uri in enumerate(uris):
        scheme = urlparse(uri).scheme
        if scheme and scheme not in ("file", "rtsp", "rtsps", "http", "https"):
            warn("source-list", f"source-list[{idx}] uses uncommon URI scheme {scheme!r}: {uri}.", "Verify the target source bin supports this scheme, or change to file/RTSP/HTTP(S).")
    max_bs = as_int(sl.get("max-batch-size"))
    if max_bs is None:
        warn("source-list", "max-batch-size missing; muxer defaults may not match num-source-bins.")
    elif n is not None and max_bs > n:
        err("source-list", f"max-batch-size ({max_bs}) > num-source-bins ({n}).", f"Set `max-batch-size={n}` or increase `num-source-bins` to match the intended stream count.")

    if "simulate-fps-interval-ms" in sl:
        err("source-list", "simulate-fps-interval-ms belongs in [source-attr-all], not [source-list].", "Move `simulate-fps-interval-ms` to [source-attr-all].")

    use_multi = as_bool_str(sl.get("use-nvmultiurisrcbin"), "0") == "1"
    if use_multi:
        for key in ("sensor-id-list", "sensor-name-list"):
            vals = split_semicolon_list(sl.get(key))
            if not vals:
                err("source-list", f"{key} is required when use-nvmultiurisrcbin=1.")
            elif n is not None and len(vals) != n:
                err("source-list", f"{key} has {len(vals)} entries but num-source-bins={n}.")
        for key in ("http-ip", "http-port"):
            if not sl.get(key):
                err("source-list", f"{key} is required when use-nvmultiurisrcbin=1.")
        if any(u.startswith(("rtsp://", "rtsps://")) for u in uris):
            warn("source-list", "RTSP with nvmultiurisrcbin is useful for REST add/remove or sensor metadata; for static RTSP sets, [sourceN] type=4 is simpler.")
        sat = cfg.get("source-attr-all") or {}
        if as_bool_str(sat.get("live-source"), as_bool_str((cfg.get("streammux") or {}).get("live-source"), "0")) != "1":
            warn("source-list", "nvmultiurisrcbin/live RTSP usually expects live-source=1.")


def check_decode_pressure(cfg: dict[str, dict[str, str]]) -> None:
    for name, kv in source_groups(cfg):
        if as_bool_str(kv.get("low-latency-decode"), "0") == "1":
            info("decode", f"{name}: low-latency-decode=1.")
        for key in ("drop-frame-interval", "dec-skip-frames"):
            val = as_int(kv.get(key))
            if val is not None and val > 0:
                warn("decode", f"{name}: {key}={val} drops/skips frames; verify analytics accuracy is acceptable.")
        if as_bool_str(kv.get("drop-on-latency"), "0") == "1":
            warn("decode", f"{name}: drop-on-latency=1 can drop frames under jitter.")
        leaky = as_int(kv.get("leaky"))
        if leaky is not None and leaky != 0:
            warn("decode", f"{name}: leaky={leaky}; queue may drop buffers.")
        extra = as_int(kv.get("num-extra-surfaces"))
        if extra is not None and extra < 1 and active_source_count(cfg) > 4:
            warn("decode", f"{name}: num-extra-surfaces={extra} may be low for multi-stream throughput.")


def check_sinks(cfg: dict[str, dict[str, str]]) -> None:
    for name, kv in sink_groups(cfg):
        stype = as_int(kv.get("type"))
        if stype is None:
            err("sink", f"{name}: missing or non-integer type.")
            continue
        if stype == 3:
            for key in ("codec", "bitrate", "output-file"):
                if key not in kv:
                    err("sink", f"{name} file sink missing {key}.")
            if "container" not in kv:
                warn("sink", f"{name} file sink missing container.")
        elif stype == 4:
            for key in ("codec", "bitrate", "rtsp-port", "udp-port"):
                if key not in kv:
                    err("sink", f"{name} RTSPStreaming sink missing {key}.")
            if as_bool_str(kv.get("sync"), "1") == "0":
                warn("sink", f"{name}: RTSP output with sync=0 can stream faster than real time.")
        elif stype == 6:
            for key in ("msg-conv-config", "msg-broker-proto-lib", "msg-broker-conn-str", "topic"):
                if key not in kv:
                    err("sink", f"{name} msgbroker sink missing {key}.")
            if kv.get("msg-broker-conn-str", "").startswith("<") or kv.get("topic", "").startswith("<"):
                err("sink", f"{name}: msgbroker connection/topic looks like a placeholder.")
        if as_bool_str(kv.get("qos"), "0") == "1":
            warn("sink", f"{name}: qos=1 may drop buffers under load; avoid if completeness is required.")


def check_demux(cfg: dict[str, dict[str, str]]) -> None:
    td = cfg.get("tiled-display", {})
    td_enable = str(td.get("enable", "0"))
    sinks = sink_groups(cfg)
    link_sinks = [name for name, kv in sinks if as_bool_str(kv.get("link-to-demux"), "0") == "1"]
    if td_enable == "2":
        if len(link_sinks) != 1:
            err("demux", f"tiled-display.enable=2 requires exactly one sink with link-to-demux=1; found {link_sinks}.")
    elif link_sinks:
        warn("demux", f"link-to-demux=1 on {link_sinks} but tiled-display.enable={td_enable}; flag is ignored outside enable=2.")

    source_ids = {as_int(kv.get("source-id")) for _, kv in sinks if as_int(kv.get("source-id")) is not None}
    src_count = active_source_count(cfg)
    if td_enable == "0" and len(source_ids) > 1 and src_count:
        missing = [i for i in range(src_count) if i not in source_ids]
        if missing:
            warn("demux", f"Per-source sink pattern has no sink for source-id(s) {missing}.")


def check_tracker(cfg: dict[str, dict[str, str]]) -> None:
    tr = cfg.get("tracker")
    if not tr or not enabled(tr):
        return
    for name, kv in source_groups(cfg):
        stype = as_int(kv.get("type"))
        uri = kv.get("uri", "")
        n_sources = as_int(kv.get("num-sources")) or 1
        if stype == 3 and uri.startswith(("rtsp://", "rtsps://")) and n_sources > 1:
            err(
                "tracker",
                f"{name}: tracker is enabled with URI_MULTIPLE (type=3) RTSP and num-sources={n_sources}; this can fail at runtime with `motFrame->numBuffers must be 1 when numTransforms is 1`.",
                "Use separate RTSP source groups (type=4 per source), use source-list/nvmultiurisrcbin for dynamic RTSP, or disable tracker for this URI_MULTIPLE RTSP pattern.",
            )
    w = as_int(tr.get("tracker-width"))
    h = as_int(tr.get("tracker-height"))
    for key, val in (("tracker-width", w), ("tracker-height", h)):
        if val is None:
            warn("tracker", f"{key} missing.")
        elif val % 32 != 0:
            warn("tracker", f"{key}={val} is not a multiple of 32; NvDCF/NvDeepSORT expect multiples of 32.")
    for key in ("ll-lib-file", "ll-config-file"):
        if key not in tr:
            warn("tracker", f"{key} missing.")
    if "operate-on-class-ids" in tr:
        info("tracker", f"tracker.operate-on-class-ids={tr.get('operate-on-class-ids')}; verify it matches PGIE classes.")


def check_smart_record(cfg: dict[str, dict[str, str]]) -> None:
    for name, kv in source_groups(cfg):
        if as_bool_str(kv.get("smart-record"), "0") == "0":
            continue
        for key in ("smart-rec-dir-path", "smart-rec-file-prefix", "smart-rec-container"):
            if key not in kv:
                err("smart-record", f"{name}: smart-record enabled but {key} missing.")
        if "smart-rec-cache" not in kv and "smart-rec-video-cache" not in kv:
            err("smart-record", f"{name}: smart-record enabled but smart-rec-cache missing.")
        if as_int(kv.get("smart-rec-duration")) == 0:
            warn("smart-record", f"{name}: smart-rec-duration=0; verify duration behavior.")
        if as_int(kv.get("type")) != 4:
            warn("smart-record", f"{name}: smart-record is usually used with RTSP/live sources.")


def check_sei_and_dewarper(cfg: dict[str, dict[str, str]]) -> None:
    sm = cfg.get("streammux", {})
    sei_keys = [k for k in ("extract-sei-type5-data", "extract-sei-sim-time", "drop-backward-sei") if k in sm]
    for name, kv in source_groups(cfg):
        if "extract-sei-type5-data" in kv or "sei-uuid" in kv:
            sei_keys.append(f"{name}.extract/sei")
    if sei_keys:
        warn("timestamp", f"SEI/simulation timestamp keys used ({sei_keys}); document timestamp semantics and backward-time handling.")

    dew = cfg.get("dewarper")
    if dew and enabled(dew):
        d_surf = as_int(dew.get("num-surfaces-per-frame"))
        sm_surf = as_int(sm.get("num-surfaces-per-frame"))
        if d_surf and sm_surf and d_surf != sm_surf:
            warn("dewarper", f"dewarper.num-surfaces-per-frame={d_surf} but streammux.num-surfaces-per-frame={sm_surf}.")


def check_platform_and_paths(cfg: dict[str, dict[str, str]], cfg_dir: Path) -> None:
    mem4_groups = [name for name, kv in cfg.items() if isinstance(kv, dict) and kv.get("nvbuf-memory-type") == "4"]
    if mem4_groups:
        for name, kv in sink_groups(cfg):
            if kv.get("type") == "2":
                err("platform", f"{name}: type=2 EglSink with Jetson-style nvbuf-memory-type=4; use type=7 nv3dsink or type=5 nvdrmvideosink on Jetson.")

    path_like_suffixes = ("-file", "-config", "config-file", "labelfile-path", "ll-lib-file", "ll-config-file", "output-file", "onnx-file")
    checked: set[str] = set()
    for group, kv in cfg.items():
        for key, value in kv.items():
            if not any(key.endswith(suf) for suf in path_like_suffixes):
                continue
            if not value or value.startswith(("<", "rtsp://", "rtsps://", "file://")):
                continue
            if value in checked:
                continue
            checked.add(value)
            p = Path(value)
            resolved = p if p.is_absolute() else (cfg_dir / p).resolve()
            if not resolved.exists() and "<" not in value:
                warn("paths", f"{group}.{key}: file not found on this machine ({resolved}). Ignore if target host has it.")


def check_msgbroker_and_test5(cfg: dict[str, dict[str, str]], path: Path) -> None:
    has_msg_sink = any(as_int(kv.get("type")) == 6 for _, kv in sink_groups(cfg))
    has_consumer = any(name.startswith("message-consumer") and enabled(kv) for name, kv in cfg.items())
    if has_msg_sink or has_consumer:
        info("test5", "Msgbroker/test5 paths serialize frame_meta->ntp_timestamp; timestamp policy affects downstream event time.")
    if has_consumer and "test5" not in path.stem.lower():
        warn("test5", "message-consumer is configured; run with deepstream-test5-app, not plain deepstream-app.")


def infer_usecases(cfg: dict[str, dict[str, str]], intent: str = "", target_platform: str = "auto") -> set[str]:
    usecases: set[str] = set()
    intent_l = intent.lower()
    sm = cfg.get("streammux") or {}
    td = cfg.get("tiled-display") or {}
    osd = cfg.get("osd") or {}
    sinks = sink_type_names(cfg)

    if has_file_source(cfg):
        usecases.add("filesrc")
    if has_rtsp_source(cfg):
        usecases.add("rtsp")
    if source_list_uses_nvmultiurisrcbin(cfg) and has_rtsp_source(cfg):
        usecases.add("dynamic-rtsp")
    if 3 in sinks:
        usecases.add("recording")
    if 4 in sinks:
        usecases.add("rtsp-out")
    if 6 in sinks or any(name.startswith("message-consumer") and enabled(kv) for name, kv in cfg.items()):
        usecases.add("msgbroker")
        usecases.add("test5")
    if has_display_sink(cfg) or td.get("enable") == "1" or as_bool_str(osd.get("enable"), "0") == "1":
        usecases.add("display")

    latency_sink = any(
        as_int(kv.get("type")) == 1
        and (as_bool_str(kv.get("show-latency"), "0") == "1" or as_bool_str(kv.get("nvdslogger"), "0") == "1")
        for _, kv in sink_groups(cfg)
    )
    latency_mux = (
        as_bool_str(sm.get("live-source"), "0") == "1"
        and as_int(sm.get("batched-push-timeout")) in (33333, 33334)
        and as_bool_str(sm.get("cache-buffer"), "0") == "0"
    )
    if latency_sink or latency_mux or any(word in intent_l for word in ("latency", "low latency", "real time", "realtime")):
        usecases.add("latency")

    perf_shape = (
        1 in sinks
        and td.get("enable", "0") != "1"
        and as_bool_str(osd.get("enable"), "0") != "1"
    )
    perf_intent = any(word in intent_l for word in ("perf", "performance", "throughput", "fps", "benchmark"))
    if perf_intent or (perf_shape and "latency" not in usecases):
        usecases.add("perf")

    if any(word in intent_l for word in ("display", "egl", "show", "monitor", "visualize")):
        usecases.add("display")
    if any(word in intent_l for word in ("record", "recording", "mp4", "file sink", "save")):
        usecases.add("recording")
    if any(word in intent_l for word in ("rtsp out", "rtsp output", "rtsp streaming", "republish")):
        usecases.add("rtsp-out")
    if any(word in intent_l for word in ("kafka", "mqtt", "msgbroker", "message broker", "test5")):
        usecases.add("msgbroker")
        if "test5" in intent_l:
            usecases.add("test5")
    if any(word in intent_l for word in ("dynamic", "add/remove", "rest api", "nvmultiurisrcbin", "sensor-id")):
        if has_rtsp_source(cfg) or source_list_uses_nvmultiurisrcbin(cfg):
            usecases.add("dynamic-rtsp")
    if any(word in intent_l for word in ("jetson", "orin", "nano", "xavier")):
        usecases.add("jetson")
    if any(word in intent_l for word in ("dgpu", "a100", "h100", "rtx", "tesla")):
        usecases.add("dgpu")

    if target_platform in ("jetson", "dgpu"):
        usecases.add(target_platform)
    elif target_platform == "auto":
        if any(kv.get("nvbuf-memory-type") == "4" for kv in cfg.values() if isinstance(kv, dict)):
            usecases.add("jetson")

    return usecases or {"auto"}


def check_usecase_alignment(cfg: dict[str, dict[str, str]], usecases: set[str]) -> None:
    if "auto" in usecases and len(usecases) == 1:
        info("usecase", "Could not infer a specific use case; running generic checks only. Pass --usecase or --intent for intent-aware review.")
        return
    info("usecase", f"Effective use case(s): {', '.join(sorted(usecases))}.")

    sm = cfg.get("streammux") or {}
    td = cfg.get("tiled-display") or {}
    osd = cfg.get("osd") or {}
    src_count = active_source_count(cfg)
    rtsp = has_rtsp_source(cfg)
    filesrc = has_file_source(cfg)
    nvmulti = source_list_uses_nvmultiurisrcbin(cfg)

    if "filesrc" in usecases:
        if not filesrc:
            warn("usecase", "--usecase filesrc was requested but no file/URI source was detected.")
        if rtsp:
            warn("usecase", "--usecase filesrc was requested but RTSP sources are also present.")
        if as_bool_str(sm.get("live-source"), "0") == "1" and "latency" not in usecases:
            warn("filesrc", "File-source use case usually uses streammux.live-source=0 unless intentionally pacing files as live.")
        if "latency" in usecases:
            sat = cfg.get("source-attr-all") or {}
            has_sim_fps = "simulate-fps-interval-ms" in sat
            if not nvmulti or not has_sim_fps:
                warn(
                    "filesrc",
                    "Latency measurement from files should usually use source-list/nvmultiurisrcbin "
                    "with source-attr-all.simulate-fps-interval-ms so files are paced like live input.",
                    "Use [source-list] with `use-nvmultiurisrcbin=1` and add `[source-attr-all] simulate-fps-interval-ms=33` for ~30 fps pacing.",
                )

    if "rtsp" in usecases:
        if not rtsp:
            warn("usecase", "--usecase rtsp was requested but no RTSP source was detected.")
        if as_bool_str(sm.get("live-source"), "0") != "1":
            warn("rtsp", "RTSP use case should normally set streammux.live-source=1.")
        if sm.get("attach-sys-ts-as-ntp") is None:
            warn("rtsp", "RTSP use case should explicitly choose attach-sys-ts-as-ntp=1 or 0 based on timestamp intent.", "Add `attach-sys-ts-as-ntp=1` for latency/local RTSP, or `0` for camera event-time with valid RTCP/NTP.")

    if "dynamic-rtsp" in usecases:
        if not nvmulti:
            err("dynamic-rtsp", "Dynamic RTSP add/remove use case requires [source-list] use-nvmultiurisrcbin=1.", "Convert static [sourceN] RTSP entries to [source-list] + [source-attr-all] and set use-nvmultiurisrcbin=1.")
        sl = cfg.get("source-list") or {}
        for key in ("http-ip", "http-port", "sensor-id-list", "sensor-name-list"):
            if not sl.get(key):
                err("dynamic-rtsp", f"Dynamic RTSP use case requires source-list.{key}.")

    if "latency" in usecases:
        if as_bool_str(sm.get("live-source"), "0") != "1":
            warn("latency", "Latency use case normally expects streammux.live-source=1.")
        bpt = as_int(sm.get("batched-push-timeout"))
        if bpt is None:
            warn("latency", "Latency use case should set batched-push-timeout explicitly, commonly ~33333 for 30 fps.")
        elif bpt > 40000:
            warn("latency", f"batched-push-timeout={bpt} is high for latency use case.")
        for key in ("cache-buffer", "align-first-buffer", "sort-batch"):
            if key in sm and as_bool_str(sm.get(key)) != "0":
                warn("latency", f"Latency use case usually expects streammux.{key}=0.")
        latency_sinks = [name for name, kv in sink_groups(cfg) if as_int(kv.get("type")) == 1]
        if not latency_sinks:
            warn("latency", "Latency use case usually uses a fakesink (type=1) unless validating an output path.")

    if "perf" in usecases:
        if td.get("enable") == "1":
            warn("perf", "Throughput/perf use case usually disables tiled-display to avoid GPU overhead.")
        if as_bool_str(osd.get("enable"), "0") == "1":
            warn("perf", "Throughput/perf use case usually disables OSD.")
        if not any(as_int(kv.get("type")) == 1 for _, kv in sink_groups(cfg)):
            warn("perf", "Throughput/perf use case usually uses fakesink unless output encoding is part of the measurement.")
        if src_count and as_int(sm.get("batch-size")) not in (None, src_count):
            warn("perf", f"Perf use case usually sets streammux.batch-size to source count ({src_count}).")

    if "display" in usecases:
        if td.get("enable") != "1":
            warn("display", "Display use case usually enables tiled-display.")
        if as_bool_str(osd.get("enable"), "0") != "1":
            warn("display", "Display use case usually enables OSD.")
        if not has_display_sink(cfg):
            warn("display", "Display use case should have an EglSink/nv3dsink/nvdrmvideosink sink.")

    if "msgbroker" in usecases or "test5" in usecases:
        if not has_sink_type(cfg, 6):
            warn("msgbroker", "Msgbroker/test5 use case should include a MsgConvBroker sink (type=6) unless using only consumer mode.")
        if "test5" in usecases:
            info("test5", "Run this config with deepstream-test5-app; validate payload timestamps against the chosen NTP policy.")

    if "recording" in usecases and not has_sink_type(cfg, 3):
        warn("recording", "Recording use case should include a file sink (type=3).")

    if "rtsp-out" in usecases and not has_sink_type(cfg, 4):
        warn("rtsp-out", "RTSP output use case should include an RTSPStreaming sink (type=4).")

    if "jetson" in usecases:
        if any(as_int(kv.get("type")) == 2 for _, kv in sink_groups(cfg)):
            err("jetson", "Jetson use case should not use EglSink type=2; use nv3dsink type=7 or nvdrmvideosink type=5.", "Change display sinks from `type=2` to `type=7` (nv3dsink) or `type=5` (nvdrmvideosink) on Jetson.")
        mem4 = any(kv.get("nvbuf-memory-type") == "4" for kv in cfg.values() if isinstance(kv, dict))
        if not mem4:
            warn("jetson", "Jetson use case usually uses nvbuf-memory-type=4 for NVMM surface-array paths.")

    if "dgpu" in usecases:
        mem4_groups = [name for name, kv in cfg.items() if isinstance(kv, dict) and kv.get("nvbuf-memory-type") == "4"]
        if mem4_groups:
            warn("dgpu", f"dGPU use case has Jetson-style nvbuf-memory-type=4 in {mem4_groups}.")


def find_executable(name: str, extra_paths: list[Path] | None = None) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for base in extra_paths or []:
        candidate = base / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def gst_plugin_available(plugin: str, timeout_sec: int) -> bool:
    gst_inspect = shutil.which("gst-inspect-1.0")
    if not gst_inspect:
        return False
    try:
        result = subprocess.run(
            [gst_inspect, plugin],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_gst_pipeline(description: str, timeout_sec: int) -> tuple[bool, str]:
    gst_launch = shutil.which("gst-launch-1.0")
    if not gst_launch:
        return False, "gst-launch-1.0 not found"
    try:
        result = subprocess.run(
            [gst_launch, "-q", *description.split()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_sec}s"
    except OSError as exc:
        return False, str(exc)


def required_plugins(cfg: dict[str, dict[str, str]], usecases: set[str]) -> set[str]:
    plugins = {"nvstreammux"}
    if has_rtsp_source(cfg):
        plugins.add("rtspsrc")
    if has_file_source(cfg) or has_rtsp_source(cfg):
        plugins.add("nvurisrcbin")
    if source_list_uses_nvmultiurisrcbin(cfg):
        plugins.add("nvmultiurisrcbin")
    if (cfg.get("primary-gie") or {}).get("enable", "1") != "0":
        plugins.add("nvinfer")
    if "tracker" in cfg and enabled(cfg["tracker"]):
        plugins.add("nvtracker")
    for _, kv in sink_groups(cfg):
        stype = as_int(kv.get("type"))
        if stype == 2:
            plugins.add("nveglglessink")
        elif stype == 5:
            plugins.add("nvdrmvideosink")
        elif stype == 7:
            plugins.add("nv3dsink")
    if has_sink_type(cfg, 3) or has_sink_type(cfg, 4):
        plugins.update({"nvv4l2h264enc", "h264parse"})
    if has_sink_type(cfg, 4):
        plugins.update({"rtph264pay", "udpsink"})
    if has_sink_type(cfg, 6) or "msgbroker" in usecases:
        plugins.update({"nvmsgconv", "nvmsgbroker"})
    return plugins


def check_runtime_readiness(
    cfg: dict[str, dict[str, str]],
    path: Path,
    usecases: set[str],
    target_platform: str,
    timeout_sec: int,
) -> None:
    ds_bin = Path("/opt/nvidia/deepstream/deepstream/bin")
    if not find_executable("deepstream-app", [ds_bin]):
        warn("runtime", "deepstream-app not found on PATH or /opt/nvidia/deepstream/deepstream/bin.", "Install/source DeepStream or run validation on the target DeepStream host/container.")
    if ("test5" in usecases or "msgbroker" in usecases) and not find_executable("deepstream-test5-app", [ds_bin]):
        warn("runtime", "deepstream-test5-app not found, but test5/msgbroker use case was inferred/requested.", "Run inside a DeepStream image/install that includes deepstream-test5-app.")

    if not shutil.which("gst-inspect-1.0"):
        warn("runtime", "gst-inspect-1.0 not found; cannot verify GStreamer/DeepStream plugin availability.", "Install GStreamer tools or run inside the target DeepStream runtime.")
    else:
        for plugin in sorted(required_plugins(cfg, usecases)):
            if not gst_plugin_available(plugin, timeout_sec):
                warn("runtime", f"GStreamer plugin `{plugin}` was not found by gst-inspect-1.0.", f"Install/enable the DeepStream package providing `{plugin}` or run in the correct container.")

    if target_platform == "dgpu" or ("dgpu" in usecases and "jetson" not in usecases):
        if not shutil.which("nvidia-smi"):
            warn("runtime", "nvidia-smi not found for dGPU-targeted validation.", "Validate on a GPU host/container with NVIDIA drivers mounted.")
        else:
            try:
                result = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_sec, check=False)
                if result.returncode != 0:
                    warn("runtime", "nvidia-smi is present but GPU query failed.", "Check driver/container GPU passthrough before running deepstream-app.")
            except (OSError, subprocess.TimeoutExpired):
                warn("runtime", "nvidia-smi GPU query timed out or failed.", "Check host GPU readiness.")

    if target_platform == "jetson" or "jetson" in usecases:
        if not Path("/etc/nv_tegra_release").exists():
            warn("runtime", "Jetson use case selected but this host does not look like Jetson (`/etc/nv_tegra_release` missing).", "Run runtime validation on the target Jetson device for sink/memory checks.")

    if has_display_sink(cfg):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            warn("runtime", "Display/EGL sink is enabled but DISPLAY/WAYLAND_DISPLAY is not set.", "For headless/SSH/Jenkins, switch to fakesink for validation or run under X/Wayland/Xvfb.")

    for name, kv in sink_groups(cfg):
        if as_int(kv.get("type")) == 3 and kv.get("output-file"):
            out = Path(kv["output-file"])
            if not out.is_absolute():
                out = (path.parent / out).resolve()
            parent = out.parent
            if not parent.exists():
                warn("runtime", f"{name}.output-file directory does not exist: {parent}.", "Create the directory or change output-file to a writable path.")
            elif not os.access(parent, os.W_OK):
                warn("runtime", f"{name}.output-file directory is not writable: {parent}.", "Choose a writable output directory.")

    for name, kv in source_groups(cfg):
        if as_bool_str(kv.get("smart-record"), "0") != "0" and kv.get("smart-rec-dir-path"):
            sr_dir = Path(kv["smart-rec-dir-path"])
            if not sr_dir.is_absolute():
                sr_dir = (path.parent / sr_dir).resolve()
            if not sr_dir.exists():
                warn("runtime", f"{name}.smart-rec-dir-path does not exist: {sr_dir}.", "Create the smart-record directory before running.")
            elif not os.access(sr_dir, os.W_OK):
                warn("runtime", f"{name}.smart-rec-dir-path is not writable: {sr_dir}.", "Use a writable smart-record directory.")


def is_lfs_pointer(path: Path) -> bool:
    try:
        data = path.read_bytes()[:256]
    except OSError:
        return False
    return data.startswith(b"version https://git-lfs.github.com/spec/v1")


def ffprobe_media(path_or_uri: str, timeout_sec: int) -> tuple[bool, str]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return False, "ffprobe not available"
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,duration",
                "-of",
                "json",
                path_or_uri,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"ffprobe timed out after {timeout_sec}s"
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "ffprobe failed").strip()
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return False, "ffprobe returned non-JSON output"
    streams = payload.get("streams") or []
    if not streams:
        return False, "no video stream found"
    stream = streams[0]
    codec = stream.get("codec_name", "unknown")
    width = stream.get("width", "?")
    height = stream.get("height", "?")
    fps = stream.get("avg_frame_rate", "?")
    duration = stream.get("duration", "?")
    return True, f"codec={codec}, resolution={width}x{height}, fps={fps}, duration={duration}s"


class _HttpOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects that leave the HTTP(S) schemes allowed by the probe."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urljoin(req.full_url, newurl)
        if urlparse(target).scheme.lower() not in ("http", "https"):
            raise urllib.error.HTTPError(
                target,
                code,
                "HTTP source redirected to a non-http(s) URI",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, target)


def probe_http_uri(name: str, uri: str, timeout_sec: int) -> None:
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        err(
            "source-probe",
            f"{name}: HTTP probe rejected non-http(s) URI scheme {parsed.scheme!r}.",
            "Use http:// or https:// for HTTP source probes.",
        )
        return

    opener = urllib.request.build_opener(_HttpOnlyRedirectHandler())

    def _open(method: str):
        request = urllib.request.Request(uri, method=method)
        return opener.open(request, timeout=timeout_sec)

    try:
        try:
            with _open("HEAD") as response:
                status = getattr(response, "status", 0)
                ctype = response.headers.get("Content-Type", "unknown")
                clen = response.headers.get("Content-Length", "unknown")
        except Exception as head_exc:
            # Many media servers reject HEAD (405/501); fall back to bounded GET.
            try:
                with _open("GET") as response:
                    status = getattr(response, "status", 0)
                    ctype = response.headers.get("Content-Type", "unknown")
                    clen = response.headers.get("Content-Length", "unknown")
                    # Drain at most a small prefix so we do not pull full media.
                    response.read(1024)
            except Exception:
                raise head_exc from None
        if status >= 400:
            err("source-probe", f"{name}: HTTP source returned status {status}: {uri}.", "Fix the URL or server-side access before running DeepStream.")
            return
        info("source-probe", f"{name}: HTTP source reachable (status={status}, content-type={ctype}, content-length={clen}).")
    except Exception as exc:
        err("source-probe", f"{name}: HTTP source is not reachable ({exc}).", "Fix the URL, credentials, proxy, VPN, or server availability.")
        return
    ok, detail = ffprobe_media(uri, timeout_sec)
    if ok:
        info("source-probe", f"{name}: HTTP media inspection passed ({detail}).")
    else:
        warn("source-probe", f"{name}: HTTP source reachable but media inspection failed ({detail}).", "Verify the URL points directly to a supported media stream/file.")


def probe_file_source(name: str, uri: str, cfg_dir: Path, timeout_sec: int) -> None:
    if uri.startswith(("http://", "https://")):
        probe_http_uri(name, uri, timeout_sec)
        return
    p = resolve_file_uri(uri, cfg_dir)
    if p is None:
        return
    if not uri.startswith("file://") and "://" in uri:
        warn("source-probe", f"{name}: unsupported URI scheme in source {uri!r}.", "Use file://, rtsp://, rtsps://, http://, or https:// sources supported by the target DeepStream source bin.")
        return
    if not p.exists():
        err("source-probe", f"{name}: file source does not exist: {p}.", "Fix the URI/path or copy the media file to the target host.")
        return
    if not p.is_file():
        err("source-probe", f"{name}: file source is not a regular file: {p}.", "Point the source URI to a real media file.")
        return
    if not os.access(p, os.R_OK):
        err("source-probe", f"{name}: file source is not readable: {p}.", "Fix file permissions or copy the media to a readable location.")
        return
    if p.stat().st_size == 0:
        err("source-probe", f"{name}: file source is empty: {p}.", "Replace the file with a valid media asset.")
        return
    if is_lfs_pointer(p):
        err("source-probe", f"{name}: file source is a Git LFS pointer, not media: {p}.", "Run `git lfs pull` or replace the file with a real media asset before running deepstream-app.")
        return
    discoverer = shutil.which("gst-discoverer-1.0")
    ffprobe = shutil.which("ffprobe")
    discoverer_detail = ""
    if discoverer:
        try:
            result = subprocess.run(
                [discoverer, str(p)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            if result.returncode == 0:
                info("source-probe", f"{name}: file media inspection passed with gst-discoverer.")
                return
            discoverer_detail = (result.stderr or "gst-discoverer failed").strip()
        except subprocess.TimeoutExpired:
            discoverer_detail = f"gst-discoverer timed out after {timeout_sec}s"
    if ffprobe:
        ok, detail = ffprobe_media(str(p), timeout_sec)
        if ok:
            info("source-probe", f"{name}: file media inspection passed ({detail}).")
            return
        warn("source-probe", f"{name}: media inspection failed for {p} (gst-discoverer: {discoverer_detail or 'not run'}; ffprobe: {detail}).", "Verify the file is a valid supported media stream.")
    elif discoverer_detail:
        warn("source-probe", f"{name}: gst-discoverer could not inspect media file {p} ({discoverer_detail}).", "Verify the file is a valid media stream supported by GStreamer or install ffprobe for fallback inspection.")
    else:
        info("source-probe", f"{name}: file exists, but gst-discoverer-1.0/ffprobe not available for media inspection.")


def probe_rtsp_source(name: str, uri: str, timeout_sec: int) -> None:
    parsed = urlparse(uri)
    if not parsed.hostname:
        err("source-probe", f"{name}: RTSP URI has no hostname: {uri}.", "Use a valid rtsp://host:port/path URI.")
        return
    port = parsed.port or (322 if parsed.scheme == "rtsps" else 554)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout_sec):
            info("source-probe", f"{name}: TCP connection to {parsed.hostname}:{port} succeeded.")
    except OSError as exc:
        err("source-probe", f"{name}: cannot connect to RTSP endpoint {parsed.hostname}:{port} ({exc}).", "Start the RTSP server/camera, verify VPN/network/firewall, or fix the URI/port.")
        return
    discoverer = shutil.which("gst-discoverer-1.0")
    if discoverer:
        try:
            result = subprocess.run([discoverer, uri], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_sec, check=False)
            if result.returncode != 0:
                warn("source-probe", f"{name}: TCP is reachable but gst-discoverer failed for RTSP URI.", "The server may require credentials, use unsupported codecs, or not be streaming yet.")
        except subprocess.TimeoutExpired:
            warn("source-probe", f"{name}: gst-discoverer timed out for RTSP URI.", "Increase --timeout-sec or validate the RTSP stream manually.")


def check_source_health(cfg: dict[str, dict[str, str]], cfg_dir: Path, timeout_sec: int) -> None:
    for name, uri in iter_source_uris(cfg):
        if uri.startswith(("rtsp://", "rtsps://")):
            probe_rtsp_source(name, uri, timeout_sec)
        else:
            probe_file_source(name, uri, cfg_dir, timeout_sec)

    for name, kv in source_groups(cfg):
        stype = as_int(kv.get("type"))
        if stype == 1 and kv.get("camera-v4l2-dev-node"):
            raw_dev = kv["camera-v4l2-dev-node"].strip()
            if not re.fullmatch(r"/dev/video\d+", raw_dev):
                err(
                    "source-probe",
                    f"{name}: camera-v4l2-dev-node must match /dev/videoN, got {raw_dev!r}.",
                    "Set camera-v4l2-dev-node to a V4L2 node such as /dev/video0.",
                )
                continue
            dev = Path(raw_dev)
            if not dev.exists():
                err("source-probe", f"{name}: V4L2 device does not exist: {dev}.", "Connect the camera or update camera-v4l2-dev-node.")
            elif not os.access(dev, os.R_OK):
                err("source-probe", f"{name}: V4L2 device is not readable: {dev}.", "Fix device permissions or run on the target camera host.")
            else:
                v4l2_ctl = shutil.which("v4l2-ctl")
                if v4l2_ctl:
                    try:
                        result = subprocess.run(
                            [v4l2_ctl, "--device", str(dev), "--all"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=timeout_sec,
                            check=False,
                        )
                        if result.returncode == 0:
                            info("source-probe", f"{name}: V4L2 device responded to v4l2-ctl.")
                        else:
                            warn("source-probe", f"{name}: v4l2-ctl could not query {dev} ({(result.stderr or result.stdout).strip()[:300]}).", "Verify the camera is usable by V4L2 on the target host.")
                    except subprocess.TimeoutExpired:
                        warn("source-probe", f"{name}: v4l2-ctl timed out for {dev}.", "Check whether the camera device is stuck or busy.")
                else:
                    info("source-probe", f"{name}: V4L2 device exists; install v4l2-ctl for capability probing.")
        elif stype == 1:
            warn("source-probe", f"{name}: V4L2 source has no camera-v4l2-dev-node to probe.", "Set camera-v4l2-dev-node for V4L2 camera validation.")
        if stype == 5 and not Path("/etc/nv_tegra_release").exists():
            warn("source-probe", f"{name}: CSI source requires Jetson runtime; this host does not look like Jetson.", "Validate CSI configs on the target Jetson device.")


def probe_sink_functionality(cfg: dict[str, dict[str, str]], cfg_dir: Path, timeout_sec: int) -> None:
    for name, kv in sink_groups(cfg):
        stype = as_int(kv.get("type"))
        if stype == 1:
            ok, detail = run_gst_pipeline("videotestsrc num-buffers=5 ! fakesink sync=false", timeout_sec)
            if ok:
                info("sink-probe", f"{name}: fakesink functional smoke test passed.")
            else:
                err("sink-probe", f"{name}: fakesink smoke test failed ({detail[:300]}).", "Fix the local GStreamer runtime before running DeepStream.")
        elif stype == 2:
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                err("sink-probe", f"{name}: EglSink cannot be functionally tested because no DISPLAY/WAYLAND_DISPLAY is set.", "Run with X/Wayland/Xvfb or change this sink to fakesink for headless validation.")
                continue
            ok, detail = run_gst_pipeline("videotestsrc num-buffers=5 ! videoconvert ! nveglglessink sync=false", timeout_sec)
            if ok:
                info("sink-probe", f"{name}: EglSink functional smoke test passed.")
            else:
                err("sink-probe", f"{name}: EglSink smoke test failed ({detail[:300]}).", "Check display access, EGL/GPU drivers, or use fakesink on headless hosts.")
        elif stype == 3:
            out = Path(kv.get("output-file") or "validator_sink_probe.mp4")
            if not out.is_absolute():
                out = (cfg_dir / out).resolve()
            parent = out.parent
            if not parent.exists() or not os.access(parent, os.W_OK):
                err("sink-probe", f"{name}: file sink output directory is not writable: {parent}.", "Create a writable output directory or change output-file.")
                continue
            probe_out = parent / ".ds_validator_sink_probe.mp4"
            pipeline = f"videotestsrc num-buffers=5 ! videoconvert ! x264enc tune=zerolatency ! mp4mux ! filesink location={probe_out}"
            ok, detail = run_gst_pipeline(pipeline, timeout_sec)
            try:
                probe_out.unlink(missing_ok=True)
            except OSError:
                pass
            if ok:
                info("sink-probe", f"{name}: file sink encode/write smoke test passed.")
            else:
                warn("sink-probe", f"{name}: file sink smoke test failed ({detail[:300]}).", "Verify encoder/muxer plugins and output directory; DeepStream hardware encoder may still work if x264enc is unavailable.")
        elif stype == 4:
            # Avoid binding the config's actual ports. This checks encoder/payloader/UDP sink construction only.
            pipeline = "videotestsrc num-buffers=5 ! videoconvert ! x264enc tune=zerolatency ! h264parse ! rtph264pay ! udpsink host=127.0.0.1 port=50000 sync=false"
            ok, detail = run_gst_pipeline(pipeline, timeout_sec)
            if ok:
                info("sink-probe", f"{name}: RTSP/UDP output encoder/payloader smoke test passed.")
            else:
                warn("sink-probe", f"{name}: RTSP/UDP output smoke test failed ({detail[:300]}).", "Verify encoder, h264parse, rtph264pay, and network/port availability on the target host.")
        elif stype == 6:
            proto = kv.get("msg-broker-proto-lib")
            conv_cfg = kv.get("msg-conv-config")
            missing = []
            for label, value in (("msg-broker-proto-lib", proto), ("msg-conv-config", conv_cfg)):
                if not value:
                    missing.append(label)
                    continue
                p = Path(value)
                if not p.is_absolute():
                    p = (cfg_dir / p).resolve()
                if not p.exists():
                    missing.append(f"{label} not found: {p}")
            if missing:
                err("sink-probe", f"{name}: msgbroker sink prerequisites missing ({'; '.join(missing)}).", "Fix msgbroker library/config paths before running deepstream-app/test5.")
            else:
                info("sink-probe", f"{name}: msgbroker library/config files exist. Broker network connectivity is not tested by sink probe.")
        elif stype in (5, 7):
            if not Path("/etc/nv_tegra_release").exists():
                warn("sink-probe", f"{name}: Jetson display sink type={stype} cannot be functionally tested on this non-Jetson host.", "Run sink probe on the target Jetson device.")
            else:
                info("sink-probe", f"{name}: Jetson display sink detected; run on target Jetson for functional display validation.")


def group_file_path(cfg_dir: Path, value: str | None) -> Path | None:
    if not value or value.startswith(("<", "rtsp://", "rtsps://", "file://")):
        return None
    p = Path(value)
    return p if p.is_absolute() else (cfg_dir / p).resolve()


def probe_group_prerequisites(cfg: dict[str, dict[str, str]], cfg_dir: Path) -> None:
    """Group-level checks for active components that cannot be safely isolated.

    The exact full config is exercised by --probe-pipeline. These checks make
    missing per-group prerequisites obvious before that smoke run.
    """
    for name, kv in gie_groups(cfg):
        plugin_type = kv.get("plugin-type", "0")
        plugin = "nvinferserver" if plugin_type == "1" else "nvinfer"
        if gst_plugin_available(plugin, 3):
            info("group-probe", f"{name}: required inference plugin `{plugin}` is available.")
        else:
            warn("group-probe", f"{name}: required inference plugin `{plugin}` is not available.", f"Install/enable `{plugin}` in the target DeepStream runtime.")
        for key in ("config-file", "model-engine-file", "onnx-file", "labelfile-path"):
            p = group_file_path(cfg_dir, kv.get(key))
            if p and not p.exists():
                warn("group-probe", f"{name}.{key} not found on this host: {p}.", "Fix the path or validate on the target host/container where the file exists.")

    tr = cfg.get("tracker")
    if tr and enabled(tr):
        if gst_plugin_available("nvtracker", 3):
            info("group-probe", "tracker: `nvtracker` plugin is available.")
        else:
            warn("group-probe", "tracker: `nvtracker` plugin is not available.", "Install/enable nvtracker in the target DeepStream runtime.")
        for key in ("ll-lib-file", "ll-config-file"):
            p = group_file_path(cfg_dir, tr.get(key))
            if p and not p.exists():
                warn("group-probe", f"tracker.{key} not found on this host: {p}.", "Fix the tracker path or validate on the target host/container.")

    for group_name in ("pre-process", "secondary-pre-process", "message-converter", "img-save", "dewarper", "nvds-analytics", "ds-example"):
        kv = cfg.get(group_name)
        if kv and enabled(kv):
            p = group_file_path(cfg_dir, kv.get("config-file"))
            if p and not p.exists():
                warn("group-probe", f"{group_name}.config-file not found on this host: {p}.", "Fix the config path or validate on the target host/container.")
            else:
                info("group-probe", f"{group_name}: active group prerequisites look present.")

    for name, kv in sink_groups(cfg):
        if as_int(kv.get("type")) == 6:
            for key in ("msg-broker-proto-lib", "msg-conv-config"):
                p = group_file_path(cfg_dir, kv.get(key))
                if p and not p.exists():
                    warn("group-probe", f"{name}.{key} not found on this host: {p}.", "Fix msgbroker paths or validate on the target host/container.")


def has_message_consumer(cfg: dict[str, dict[str, str]]) -> bool:
    return any(name.startswith("message-consumer") and enabled(kv) for name, kv in cfg.items())


PIPELINE_PROBE_SIDE_EFFECTS = (
    "GPU/inference pipeline execution",
    "possible output file writes from sinks",
    "outbound network to RTSP/HTTP sources and broker sinks",
    "messages to Kafka/MQTT/msgbroker if configured",
)


def print_pipeline_probe_warning(app_name: str, path: Path, timeout_sec: int) -> None:
    print(
        "PIPELINE-PROBE WARNING: about to run a bounded smoke test with "
        f"`{app_name} -c {path}` (timeout={timeout_sec}s). Side effects may include: "
        + "; ".join(PIPELINE_PROBE_SIDE_EFFECTS)
        + ". This is an opt-in runtime probe, not nsys profiling.",
        file=sys.stderr,
    )


def probe_full_pipeline(path: Path, cfg: dict[str, dict[str, str]], usecases: set[str], timeout_sec: int) -> None:
    ds_bin = Path("/opt/nvidia/deepstream/deepstream/bin")
    app_name = "deepstream-test5-app" if "test5" in usecases or has_message_consumer(cfg) else "deepstream-app"
    app = find_executable(app_name, [ds_bin])
    if not app:
        err("pipeline-probe", f"{app_name} not found; cannot run exact-config pipeline smoke test.", f"Install/source DeepStream or run probe in the target container that provides `{app_name}`.")
        return
    print_pipeline_probe_warning(app_name, path, timeout_sec)
    try:
        result = subprocess.run(
            [app, "-c", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
            cwd=str(path.parent),
        )
        output = result.stdout or ""
        markers = fatal_markers(output)
        if result.returncode == 0 and not markers:
            info("pipeline-probe", f"{app_name} exited successfully with the exact config.")
        elif markers:
            err("pipeline-probe", f"{app_name} reported fatal/error lines during exact-config smoke run: {' | '.join(markers)}", "Fix the reported runtime errors, then re-run validation.")
        else:
            err("pipeline-probe", f"{app_name} exited with code {result.returncode} during exact-config smoke run.", "Inspect DeepStream output and fix the failing group/source/sink.")
    except subprocess.TimeoutExpired as exc:
        output = ""
        if exc.stdout:
            output += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(errors="replace")
        if exc.stderr:
            output += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(errors="replace")
        markers = fatal_markers(output)
        if markers:
            err("pipeline-probe", f"{app_name} produced fatal/error lines before timeout: {' | '.join(markers)}", "Fix the reported runtime errors, then re-run validation.")
        else:
            info("pipeline-probe", f"{app_name} ran for {timeout_sec}s with the exact config and no fatal errors were detected.")


def ntp_probe_script_path() -> Path:
    return Path(__file__).resolve().parent / "ntp_probe.py"


def probe_rtsp_ntp_source(name: str, uri: str, timeout_sec: int) -> None:
    if not uri.startswith(("rtsp://", "rtsps://")):
        return
    script = ntp_probe_script_path()
    if not script.is_file():
        warn(
            "ntp-probe",
            f"{name}: NTP probe module missing at {script}.",
            "Restore scripts/ntp_probe.py next to validate_config.py.",
        )
        return
    try:
        # Pass URI on stdin so credentialed RTSP URLs are not visible in argv.
        result = subprocess.run(
            [sys.executable, str(script), str(timeout_sec)],
            input=uri + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec + 5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        warn("ntp-probe", f"{name}: NTP probe timed out after {timeout_sec}s.", "Increase --timeout-sec or validate the RTSP stream manually on the target host.")
        return

    raw = (result.stdout or "").strip().splitlines()
    payload: dict[str, Any] | None = None
    if raw:
        try:
            payload = json.loads(raw[-1])
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        warn("ntp-probe", f"{name}: NTP probe did not return structured output. stderr={result.stderr.strip()[:300]!r}", "Run inside a DeepStream Python environment with GStreamer GI and pyds available.")
        return

    status = payload.get("status")
    if status == "ok":
        info(
            "ntp-probe",
            f"{name}: RTCP Sender Report/NTP sync observed (ntp-time-epoch-ns={payload.get('ntp_time_epoch_ns')}, frame-timestamp={payload.get('frame_timestamp')}).",
        )
    elif status in ("dependency-error", "pyds-error"):
        warn("ntp-probe", f"{name}: cannot run DeepStream NTP probe ({payload.get('error')}).", "Run this probe in the target DeepStream Python environment where `gi` and `pyds` are available.")
    elif status == "pipeline-error":
        err("ntp-probe", f"{name}: failed to create/run RTSP NTP probe pipeline ({payload.get('error')}).", "Verify RTSP URI, codec support, and GStreamer plugin availability.")
    else:
        detail = "; ".join(payload.get("errors") or []) or "no non-zero nvds-ntp-sync values observed"
        warn(
            "ntp-probe",
            f"{name}: RTSP was probed but no valid RTCP Sender Report/NTP sync was observed within {payload.get('elapsed_sec', timeout_sec)}s ({detail}).",
            "If camera event-time is required, configure the camera/server to send RTCP Sender Reports or keep `attach-sys-ts-as-ntp=1` for system-time timestamps.",
        )


def check_rtsp_ntp_health(cfg: dict[str, dict[str, str]], timeout_sec: int) -> None:
    rtsp_uris = [(name, uri) for name, uri in iter_source_uris(cfg) if uri.startswith(("rtsp://", "rtsps://"))]
    if not rtsp_uris:
        warn("ntp-probe", "--probe-ntp requested but no RTSP URI was found in the config.")
        return
    for name, uri in rtsp_uris:
        probe_rtsp_ntp_source(name, uri, timeout_sec)


def run_checks(
    cfg: dict[str, dict[str, str]],
    path: Path,
    usecases: set[str],
    *,
    check_runtime: bool = True,
    probe_sources: bool = False,
    probe_rtsp: bool = False,
    probe_sinks: bool = False,
    probe_groups: bool = False,
    probe_pipeline: bool = False,
    probe_ntp: bool = False,
    target_platform: str = "auto",
    timeout_sec: int = 5,
) -> None:
    check_usecase_alignment(cfg, usecases)
    check_source_contract(cfg)
    check_source_list(cfg)
    check_rtsp_policy(cfg)
    check_rtsp_stability(cfg)
    check_streammux(cfg)
    check_batch_and_gie(cfg)
    check_preprocess(cfg)
    check_decode_pressure(cfg)
    check_sinks(cfg)
    check_demux(cfg)
    check_tracker(cfg)
    check_smart_record(cfg)
    check_sei_and_dewarper(cfg)
    check_platform_and_paths(cfg, path.parent)
    check_msgbroker_and_test5(cfg, path)
    if check_runtime:
        check_runtime_readiness(cfg, path, usecases, target_platform, timeout_sec)
    if probe_sources or probe_rtsp:
        check_source_health(cfg, path.parent, timeout_sec)
    if probe_sinks:
        probe_sink_functionality(cfg, path.parent, timeout_sec)
    if probe_groups:
        probe_group_prerequisites(cfg, path.parent)
    if probe_pipeline:
        probe_full_pipeline(path, cfg, usecases, timeout_sec)
    if probe_ntp or probe_rtsp:
        check_rtsp_ntp_health(cfg, timeout_sec)


def print_findings(json_output: bool = False, usecases: set[str] | None = None) -> int:
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    counts = {"ERROR": 0, "WARN": 0, "INFO": 0}
    for finding in FINDINGS:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    sorted_findings = sorted(FINDINGS, key=lambda f: (order.get(f.severity, 99), f.area, f.message))
    for finding in sorted_findings:
        if finding.severity in ("ERROR", "WARN") and not finding.suggestion:
            finding.suggestion = default_suggestion(finding)
    if json_output:
        print(json.dumps({
            "summary": {
                "errors": counts.get("ERROR", 0),
                "warnings": counts.get("WARN", 0),
                "info": counts.get("INFO", 0),
                "usecases": sorted(usecases or []),
            },
            "findings": [dataclasses.asdict(f) for f in sorted_findings],
        }, indent=2))
    else:
        for finding in sorted_findings:
            print(f"{finding.severity} [{finding.area}]: {finding.message}")
            if finding.suggestion:
                print(f"SUGGEST [{finding.area}]: {finding.suggestion}")
        print(
            f"SUMMARY: {counts.get('ERROR', 0)} error(s), "
            f"{counts.get('WARN', 0)} warning(s), {counts.get('INFO', 0)} info note(s)"
        )
    return 1 if counts.get("ERROR", 0) else 0


def default_suggestion(finding: Finding) -> str:
    area_defaults = {
        "parse": "Fix the config syntax or install the missing parser dependency, then re-run validation.",
        "source": "Fix the source type/URI/device fields so they match the intended input source, then re-run validation.",
        "timestamp": "Choose and set an explicit timestamp policy that matches the use case.",
        "rtsp": "Add the recommended RTSP source settings or adjust the use case, then re-run validation.",
        "streammux": "Adjust streammux settings to match the active sources and use case.",
        "gie": "Fix the GIE configuration, batch size, unique IDs, or upstream references, then re-run validation.",
        "preprocess": "Align preprocess and input-tensor-meta settings, then re-run validation.",
        "source-list": "Fix source-list/source-attr-all consistency, then re-run validation.",
        "decode": "Review frame-dropping/queue/decode settings against latency versus accuracy requirements.",
        "sink": "Fix enabled sink required keys or change the sink type for the target environment.",
        "demux": "Fix tiled-display/demux/source-id sink mapping, then re-run validation.",
        "tracker": "Fix tracker/source compatibility or tracker configuration, then re-run validation.",
        "smart-record": "Fix smart-record paths/cache/container/duration settings, then re-run validation.",
        "dewarper": "Align dewarper and streammux multi-surface settings.",
        "platform": "Choose sink and memory settings that match the target platform.",
        "paths": "Fix the path if it is required on this host; ignore only if it is intentionally target-host specific or disabled.",
        "test5": "Run with the correct DeepStream app variant and verify metadata/timestamp behavior.",
        "usecase": "Clarify the intended use case or pass the correct intent/usecase to validation.",
        "filesrc": "Adjust file-source settings for the requested use case.",
        "latency": "Apply the latency profile choices needed for this config, then re-run validation.",
        "perf": "Apply throughput/perf-oriented settings, then re-run validation.",
        "display": "Use a display sink for display use cases or change the intent to headless/fakesink validation.",
        "msgbroker": "Add or fix msgbroker/test5 settings, then re-run validation.",
        "recording": "Add a file sink or change the intended use case.",
        "rtsp-out": "Add an RTSPStreaming sink or change the intended use case.",
        "jetson": "Use Jetson-compatible sink/memory settings.",
        "dgpu": "Use dGPU-compatible sink/memory settings.",
        "runtime": "Install/enable the missing runtime prerequisite or run validation on the target host/container.",
        "source-probe": "Fix source reachability/media/device availability, then re-run validation.",
        "sink-probe": "Fix the sink runtime prerequisite or switch to a sink supported by the current host/container.",
        "group-probe": "Fix the active group prerequisite or validate on the target host/container.",
        "pipeline-probe": "Fix the reported exact-config runtime failure, then re-run validation.",
        "ntp-probe": "Fix RTSP/NTP runtime prerequisites or validate on the target camera host/network.",
    }
    return area_defaults.get(finding.area, "Fix this finding based on the message above, then re-run validation.")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit/validate DeepStream deepstream-app / deepstream-test5 configs. "
            "Default mode is offline/static parsing. Opt-in --check-runtime and "
            "--probe-* flags may run "
            "subprocess tools (gst-inspect, gst-launch, nvidia-smi, v4l2-ctl, ffprobe, "
            "deepstream-app/test5), open RTSP/HTTP network connections, write probe "
            "output, and (with --probe-pipeline --confirm-pipeline-probe) execute a "
            "bounded full-app smoke run. Not for nsys profiling."
        )
    )
    parser.add_argument("config", help="Path to deepstream-app/deepstream-test5 config (.txt/.yml/.yaml)")
    parser.add_argument(
        "--usecase",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Use-case intent for stricter review. Can be repeated or comma-separated. "
            "Valid: " + ", ".join(USECASE_CHOICES)
        ),
    )
    parser.add_argument(
        "--intent",
        default="",
        help="Natural-language user intent/conversation context (for example: 'generate low-latency RTSP test5 config for Kafka').",
    )
    parser.add_argument(
        "--target-platform",
        choices=("auto", "jetson", "dgpu"),
        default="auto",
        help="Target runtime platform. Defaults to auto/inferred hints.",
    )
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help=(
            "Opt-in: inspect target-host runtime readiness with bounded local "
            "subprocesses such as gst-inspect-1.0 and nvidia-smi."
        ),
    )
    parser.add_argument(
        "--probe-sources",
        action="store_true",
        help=(
            "Opt-in: probe file/RTSP/HTTP/V4L2 sources with bounded checks (may open "
            "outbound network sockets and run ffprobe/gst-discoverer). Prefer after "
            "explicit operator approval."
        ),
    )
    parser.add_argument(
        "--probe-rtsp",
        action="store_true",
        help=(
            "Opt-in: for RTSP sources, run reachability/media checks plus RTCP Sender "
            "Report/NTP checks via scripts/ntp_probe.py (outbound RTSP)."
        ),
    )
    parser.add_argument(
        "--probe-sinks",
        action="store_true",
        help=(
            "Opt-in: bounded functional smoke tests for enabled sink types on the "
            "current host/container (may invoke gst-launch / display helpers)."
        ),
    )
    parser.add_argument(
        "--probe-groups",
        action="store_true",
        help=(
            "Opt-in: active group prerequisite checks for GIE/tracker/preprocess/"
            "msgbroker/etc (may call gst-inspect-1.0)."
        ),
    )
    parser.add_argument(
        "--probe-pipeline",
        action="store_true",
        help=(
            "Opt-in bounded deepstream-app/deepstream-test5-app smoke test using the exact "
            "supplied config. Side effects may include GPU/inference, file writes, RTSP/HTTP "
            "network I/O, and broker messages. Requires --confirm-pipeline-probe. Not nsys profiling."
        ),
    )
    parser.add_argument(
        "--confirm-pipeline-probe",
        action="store_true",
        help=(
            "Required confirmation that the operator approved --probe-pipeline side effects "
            "(GPU/inference, possible file writes, outbound network, broker messages)."
        ),
    )
    parser.add_argument(
        "--probe-ntp",
        action="store_true",
        help=(
            "For RTSP sources, run scripts/ntp_probe.py (bounded DeepStream/GStreamer probe) "
            "to verify RTCP Sender Report based NTP sync. Opens outbound RTSP connections."
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=5,
        help="Timeout for runtime/source probes (default: 5 seconds).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON findings instead of human-readable text.",
    )
    args = parser.parse_args(argv)
    try:
        explicit_usecases = normalize_usecases(args.usecase)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.timeout_sec < 1:
        parser.error("--timeout-sec must be >= 1")
    if args.probe_pipeline and not args.confirm_pipeline_probe:
        parser.error(
            "--probe-pipeline requires --confirm-pipeline-probe after explicit operator approval "
            "(side effects: GPU/inference, file writes, network I/O, broker messages)"
        )
    if args.confirm_pipeline_probe and not args.probe_pipeline:
        parser.error("--confirm-pipeline-probe is only valid together with --probe-pipeline")

    path = Path(args.config).expanduser().resolve()
    if not path.is_file():
        print(f"config file not found: {path}", file=sys.stderr)
        return 2
    cfg = load_config(path)
    if not cfg:
        if not any(f.severity == "ERROR" for f in FINDINGS):
            err("parse", "No DeepStream config groups found.")
        return print_findings(args.json)
    inferred_usecases = infer_usecases(cfg, args.intent, args.target_platform)
    if explicit_usecases == {"auto"}:
        usecases = inferred_usecases
        if inferred_usecases != {"auto"}:
            info("usecase", f"Inferred use case(s): {', '.join(sorted(inferred_usecases))}.")
    else:
        usecases = (explicit_usecases - {"auto"}) | (inferred_usecases - {"auto"})
        info("usecase", f"Explicit use case(s): {', '.join(sorted(explicit_usecases - {'auto'}))}; inferred additions: {', '.join(sorted(inferred_usecases - explicit_usecases - {'auto'})) or 'none'}.")
    runtime_requested = any(
        (
            args.check_runtime,
            args.probe_sources,
            args.probe_rtsp,
            args.probe_sinks,
            args.probe_groups,
            args.probe_pipeline,
            args.probe_ntp,
        )
    )
    run_checks(
        cfg,
        path,
        usecases,
        check_runtime=runtime_requested,
        probe_sources=args.probe_sources,
        probe_rtsp=args.probe_rtsp,
        probe_sinks=args.probe_sinks,
        probe_groups=args.probe_groups,
        probe_pipeline=args.probe_pipeline,
        probe_ntp=args.probe_ntp,
        target_platform=args.target_platform,
        timeout_sec=args.timeout_sec,
    )
    return print_findings(args.json, usecases)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
