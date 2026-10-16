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

"""Bounded RTSP NTP / RTCP Sender Report probe for deepstream-audit-config.

Invoked by validate_config.py as a sibling module (not via python -c) so the
probe logic stays auditable. Requires GStreamer GI bindings and pyds in the
target DeepStream Python environment.

Usage:
  echo '<rtsp-uri>' | python3 ntp_probe.py <timeout-sec>

Arguments:
  timeout-sec  Positive integer seconds for the bounded probe.

Output:
  One JSON object on stdout with status / NTP fields.

Exit codes:
  0  Handled result (including dependency/pipeline errors reported in JSON)
  1  Unhandled Python/runtime failure
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any
from urllib.parse import urlparse


def run_ntp_probe(uri: str, timeout_sec: int) -> dict[str, Any]:
    """Probe one RTSP URI for nvds-ntp-sync / RTCP Sender Report values."""
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst, GLib
        import pyds
    except Exception as exc:
        return {"status": "dependency-error", "error": str(exc)}

    Gst.init(None)
    state: dict[str, Any] = {
        "epoch": 0,
        "pts": 0,
        "queries": 0,
        "buffers": 0,
        "errors": [],
    }

    pipeline = Gst.Pipeline.new("ntp-probe")
    src = Gst.ElementFactory.make("uridecodebin", "src")
    queue = Gst.ElementFactory.make("queue", "queue")
    conv = Gst.ElementFactory.make("videoconvert", "conv")
    sink = Gst.ElementFactory.make("fakesink", "sink")

    if not pipeline or not src or not queue or not conv or not sink:
        return {
            "status": "pipeline-error",
            "error": "failed to create uridecodebin/queue/videoconvert/fakesink",
        }

    src.set_property("uri", uri)
    sink.set_property("sync", False)
    sink.set_property("async", False)

    pipeline.add(src)
    pipeline.add(queue)
    pipeline.add(conv)
    pipeline.add(sink)
    queue.link(conv)
    conv.link(sink)

    try:
        pyds.configure_source_for_ntp_sync(hash(src))
    except Exception as exc:
        return {"status": "pyds-error", "error": str(exc)}

    def on_pad_added(_src, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.is_empty():
            return
        structure = caps.get_structure(0)
        if structure is None:
            return
        media = structure.get_name() or ""
        if not media.startswith("video/"):
            return
        sinkpad = queue.get_static_pad("sink")
        if sinkpad and not sinkpad.is_linked():
            pad.link(sinkpad)

    src.connect("pad-added", on_pad_added)

    def on_buffer(pad, info):
        state["buffers"] += 1
        return Gst.PadProbeReturn.OK

    loop = GLib.MainLoop()

    def query_once():
        pad = conv.get_static_pad("sink")
        if not pad:
            return True
        struct = Gst.Structure.new_empty("nvds-ntp-sync")
        query = Gst.Query.new_custom(Gst.QueryType.CUSTOM, struct)
        state["queries"] += 1
        try:
            ok = pad.query(query)
        except Exception as exc:
            state["errors"].append(f"query failed: {exc}")
            return True
        if ok:
            out = query.get_structure()
            ok_epoch, epoch = out.get_uint64("ntp-time-epoch-ns")
            ok_pts, pts = out.get_uint64("frame-timestamp")
            if ok_epoch:
                state["epoch"] = int(epoch)
            if ok_pts:
                state["pts"] = int(pts)
            if state["epoch"] > 0 and state["pts"] > 0:
                loop.quit()
                return False
        return True

    queue.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_buffer)
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(_bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            state["errors"].append(f"{err}; {dbg or ''}")
            loop.quit()
        elif msg.type == Gst.MessageType.EOS:
            state["errors"].append("EOS before NTP sync observed")
            loop.quit()

    bus.connect("message", on_message)
    GLib.timeout_add(1000, query_once)
    GLib.timeout_add_seconds(timeout_sec, lambda: (loop.quit() or False))

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        return {"status": "pipeline-error", "error": "failed to set pipeline PLAYING"}

    started = time.time()
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

    status = "ok" if state["epoch"] > 0 and state["pts"] > 0 else "no-ntp"
    return {
        "status": status,
        "ntp_time_epoch_ns": state["epoch"],
        "frame_timestamp": state["pts"],
        "queries": state["queries"],
        "buffers": state["buffers"],
        "elapsed_sec": round(time.time() - started, 3),
        "errors": state["errors"],
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        print(__doc__.strip())
        return 0
    if len(args) != 1:
        print(
            json.dumps(
                {
                    "status": "pipeline-error",
                    "error": "usage: ntp_probe.py <timeout-sec>  (RTSP URI on stdin)",
                }
            )
        )
        return 0
    timeout_raw = args[0]
    try:
        timeout_sec = int(timeout_raw)
    except ValueError:
        print(
            json.dumps(
                {"status": "pipeline-error", "error": "timeout-sec must be an integer"}
            )
        )
        return 0
    if timeout_sec <= 0:
        print(
            json.dumps({"status": "pipeline-error", "error": "timeout-sec must be > 0"})
        )
        return 0

    uri = sys.stdin.read().strip()
    if not uri:
        print(
            json.dumps(
                {"status": "pipeline-error", "error": "RTSP URI required on stdin"}
            )
        )
        return 0
    parsed = urlparse(uri)
    if parsed.scheme not in ("rtsp", "rtsps"):
        print(
            json.dumps(
                {
                    "status": "pipeline-error",
                    "error": f"URI scheme must be rtsp/rtsps, got {parsed.scheme!r}",
                }
            )
        )
        return 0

    print(json.dumps(run_ntp_probe(uri, timeout_sec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
