#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate sustained FPS for every logical DeepStream source separately."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


PAIR = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)")


def parse(log: str, streams: int, *, minimum_fps: float = 30.0,
          minimum_windows: int = 5) -> dict:
    if streams < 1 or minimum_windows < 1 or not math.isfinite(minimum_fps) or minimum_fps <= 0:
        raise ValueError("streams, minimum FPS, and window count must be positive")
    windows: list[list[float]] = []
    for line in log.splitlines():
        if "**PERF:" not in line:
            continue
        payload = line.split("**PERF:", 1)[1]
        # deepstream-app prints a column header before the first measurement.
        # Skip only that recognized header; malformed measurement lines fail.
        if re.fullmatch(r"(?:\s*FPS\s+\d+\s+\(Avg\)\s*)+", payload):
            continue
        values = [float(match.group(1)) for match in PAIR.finditer(payload)]
        if len(values) != streams:
            raise ValueError(f"PERF line has {len(values)} sources; expected {streams}")
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("PERF line contains invalid FPS")
        windows.append(values)
    recent = windows[-minimum_windows:]
    per_stream = []
    for source_id in range(streams):
        values = [window[source_id] for window in recent]
        per_stream.append({
            "source_id": source_id,
            "windows_fps": values,
            "minimum_fps": min(values) if values else None,
            "average_fps": sum(values) / len(values) if values else None,
        })
    status = ("PASS" if len(recent) == minimum_windows and
              all(item["minimum_fps"] >= minimum_fps for item in per_stream)
              else "FAIL")
    return {
        "schema_version": 1,
        "producer": "deepstream-per-source-perf-parser",
        "status": status,
        "streams": streams,
        "window_count": len(windows),
        "sustained_windows": minimum_windows,
        "target_fps_per_stream": minimum_fps,
        "per_stream": per_stream,
        "aggregate_average_fps": sum(item["average_fps"] or 0 for item in per_stream),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("streams", type=int)
    parser.add_argument("--minimum-fps", type=float, default=30.0)
    parser.add_argument("--minimum-windows", type=int, default=5)
    parser.add_argument("--task", default="")
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = parse(args.log.read_text(encoding="utf-8", errors="replace"), args.streams,
                   minimum_fps=args.minimum_fps, minimum_windows=args.minimum_windows)
    result["log_sha256"] = hashlib.sha256(args.log.read_bytes()).hexdigest()
    result["task"] = args.task
    result["campaign_id"] = args.campaign_id
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
