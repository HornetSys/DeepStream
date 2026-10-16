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

"""
Step 8b: Write benchmark_data.json from the env vars collect-report-vars.sh exports.

Usage: python3 write-benchmark-data.py
"""
import json
import math
import os
import re


def to_num(v, cast=float):
    """Return cast(v) or None if v is empty/invalid -- prevents malformed JSON."""
    try:
        value = cast(v) if v and str(v).strip() else None
        return value if value is not None and math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def main():
    model_name = os.environ.get("MODEL_NAME", "")
    if model_name in {"", ".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_name):
        raise SystemExit("ERROR: MODEL_NAME must be a safe model directory name")
    data = {
        "model_name": os.environ.get("MODEL_NAME", ""),
        "engine": os.environ.get("ENGINE", ""),
        "max_bs": to_num(os.environ.get("MAX_BS"), int),
        "trtexec": {
            "bs1": {
                "qps": to_num(os.environ.get("QPS_BS1")),
                "gpu_mean_ms": to_num(os.environ.get("GPU_MEAN_BS1")),
            },
            "bsmax": {
                "qps": to_num(os.environ.get("QPS_BS_MAX")),
                "gpu_mean_ms": to_num(os.environ.get("GPU_MEAN_BS_MAX")),
                "p99_ms": to_num(os.environ.get("GPU_P99_BS_MAX")),
                "imgs_per_sec": to_num(os.environ.get("IMGS_PER_SEC")),
            },
        },
        "peak_gpu_streams": to_num(os.environ.get("PEAK_GPU_STREAMS"), int),
        "deepstream": {
            "run1": {
                "streams": to_num(os.environ.get("N_RUN1"), int),
                "total_fps": to_num(os.environ.get("TOTAL_FPS_RUN1")),
                "fps_per_stream": to_num(os.environ.get("FPS_RAW_RUN1")),
                "worst_sustained_fps_per_stream": to_num(os.environ.get("WORST_FPS_RUN1")),
                "all_sources_realtime": os.environ.get("RT_LABEL_RUN1") == "YES",
            },
            "run2": {
                "streams": to_num(os.environ.get("N_RUN2"), int),
                "total_fps": to_num(os.environ.get("TOTAL_FPS_RUN2")),
                "fps_per_stream": to_num(os.environ.get("FPS_RAW_RUN2")),
                "worst_sustained_fps_per_stream": to_num(os.environ.get("WORST_FPS_RUN2")),
                "all_sources_realtime": os.environ.get("RT_LABEL_RUN2") == "YES",
            },
        },
    }
    out_dir = os.path.join("models", model_name, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_data.json")
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    print("benchmark_data.json written")


if __name__ == "__main__":
    main()
