#!/usr/bin/env bash

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

# Step 8 pre-flight: derive every variable the report scripts need by reading
# completed benchmark logs, then export them into the caller's shell.
#
# Usage: source .claude/skills/deepstream-import-vision-model/scripts/report/collect-report-vars.sh "$ARGUMENTS"

collect_report_vars() {
[ -n "${1:-}" ] || { echo "ERROR: model directory is required" >&2; return 1; }
STEP8_START=$(date +%s.%N)
export STEP8_START

MODEL_DIR="${1%/}"
MODEL_NAME=$(basename "$MODEL_DIR")
case "$MODEL_NAME" in ""|.|..|*[!A-Za-z0-9._-]*) echo "ERROR: unsafe model name" >&2; return 1;; esac
export MODEL_NAME

SKILL_ROOT="${SKILL_ROOT:-.claude/skills/deepstream-import-vision-model}"

# Engine + MAX_BS + MODEL_FILENAME from the shared resolver.
RESOLVED_ENGINE=$(bash "$SKILL_ROOT/scripts/model/resolve-engine.sh" "$MODEL_NAME") || return 1
eval "$RESOLVED_ENGINE" || return 1
export ENGINE MAX_BS MODEL_FILENAME

# Extract input name and spatial dims from ONNX (needed for reference commands in the report)
unset INPUT_NAME H W
ONNX_FILE=$(ls "models/$MODEL_NAME/model/"*.onnx 2>/dev/null | grep -v '_dynamic' | head -1)
if [ -n "$ONNX_FILE" ]; then
  INSPECT_OUT=$(python3 "$SKILL_ROOT/scripts/model/inspect-onnx.py" "$ONNX_FILE" 2>/dev/null)
  INPUT_NAME=$(echo "$INSPECT_OUT" | grep -oP 'input_name:\s*\K\S+')
  H=$(echo "$INSPECT_OUT" | grep -oP 'height:\s*\K[0-9]+')
  W=$(echo "$INSPECT_OUT" | grep -oP 'width:\s*\K[0-9]+')
fi
INPUT_NAME=${INPUT_NAME:-${MODEL_INPUT_NAME:-}}
H=${H:-${MODEL_INPUT_H:-}}; W=${W:-${MODEL_INPUT_W:-}}
[ -n "$INPUT_NAME" ] && [[ "$H" =~ ^[1-9][0-9]*$ ]] && [[ "$W" =~ ^[1-9][0-9]*$ ]] || {
  echo "BLOCKED: missing numeric model input contract; do not guess spatial dimensions" >&2; return 1;
}
export INPUT_NAME H W

# Parse trtexec BS=1 log — fixed filename trtexec_b1.log (no timestamp, no wildcard needed)
TRTEXEC_LOG_BS1="models/$MODEL_NAME/benchmarks/b1/trtexec_b1.log"
[ -f "$TRTEXEC_LOG_BS1" ] || { echo "ERROR: $TRTEXEC_LOG_BS1 not found — run Steps 4-5 first (references/engine-build.md)"; return 1; }
QPS_BS1=$(grep -oP 'Throughput:\s*\K[0-9.]+' "$TRTEXEC_LOG_BS1" | tail -1)
GPU_MEAN_BS1=$(grep -oP 'GPU Compute Time:.*mean = \K[0-9.]+' "$TRTEXEC_LOG_BS1" | tail -1)
export TRTEXEC_LOG_BS1 QPS_BS1 GPU_MEAN_BS1

# Parse trtexec BS=MAX_BS log — fixed filename trtexec_b${MAX_BS}.log
TRTEXEC_LOG_BSMAX="models/$MODEL_NAME/benchmarks/b${MAX_BS}/trtexec_b${MAX_BS}.log"
[ -f "$TRTEXEC_LOG_BSMAX" ] || { echo "ERROR: $TRTEXEC_LOG_BSMAX not found — run Steps 4-5 first (references/engine-build.md)"; return 1; }
QPS_BS_MAX=$(grep -oP 'Throughput:\s*\K[0-9.]+' "$TRTEXEC_LOG_BSMAX" | tail -1)
GPU_MEAN_BS_MAX=$(grep -oP 'GPU Compute Time:.*mean = \K[0-9.]+' "$TRTEXEC_LOG_BSMAX" | tail -1)
GPU_P99_BS_MAX=$(grep -oP 'GPU Compute Time:.*percentile\(99%\) = \K[0-9.]+' "$TRTEXEC_LOG_BSMAX" | tail -1)
[ -z "$QPS_BS_MAX" ] && { echo "ERROR: Could not parse Throughput from $TRTEXEC_LOG_BSMAX — log may be empty or malformed"; return 1; }
[ -z "$MAX_BS" ] && { echo "ERROR: Could not parse batch size from engine filename: $ENGINE"; return 1; }
export TRTEXEC_LOG_BSMAX QPS_BS_MAX GPU_MEAN_BS_MAX GPU_P99_BS_MAX

BENCHMARK_COUNTS=$(python3 - "$QPS_BS1" "$QPS_BS_MAX" "$MAX_BS" "$GPU_MEAN_BS1" "$GPU_MEAN_BS_MAX" "$GPU_P99_BS_MAX" <<'PY'
import math, sys
values = [float(value) for value in sys.argv[1:]]
if not all(math.isfinite(value) and value > 0 for value in values):
    raise ValueError("Benchmark throughput, batch, and GPU timings must be positive and finite")
imgs = values[1] * values[2]
if not math.isfinite(imgs):
    raise ValueError("Non-finite images/s")
print(round(imgs, 2), math.floor(imgs / 30), round(values[0], 2))
PY
) || return 1
read -r IMGS_PER_SEC PEAK_GPU_STREAMS IMGS_PER_SEC_BS1 <<< "$BENCHMARK_COUNTS"
export IMGS_PER_SEC PEAK_GPU_STREAMS

# Parse DeepStream Run 1 and Run 2 FPS from logs written by pipeline-run
# Fixed filename pattern: benchmarks/ds/ds_s{N}_run1.log and ds_s{N}_run2.log
# Use glob to find them (N varies per model) then extract N from filename
# Step 7's convergence loop can leave SEVERAL run2 logs (e.g. s17 -> s11 failed -> s9 passed).
# Lexicographic `head -1` picks ds_s11 over ds_s9, reporting the NON-real-time run as the
# validated result. Convergence steps downward, so the converged log is the newest — pick by mtime.
# (`sort -V | tail -1` is NOT correct here: it re-selects the highest N, i.e. the failing run.)
DS_LOG_RUN2=$(ls -t models/$MODEL_NAME/benchmarks/ds/ds_s*_run2.log 2>/dev/null | head -1)
DS_LOG_RUN1=""
if [ -n "$DS_LOG_RUN2" ]; then
  DS_LOG_RUN1=$(find "models/$MODEL_NAME/benchmarks/ds" -maxdepth 1 -type f \
    -name 'ds_s*_run1.log' ! -newer "$DS_LOG_RUN2" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-)
fi
[ -z "$DS_LOG_RUN1" ] && { echo "ERROR: No DS Run 1 log found at benchmarks/ds/ds_s*_run1.log — run Steps 6-7 first (references/pipeline-run.md)"; return 1; }
[ -z "$DS_LOG_RUN2" ] && { echo "ERROR: No DS Run 2 log found at benchmarks/ds/ds_s*_run2.log — run Steps 6-7 first (references/pipeline-run.md)"; return 1; }

N_RUN1=$(basename "$DS_LOG_RUN1" | grep -oP 'ds_s\K[0-9]+(?=_run1)')
N_RUN2=$(basename "$DS_LOG_RUN2" | grep -oP 'ds_s\K[0-9]+(?=_run2)')
[[ "$N_RUN1" =~ ^[0-9]+$ ]] || { echo "ERROR: Could not parse stream count from $(basename "$DS_LOG_RUN1") — expected filename pattern ds_s<N>_run1.log"; return 1; }
[[ "$N_RUN2" =~ ^[0-9]+$ ]] || { echo "ERROR: Could not parse stream count from $(basename "$DS_LOG_RUN2") — expected filename pattern ds_s<N>_run2.log"; return 1; }
RT_STREAMS=$N_RUN2
export N_RUN1 N_RUN2 RT_STREAMS

# Use the shared all-source sustained-window parser; never extrapolate source 0.
for REPORT_RUN in 1 2; do
  if [ "$REPORT_RUN" -eq 1 ]; then REPORT_LOG="$DS_LOG_RUN1"; REPORT_STREAMS="$N_RUN1";
  else REPORT_LOG="$DS_LOG_RUN2"; REPORT_STREAMS="$N_RUN2"; fi
  REPORT_PERF=$(python3 - "$SKILL_ROOT/scripts/deepstream/parse-perf.py" "$REPORT_LOG" "$REPORT_STREAMS" <<'PY'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("perf", sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
result = module.parse(Path(sys.argv[2]).read_text(), int(sys.argv[3]))
if result["window_count"] < 5:
    raise ValueError("At least five per-source measurement windows are required")
print(result["aggregate_average_fps"] / result["streams"],
      result["aggregate_average_fps"],
      min(item["minimum_fps"] for item in result["per_stream"]),
      "YES" if result["status"] == "PASS" else "NO")
PY
  ) || return 1
  read -r REPORT_AVERAGE REPORT_TOTAL REPORT_WORST REPORT_RT <<< "$REPORT_PERF"
  if [ "$REPORT_RUN" -eq 1 ]; then
    FPS_RAW_RUN1="$REPORT_AVERAGE"; TOTAL_FPS_RUN1="$REPORT_TOTAL"; WORST_FPS_RUN1="$REPORT_WORST"; RT_LABEL_RUN1="$REPORT_RT";
  else
    FPS_RAW_RUN2="$REPORT_AVERAGE"; TOTAL_FPS_RUN2="$REPORT_TOTAL"; WORST_FPS_RUN2="$REPORT_WORST"; RT_LABEL_RUN2="$REPORT_RT";
  fi
done
export FPS_RAW_RUN1 FPS_RAW_RUN2 TOTAL_FPS_RUN1 TOTAL_FPS_RUN2 WORST_FPS_RUN1 WORST_FPS_RUN2 RT_LABEL_RUN1 RT_LABEL_RUN2

# System info not already captured in pre-flight
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
GPU_NAME=$(echo "$GPU_INFO" | cut -d, -f1 | xargs)
GPU_VRAM=$(echo "$GPU_INFO" | cut -d, -f2 | xargs)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "N/A")
TRT_VER=$(trtexec 2>&1 | head -3 | grep -oP 'TensorRT v\K[0-9.]+' || echo "N/A")
DS_VER=$(deepstream-app --version-all 2>/dev/null | grep -oP 'DeepStreamSDK \K[0-9.]+' || echo "N/A")
ENGINE_SIZE_MB=$(du -m "$ENGINE" | cut -f1)
GPU_P99_BS1=$(grep -oP 'GPU Compute Time:.*percentile\(99%\) = \K[0-9.]+' "$TRTEXEC_LOG_BS1" | tail -1)
GPU_P99_BS1=${GPU_P99_BS1:-"N/A"}  # fallback if log too short to have P99
EFFICIENCY_RUN1=$(python3 -c 'import sys; print(round(float(sys.argv[1]) / float(sys.argv[2]) * 100, 1))' "$TOTAL_FPS_RUN1" "$IMGS_PER_SEC") || return 1
EFFICIENCY_RUN2=$(python3 -c 'import sys; print(round(float(sys.argv[1]) / float(sys.argv[2]) * 100, 1))' "$TOTAL_FPS_RUN2" "$IMGS_PER_SEC") || return 1
export GPU_NAME GPU_VRAM DRIVER_VER CUDA_VER TRT_VER DS_VER ENGINE_SIZE_MB \
       IMGS_PER_SEC_BS1 GPU_P99_BS1 EFFICIENCY_RUN1 EFFICIENCY_RUN2 \
       RT_LABEL_RUN1 RT_LABEL_RUN2

echo "=== Report Variables ==="
echo "MODEL_NAME=$MODEL_NAME  MAX_BS=$MAX_BS"
echo "BS=1:       QPS=$QPS_BS1  GPU mean=${GPU_MEAN_BS1}ms"
echo "BS=$MAX_BS: QPS=$QPS_BS_MAX  imgs/s=$IMGS_PER_SEC  PEAK_GPU_STREAMS=$PEAK_GPU_STREAMS"
echo "DS Run 1:   FPS/stream=$FPS_RAW_RUN1  streams=$N_RUN1  total=$TOTAL_FPS_RUN1 img/s"
echo "DS Run 2:   FPS/stream=$FPS_RAW_RUN2  streams=$N_RUN2  total=$TOTAL_FPS_RUN2 img/s  RT_STREAMS=$RT_STREAMS"
}
collect_report_vars "$@"
