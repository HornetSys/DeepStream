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

################################################################################
# Step 7c: DeepStream perf-measurement run via deepstream-app.
#
# Replaces the older `gst-launch-1.0 ... ! fpsdisplaysink ...` benchmark, which
# pulled in `gstreamer1.0-plugins-bad`. `deepstream-app` is part of the NVIDIA
# DeepStream SDK and emits `**PERF: fps_run0 (fps_avg0)  fps_run1 (fps_avg1)  ...`
# lines (one pair per active source) that the report-generation phase parses.
#
# Usage: ./ds-perf-run.sh <nvinfer_config> <num_streams> <log_path> [input_video]
# Example:
#   ./ds-perf-run.sh config_infer_ds_yolox.txt 32 \
#       models/yolox/benchmarks/ds/ds_s32_run1.log \
#       /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4
#
# Notes:
#   - `[primary-gie] batch-size` and `[streammux] batch-size` are both set to N
#     (matches the skill-wide rule batch_size == num_streams).
#   - `num-sources=N` fans the single input video out to N pipeline sources;
#     deepstream-app handles the file-loop / EOS bookkeeping.
#   - The nvinfer config must already point at the engine, parser, and labels.
#     This script does NOT mutate the nvinfer config.
#   - Writes ${log_path}.json using the canonical per-source parser. Every
#     source must sustain DS_TARGET_FPS (default 30) in the final five windows.
#     DS_TASK (default detection) and DS_CAMPAIGN_ID label the result.
#   - Existing log/JSON outputs are never overwritten; choose a fresh path.
################################################################################
set -euo pipefail

NVINFER_CONFIG="${1:-}"
NUM_STREAMS="${2:-}"
LOG_PATH="${3:-}"
VIDEO="${4:-/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4}"

if [ -z "$NVINFER_CONFIG" ] || [ -z "$NUM_STREAMS" ] || [ -z "$LOG_PATH" ]; then
    echo "Usage: $0 <nvinfer_config> <num_streams> <log_path> [input_video]"
    exit 1
fi

[[ "$NUM_STREAMS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: num_streams must be a positive integer" >&2
    exit 1
}

[ -f "$NVINFER_CONFIG" ] || { echo "ERROR: nvinfer config not found: $NVINFER_CONFIG"; exit 1; }
[ -f "$VIDEO" ] || { echo "ERROR: video file not found: $VIDEO"; exit 1; }
command -v deepstream-app >/dev/null 2>&1 || { echo "ERROR: deepstream-app not on PATH"; exit 1; }

NVINFER_CONFIG="$(realpath "$NVINFER_CONFIG")"
VIDEO="$(realpath "$VIDEO")"
# Resolve the parent, not the final component: even dangling output symlinks
# must be rejected rather than followed to a new target.
LOG_PATH="$(realpath -m "$(dirname "$LOG_PATH")")/$(basename "$LOG_PATH")"
JSON_PATH="${LOG_PATH}.json"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for OUTPUT_PATH in "$LOG_PATH" "$JSON_PATH"; do
    if [ -e "$OUTPUT_PATH" ] || [ -L "$OUTPUT_PATH" ]; then
        echo "ERROR: output already exists: $OUTPUT_PATH" >&2
        exit 1
    fi
done
mkdir -p "$(dirname "$LOG_PATH")"
# Atomically reserve the log and retain its descriptor so tee never reopens
# the caller's path. The parser independently creates the JSON exclusively.
set -o noclobber
if ! exec {LOG_FD}>"$LOG_PATH"; then
    echo "ERROR: cannot exclusively create log: $LOG_PATH" >&2
    exit 1
fi
set +o noclobber

N="$NUM_STREAMS"
MUXER_W=1280
MUXER_H=720

echo "=== DeepStream Perf Run ==="
echo "nvinfer config: $NVINFER_CONFIG"
echo "Streams (=N):   $N"
echo "Input video:    $VIDEO"
echo "Log path:       $LOG_PATH"
echo ""

trap 'rm -f "${TMPCONFIG:-}"' EXIT
TMPCONFIG=$(mktemp /tmp/ds_perf_XXXXXX.txt)

cat > "$TMPCONFIG" <<EOF
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=2

[tiled-display]
enable=0

[source0]
enable=1
type=3
uri=file://${VIDEO}
num-sources=${N}
gpu-id=0

[sink0]
enable=1
type=1
sync=0

[osd]
enable=0

[streammux]
live-source=0
batch-size=${N}
batched-push-timeout=-1
width=${MUXER_W}
height=${MUXER_H}

[primary-gie]
enable=1
batch-size=${N}
gie-unique-id=1
config-file=${NVINFER_CONFIG}

[tests]
file-loop=1
EOF

echo "Temp config: $TMPCONFIG"
echo "Running deepstream-app..."

set +o pipefail
# file-loop=1 has no built-in stop condition; timeout(1) kills deepstream-app
# after 60 s and returns exit 124 — treated as success below.
timeout 60s deepstream-app -c "$TMPCONFIG" 2>&1 | tee "/dev/fd/$LOG_FD"
PIPE_EXIT_CODES=("${PIPESTATUS[@]}")
DS_EXIT_CODE=${PIPE_EXIT_CODES[0]}
set -o pipefail
exec {LOG_FD}>&-

if [ "${PIPE_EXIT_CODES[1]}" -ne 0 ]; then
    echo "ERROR: failed to write log: $LOG_PATH" >&2
    exit "${PIPE_EXIT_CODES[1]}"
fi

# exit 124 = timeout fired as expected (file-loop=1, 60 s cap)
if [ $DS_EXIT_CODE -ne 0 ] && [ $DS_EXIT_CODE -ne 124 ]; then
    echo "ERROR: deepstream-app exited with code $DS_EXIT_CODE — see $LOG_PATH" >&2
    exit "$DS_EXIT_CODE"
fi

echo ""
echo "=== Per-source Perf Summary ==="
python3 "$SCRIPT_DIR/parse-perf.py" "$LOG_PATH" "$N" \
    --minimum-fps "${DS_TARGET_FPS:-30}" \
    --minimum-windows 5 \
    --task "${DS_TASK:-detection}" \
    --campaign-id "${DS_CAMPAIGN_ID:-}" \
    --output "$JSON_PATH"
