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

# Run inside the DeepStream container. Source each filename-sorted pipeline from
# PIPELINES_DIR and write structured results and logs to the bind-mounted OUT.
# Adding a verification means adding a file under pipelines/.
#
# Environment:
#   PIPELINES_DIR  default /verify/pipelines
#   OUT            default /out

set -uo pipefail

export DS=/opt/nvidia/deepstream/deepstream
export STREAMS="$DS/samples/streams"
export CONFIGS="$DS/samples/configs/deepstream-app"
PIPELINES_DIR="${PIPELINES_DIR:-/verify/pipelines}"
OUT="${OUT:-/out}"
export LOGDIR="$OUT/logs"
# Per-pipeline wall-clock cap so a stuck element fails instead of hanging forever.
export PIPELINE_TIMEOUT="${PIPELINE_TIMEOUT:-180}"
mkdir -p "$LOGDIR"
RESULTS_FILE="$OUT/results.jsonl"
: > "$RESULTS_FILE"

json_escape() {
  local value="$1"
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\n'/\\n}
  value=${value//$'\r'/\\r}
  value=${value//$'\t'/\\t}
  printf '%s' "$value"
}

# emit <id> <PASS|FAIL|SKIP> "<detail>" [key=value ...]
#   The optional trailing key=value tokens are recorded as structured, comparable
#   metrics (numeric values kept as numbers) — used for baseline/regression diffs.
emit() {
  local id="$1" status="$2" detail="$3"; shift 3 || true
  detail=${detail//\\/\\\\}; detail=${detail//\"/\\\"}
  local metrics="" kv k v escaped_k escaped_v
  for kv in "$@"; do
    k=${kv%%=*}; v=${kv#*=}
    escaped_k=$(json_escape "$k")
    if [[ "$v" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
      metrics="$metrics${metrics:+,}\"$escaped_k\":$v"        # numeric -> unquoted
    else
      escaped_v=$(json_escape "$v")
      metrics="$metrics${metrics:+,}\"$escaped_k\":\"$escaped_v\""
    fi
  done
  printf '{"id":"%s","status":"%s","detail":"%s","metrics":{%s}}\n' \
    "$id" "$status" "$detail" "$metrics" >> "$RESULTS_FILE"
  echo "[$status] $id — $detail"
}
export -f emit

# environment snapshot for the report
{
  echo "deepstream: $(deepstream-app --version 2>&1 | grep -aoiE 'version [0-9.]+' | head -1)"
  echo "cuda: $(nvcc --version 2>/dev/null | grep -aoiE 'release [0-9.]+' | head -1)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
} > "$OUT/environment.txt" 2>&1

shopt -s nullglob
count=0
for f in "$PIPELINES_DIR"/*.sh; do
  echo "=== pipeline: $(basename "$f") ==="
  # shellcheck disable=SC1090
  ( timeout -k 10 "$PIPELINE_TIMEOUT" bash -u -o pipefail -c 'source "$1"' _ "$f" ) || \
    emit "$(basename "$f" .sh)" FAIL "pipeline script errored (see console)"
  count=$((count + 1))
done
echo "RAN_PIPELINES=$count"
