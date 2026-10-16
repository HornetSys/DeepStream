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

# Step 8c-2 pre-check: verify all 5 required charts exist before HTML/PDF generation.
#
# Usage: bash .claude/skills/deepstream-import-vision-model/scripts/report/verify-charts.sh "$MODEL_NAME"

MODEL_NAME="${1:-}"
case "$MODEL_NAME" in ""|.|..|*[!A-Za-z0-9._-]*) echo "ERROR: unsafe MODEL_NAME" >&2; exit 2;; esac
CHART_DIR="models/$MODEL_NAME/reports/charts"
MISSING_CHARTS=0
for CHART in chart_trtexec_bs1_vs_bsmax.png chart_trtexec_throughput.png \
             chart_ds_streams_vs_fps.png chart_trt_vs_ds.png chart_efficiency.png; do
  [ ! -s "$CHART_DIR/$CHART" ] && { echo "ERROR: Missing or empty $CHART_DIR/$CHART"; MISSING_CHARTS=$((MISSING_CHARTS+1)); }
done
[ "$MISSING_CHARTS" -gt 0 ] && { echo "ERROR: $MISSING_CHARTS chart(s) missing -- re-run 8c-1"; exit 1; }
echo "All 5 charts verified OK"
