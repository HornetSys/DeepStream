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

# desc: DOC test 3 — deepstream-app on source30 tiled-inference config, sink patched
#       to filesink (nveglglessink fails on WSL). Verified by FRAMES processed +
#       OBJECTS detected (KITTI dump), not just EOS.
_id=deepstream-app-infer
_log="$LOGDIR/$_id.log"
_src="$CONFIGS/source30_1080p_dec_infer-resnet_tiled_display.txt"
_wd=/tmp/dsapp
rm -rf "$_wd"
mkdir -p "$_wd/kitti"
# IMPORTANT: deepstream-app resolves the [primary-gie] config-file, model and label
# paths RELATIVE TO THE MAIN CONFIG'S DIRECTORY — so the patched config must live in
# $CONFIGS (next to the originals), not in /tmp.
_cfg="$CONFIGS/source30_filesink_gen.txt"

if [ ! -f "$_src" ]; then
  emit "$_id" SKIP "source30 config not shipped in this image"
  return 2>/dev/null || exit 0
fi

# Patch the config: [sink0] -> filesink (type=3) since on-screen sinks fail on WSL2;
# add KITTI detection dump so we can count objects actually inferred; stop at EOS.
python3 - "$_src" "$_cfg" "$_wd" <<'PY'
import configparser, os, sys
src, dst, wd = sys.argv[1], sys.argv[2], sys.argv[3]
c = configparser.ConfigParser(strict=False)
c.optionxform = str
c.read(src)
if not c.has_section("sink0"):
    c.add_section("sink0")
for k, v in {"enable": "1", "type": "3", "sync": "0", "container": "1",
             "codec": "1", "enc-type": "0", "bitrate": "4000000",
             "output-file": os.path.join(wd, "out.mp4")}.items():
    c["sink0"][k] = v
# disable any extra on-screen sinks
for s in c.sections():
    if s.startswith("sink") and s != "sink0" and c.has_option(s, "type") and c[s].get("type") == "2":
        c[s]["enable"] = "0"
# deepstream-app dumps primary-detector detections in KITTI format via
# [application] gie-kitti-output-dir — one file per frame, one line per object.
if not c.has_section("application"):
    c.add_section("application")
c["application"]["gie-kitti-output-dir"] = os.path.join(wd, "kitti")
c["application"]["enable-perf-measurement"] = "1"
if c.has_section("tests"):
    c["tests"]["file-loop"] = "0"
with open(dst, "w") as f:
    c.write(f)
PY

_cmd="cd $CONFIGS && deepstream-app -c $_cfg   # source30 tiled inference; [sink0] patched to filesink + KITTI dump"
echo "$_cmd" > "$LOGDIR/$_id.cmd"
echo "COMMAND: $_cmd" > "$_log"; echo "--- patched sink0/primary-gie ---" >> "$_log"
sed -n '/\[sink0\]/,/^\[/p' "$_cfg" >> "$_log"

# cd into $CONFIGS so the config's relative model/label paths resolve; fail fast
# (running deepstream-app from the wrong dir yields misleading errors).
if ! cd "$CONFIGS"; then
  emit "$_id" FAIL "missing config directory: $CONFIGS"
  return 2>/dev/null || exit 0
fi
if timeout -k 15 "${DSAPP_TIMEOUT:-900}" deepstream-app -c "$_cfg" >>"$_log" 2>&1; then
  _rc=0
else
  _rc=$?
fi

_out="$_wd/out.mp4"
# NOTE: the runner sets `shopt -s nullglob`, so an empty kitti glob would make
# `cat …/*.txt` read stdin and `ls …/*.txt` list the cwd. Count with find, and
# only cat when files exist, so "no detections" => 0 (not a hang / cwd count).
_kframes=$(find "$_wd/kitti" -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l)  # frames w/ a KITTI file
if [ "${_kframes:-0}" -gt 0 ]; then
  _objs=$(cat "$_wd"/kitti/*.txt 2>/dev/null </dev/null | wc -l)             # total detected objects
else
  _objs=0
fi
_bytes=$(stat -c%s "$_out" 2>/dev/null || echo 0)
_mb=$(awk "BEGIN{printf \"%.1f\", $_bytes/1048576}")
# Average FPS across sources from deepstream-app's last PERF report (the "(avg)"
# value per source), i.e. real inference throughput — not just "a report printed".
_avgfps=$(grep -aE "PERF:" "$_log" | tail -1 | grep -oE '\([0-9]+\.[0-9]+\)' | tr -d '()' \
  | awk '{s+=$1;n++} END{if(n)printf "%.1f",s/n; else print 0}')
_m=(objects_detected="$_objs" frames_processed="$_kframes" avg_fps="${_avgfps:-0}" out_mb="$_mb")

if [ "$_rc" -eq 0 ] && [ "$_bytes" -gt 1000 ] && [ "${_objs:-0}" -gt 0 ]; then
  emit "$_id" PASS "objects=$_objs across $_kframes frames @ ${_avgfps} avg fps, output ${_mb} MB" "${_m[@]}"
elif [ "$_rc" -eq 124 ]; then
  emit "$_id" FAIL "deepstream-app timed out after ${DSAPP_TIMEOUT:-900}s (objects so far=$_objs) — see logs/$_id.log" "${_m[@]}"
elif [ "$_rc" -ne 0 ]; then
  emit "$_id" FAIL "deepstream-app exit=$_rc (see logs/$_id.log)" "${_m[@]}"
else
  emit "$_id" FAIL "ran but weak result: output ${_mb} MB, objects=$_objs, ${_avgfps} fps" "${_m[@]}"
fi
