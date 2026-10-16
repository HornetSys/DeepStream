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

# desc: NVDEC decode-only health — isolates the decoder from the encoder (02-transcode).
# Named 01a- to run between version (01) and transcode (02): decode is the more
# fundamental capability, so a failure here pinpoints the decoder vs the encoder.
_id=decode
_log="$LOGDIR/$_id.log"
_W=1280; _H=720; _FSZ=$(( _W * _H * 3 / 2 ))   # I420 bytes per frame
_cmd="gst-launch-1.0 filesrc location=$STREAMS/sample_720p.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw,format=I420,width=$_W,height=$_H ! fakesink"
echo "$_cmd" > "$LOGDIR/$_id.cmd"
echo "COMMAND: $_cmd" > "$_log"

# Decode to raw I420 and count frames from the raw byte total (deterministic).
# pipefail ensures a failed gst-launch exit is not masked by wc -c receiving bytes.
_pipefail_was_enabled=0
if set -o | grep -q '^pipefail[[:space:]]*on$'; then
  _pipefail_was_enabled=1
fi
set -o pipefail
_bytes=$(timeout -k 10 "${PIPELINE_TIMEOUT:-180}" gst-launch-1.0 -q \
  filesrc location="$STREAMS/sample_720p.mp4" ! qtdemux ! h264parse ! nvv4l2decoder ! \
  nvvideoconvert ! "video/x-raw,format=I420,width=$_W,height=$_H" ! fdsink fd=1 2>>"$_log" | wc -c)
_pipe_rc=$?
if [ "$_pipefail_was_enabled" -eq 1 ]; then
  set -o pipefail
else
  set +o pipefail
fi
_frames=$(( _bytes / _FSZ ))

if [ "$_pipe_rc" -eq 0 ] && [ "$_frames" -gt 0 ]; then
  emit "$_id" PASS "NVDEC decoded $_frames frames" frames="$_frames"
else
  emit "$_id" FAIL "no frames decoded (NVDEC, pipeline_rc=$_pipe_rc) — see logs/$_id.log" frames=0
fi
