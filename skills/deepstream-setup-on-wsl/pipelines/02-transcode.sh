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

# desc: DOC test 2 — HW decode+encode to filesink, validated by exact frame count
# (decode both input and output to raw I420 at a fixed size and count frames from
# the raw byte total — deterministic, needs no ffprobe/discoverer). PASS requires
# the output to decode to the SAME number of frames as the input.
_id=transcode
_log="$LOGDIR/$_id.log"
_out="$LOGDIR/$_id.mp4"
rm -f "$_out"
_cmd="gst-launch-1.0 -e filesrc location=$STREAMS/sample_720p.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=$_out"
echo "$_cmd" > "$LOGDIR/$_id.cmd"
echo "COMMAND: $_cmd" > "$_log"

_W=1280; _H=720; _FSZ=$(( _W * _H * 3 / 2 ))   # I420 bytes per frame
count_frames() {  # $1 = media file -> exact decoded frame count
  local bytes
  bytes=$(timeout -k 10 "${PIPELINE_TIMEOUT:-180}" gst-launch-1.0 -q filesrc location="$1" ! qtdemux ! h264parse ! \
    nvv4l2decoder ! nvvideoconvert ! "video/x-raw,format=I420,width=$_W,height=$_H" ! \
    fdsink fd=1 2>/dev/null | wc -c)
  echo $(( bytes / _FSZ ))
}
mb() { awk "BEGIN{printf \"%.1f\", $1/1048576}"; }

if timeout -k 10 "${PIPELINE_TIMEOUT:-180}" bash -c "$_cmd" >>"$_log" 2>&1 && [ -s "$_out" ]; then
  _bytes=$(stat -c%s "$_out"); _mb=$(mb "$_bytes")
  _fin=$(count_frames "$STREAMS/sample_720p.mp4")
  _fout=$(count_frames "$_out")
  echo "frames_in=$_fin frames_out=$_fout out_mb=$_mb" >> "$_log"
  if [ "${_fout:-0}" -gt 0 ] && [ "${_fout:-0}" = "${_fin:-x}" ]; then
    emit "$_id" PASS "frames in=$_fin == out=$_fout, output ${_mb} MB" \
      frames_in="$_fin" frames_out="$_fout" out_mb="$_mb"
  elif [ "${_fout:-0}" -gt 0 ]; then
    emit "$_id" FAIL "frame count mismatch: in=$_fin out=$_fout (encoder dropped/added frames)" \
      frames_in="$_fin" frames_out="$_fout" out_mb="$_mb"
  else
    emit "$_id" FAIL "output produced (${_mb} MB) but no decodable frames" \
      frames_in="$_fin" frames_out="$_fout" out_mb="$_mb"
  fi
elif [ $? -eq 124 ]; then
  emit "$_id" FAIL "timed out after ${PIPELINE_TIMEOUT:-180}s (encoder busy/unavailable?) — see logs/$_id.log"
else
  emit "$_id" FAIL "transcode pipeline error / no output — see logs/$_id.log"
fi
