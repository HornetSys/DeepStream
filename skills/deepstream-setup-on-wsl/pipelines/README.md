# DeepStream verification pipelines

Each `*.sh` file here is one **verification** that runs **inside the DeepStream
container** (inside Docker, inside WSL). `verify_runner.sh` sources them in
filename-sorted order and aggregates their results.

## Adding a new verification (this is the whole contract)

Drop a new `NN-name.sh` file here. The two-digit `NN` prefix just controls run
order (files run in sorted order) — number them `01`, `02`, `03`, … To insert a
test between two existing ones without renumbering, use a letter suffix
(e.g. `02a-tracker.sh` runs between `02-` and `03-`). Inside the file, these are
provided for you by the runner:

| Provided | What it is |
| --- | --- |
| `$DS` | `/opt/nvidia/deepstream/deepstream` |
| `$STREAMS` | `$DS/samples/streams` (sample_720p.mp4, etc.) |
| `$CONFIGS` | `$DS/samples/configs/deepstream-app` |
| `$LOGDIR` | write per-test logs here (collected back to the local machine) |
| `emit <id> <PASS\|FAIL\|SKIP> "<detail>" [k=v …]` | record the verdict + optional metrics |

Rules:
- Call `emit` **once** with a stable `<id>` (kebab-case), a status, and a short
  human reason. `PASS`/`FAIL`/`SKIP`. Anything not PASS/SKIP counts as a failure.
- **Don't PASS on EOS alone.** EOS can occur even if nothing useful happened. Prove
  real work: frame counts, objects detected (e.g. KITTI dump), a valid decodable
  output, non-zero PERF/FPS — and put the evidence in the `emit` detail.
- **Record the exact command** you run to `"$LOGDIR/<id>.cmd"` — the driver prints
  it in the summary and stores it in `results.json` for transparency.
- **Emit structured metrics** as trailing `key=value` tokens on `emit` (numeric
  values become JSON numbers), e.g.
  `emit "$_id" PASS "frames in=$fin==out=$fout" frames_in=$fin frames_out=$fout out_mb=$mb`.
  They land in `results.json` and are a handy **manual regression reference** across
  releases (same input → same frame counts; similar object counts). Report **sizes
  in MB**. Prefer stable metrics (frames, objects) over volatile ones (exact bytes).
- Send verbose output to `"$LOGDIR/<id>.log"` (not stdout).
- **Always bound** the pipeline: `timeout -k 10 "${PIPELINE_TIMEOUT:-180}" gst-launch…`
  so a stuck element fails instead of hanging.
- Keep it headless-safe (`filesink`/`fakesink`; no on-screen sinks — they fail on WSL2).
- Note: the DeepStream image has **no ffprobe / gst-discoverer / python-gi** — rely
  on `gst-launch`, `deepstream-app`, and KITTI/PERF output for evidence.

### Minimal template

```bash
# pipelines/40-my-check.sh
# desc: one-line description of what this validates
_id=my-check
_log="$LOGDIR/$_id.log"
# PASS requires success EXIT + real evidence (never EOS alone): here, a decoded
# frame count > 0 and no errors in the log. Adapt the evidence to your check.
_W=1280; _H=720; _FSZ=$(( _W * _H * 3 / 2 ))  # I420 bytes per frame
_pipefail_was_enabled=0
if set -o | grep -q '^pipefail[[:space:]]*on$'; then _pipefail_was_enabled=1; fi
set -o pipefail
_bytes=$(timeout -k 10 "${PIPELINE_TIMEOUT:-240}" gst-launch-1.0 -q \
     filesrc location="$STREAMS/sample_720p.mp4" ! qtdemux ! h264parse ! nvv4l2decoder ! \
     nvvideoconvert ! "video/x-raw,format=I420,width=$_W,height=$_H" ! fdsink fd=1 \
     2>>"$_log" | wc -c)
_pipe_rc=$?
if [ "$_pipefail_was_enabled" -eq 1 ]; then set -o pipefail; else set +o pipefail; fi
frames=$(( _bytes / _FSZ ))
if [ "$_pipe_rc" -eq 0 ] && [ "$frames" -gt 0 ] && ! grep -qiE 'error|SIGSEGV' "$_log"; then
  emit "$_id" PASS "verified decoded frames=$frames" frames="$frames"
else
  emit "$_id" FAIL "no decoded frames or pipeline error — see logs/$_id.log"
fi
```

Existing checks: `01-version`, `01a-decode` (NVDEC-only, isolates decoder from
encoder), `02-transcode`, `03-deepstream-app-infer`.
