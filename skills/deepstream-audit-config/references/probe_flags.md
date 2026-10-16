# Runtime probe flags

Default to **offline/static** validation. Add `--probe-*` only after the user
**explicitly approves** runtime or network checks. Skip probes when the user asks
for static-only review.

| Situation | Flag | Side effects |
|-----------|------|--------------|
| Local app/plugin/GPU readiness | `--check-runtime` | Bounded local subprocesses (`gst-inspect-1.0`, optional `nvidia-smi`) |
| RTSP sources | `--probe-rtsp` | Outbound RTSP; also runs `scripts/ntp_probe.py` |
| File / HTTP / V4L2 / CSI sources | `--probe-sources` | Local path checks; outbound HTTP; ffprobe/gst-discoverer |
| Enabled sinks | `--probe-sinks` | May run gst-launch / display helpers |
| Active PGIE / SGIE / tracker / preprocess / msgbroker | `--probe-groups` | May call gst-inspect-1.0 |
| "Will this run", sign-off, thorough check | `--probe-pipeline` **and** `--confirm-pipeline-probe` | Runs `deepstream-app` / `deepstream-test5-app`: GPU/inference, possible file writes, broker messages, network I/O |
| Camera RTCP Sender Report / NTP flow | `--probe-ntp` (often with `--probe-rtsp`) | Outbound RTSP via `scripts/ntp_probe.py` |

All probes honor `--timeout-sec`.

`--probe-pipeline` is a **bounded smoke test**, not nsys profiling. The CLI
refuses `--probe-pipeline` unless `--confirm-pipeline-probe` is also set, and
prints a side-effect warning before launching the app.

For an exact-config smoke test, approval is a hard precondition: do not infer it
from "thorough", "will this run", or similar wording. Before the command, state
the finite timeout, output paths, network endpoints, and broker topics that may
be touched. Afterward, report created files and external effects. Do not delete
files without approval; broker messages cannot be rolled back.

Examples:

```bash
# Static only (default)
python3 scripts/validate_config.py <config> --intent "<user request>"

# After approval: RTSP + NTP
python3 scripts/validate_config.py <config> --intent "<user request>" --probe-rtsp --timeout-sec 10

# After approval: thorough runtime including exact-config smoke
python3 scripts/validate_config.py <config> --intent "<user request>" \
  --probe-sources --probe-sinks --probe-groups --probe-ntp \
  --probe-pipeline --confirm-pipeline-probe --timeout-sec 10
```

Supported `--usecase` values: `filesrc`, `rtsp`, `latency`, `perf`, `display`,
`msgbroker`, `test5`, `dynamic-rtsp`, `recording`, `rtsp-out`, `jetson`, `dgpu`.
Prefer `--intent` from conversation; add `--usecase` only to force/clarify intent.
