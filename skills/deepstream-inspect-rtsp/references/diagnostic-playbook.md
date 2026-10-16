# Diagnostic Playbook

Use this reference to map RTSP ingest symptoms to likely causes, tests, and source settings. Treat each row as a hypothesis until a minimal pipeline verifies it.

## Source Basis

- NVIDIA DeepStream overview: DeepStream uses GStreamer graphs; RTSP data is captured by CPU, decoded by NVDEC via `Gst-nvvideo4linux2`, then batched with `Gst-nvstreammux`.
- NVIDIA DeepStream troubleshooting: use DeepStream latency measurement env vars, set `[streammux] live-source=1` for RTSP input with the default mux, tune `batched-push-timeout`, match streammux width/height to source resolution, use sink `sync=0`, increase decoder `num-extra-surfaces` when buffers are starved, and adjust `rtspsrc` latency for high-jitter RTSP.
- NVIDIA DeepStream FAQ: for RTSP camera FPS jitter, raise UDP receive capacity, set source `udp-buffer-size`, tune source `latency`, and configure reconnect intervals.
- NVIDIA `nvurisrcbin` docs: `nvurisrcbin` wraps `uridecodebin` and exposes RTSP reconnection, `latency`, `udp-buffer-size`, `select-rtp-protocol`, `num-extra-surfaces`, `drop-frame-interval`, `dec-skip-frames`, `low-latency-mode`, and `drop-on-latency`.
- GStreamer `rtspsrc` docs: `rtspsrc` creates RTP pads, uses RTP session/jitter handling, supports `latency`, `drop-on-latency`, `protocols`, `timeout`, `tcp-timeout`, `udp-buffer-size`, `udp-reconnect`, `ntp-sync`, and newer `tcp-timestamp`.
- NVIDIA forum case on RTSP memory growth: an NVIDIA reply advised isolating RTSP by replacing `rtspsrc` with `filesrc`, testing `rtspsrc ... ! fakesink`, and checking `drop-on-latency` because RTSP buffering can grow when late data is not dropped.
- NVIDIA forum case "Multi-stream Deepstream 9.0 app": a multi-camera deployment with artifacts and 100% decoder utilization was discussed as packet loss coupled with decode saturation; the reply also clarified that video decode is independent of `nvstreammux`.
- NVIDIA forum case "How to resolve latency issues when dynamically adding/removing RTSP streams in nvstreammux": the guidance linked latency to `batched-push-timeout`, active sink pads, dummy-source FPS, and forum-sourced `NVSTREAMMUX_ADAPTIVE_BATCHING=yes`; changing `nvstreammux batch-size` requires a state change.

## Triage Order

1. Verify the symptom with the full pipeline and capture exact messages: GStreamer warnings, EOS, reconnect logs, FPS, latency, RSS, GPU/decoder utilization.
2. Replace the application with the smallest `gst-launch-1.0` pipeline that still uses the camera.
3. Remove DeepStream processing: test `rtspsrc ! fakesink`, then depay/parse, then hardware decode, then mux/fakesink.
4. Replace RTSP with a file or `videotestsrc` at the same resolution/FPS to distinguish RTSP/camera/network faults from downstream DeepStream faults.
5. Tune only one variable per run: transport, `latency`, `drop-on-latency`, UDP buffer, decoder surfaces, sink sync/qos.
6. If only full DeepStream reproduces the issue, inspect component latency and buffer starvation before blaming inference or tracker logic.

## Symptom Matrix

| Symptom | Likely first causes | Minimal tests | First settings to try |
| --- | --- | --- | --- |
| Choppy video or visible packet loss over RTSP output | UDP loss on viewer path, Wi-Fi/VPN/switch loss, overloaded RTSP output port | Force TCP in VLC/client; test local playback near source; check NIC drops | Client TCP, distinct RTSP/UDP sink ports, lower output bitrate |
| Input frame drops with RTSP cameras | Camera jitter, UDP receive buffer too small, low source latency, downstream backpressure | `rtspsrc ! fakesink`; depay/parse/fakesink; compare TCP-only vs UDP/multi | Increase `latency`; set `udp-buffer-size`; raise kernel receive max with approval; consider TCP |
| Latency grows over time | Pipeline slower than realtime, jitterbuffer accumulation, TCP timestamp drift, sink sync behavior | Component latency env vars; `rtspsrc drop-on-latency=true`; fakesink `sync=false` | `drop-on-latency=true`; bounded `latency`; sink `sync=0`; if available, test `tcp-timestamp=true` for RTSP over TCP |
| RSS or memory grows after RTSP EOS/recreate | RTSP jitterbuffer/session teardown, repeated source recreation, late-buffer accumulation, app references not released | Replace RTSP with `filesrc`/`videotestsrc`; test `rtspsrc ! fakesink`; run source/decode only | Cap jitterbuffer with `drop-on-latency=true`; compare teardown without full recreate; verify bus/probe cleanup separately |
| Random disconnects | Camera reboot/network loss, UDP timeout, NAT/firewall, bad switch/PoE, camera session limits | `rtspsrc` debug logs; `ping`; NIC counters; compare TCP-only | DeepStream `rtsp-reconnect-interval-sec`; `rtsp-reconnect-attempts`; `select-rtp-protocol=4`; `tcp-timeout` |
| Low FPS with low CPU/GPU | Buffer starvation, sink QoS drops, too few decoder surfaces, streammux batching timeout | Enable DeepStream latency measurement; inspect FPS per source; fakesink `sync=false` | `num-extra-surfaces`; sink `qos=0`; sink `sync=0`; `batched-push-timeout ~= 1/max_fps` |
| High GPU/NVDEC utilization | Decoder saturation or too many streams for platform | Aggregate stream count x resolution x FPS x codec; platform monitoring | Reduce source FPS/resolution/bitrate; `drop-frame-interval`; `nvurisrcbin dec-skip-frames`; split across GPUs/Jetsons |
| Pixel corruption, smeared snapshots, artifacts | Packet loss, decode saturation, camera bitrate/resolution too high, downstream delay causing RTSP buffering | Check NIC drops and decoder utilization together; compare one camera vs full count; use source/decode-only tests | Camera substream/lower bitrate from RTSP server side; fewer streams per GPU/Jetson; explicit load shedding if acceptable |
| Dynamic add/remove latency with dummy source | `nvstreammux` waiting for batch, dummy source FPS mismatch, active source count below max batch size | Compare with no dummy source; match dummy FPS/resolution; inspect `batched-push-timeout`; measure component latency | Set frame-period timeout; consider forum-sourced `NVSTREAMMUX_ADAPTIVE_BATCHING=yes`; use batch-size/state-change guidance |
| Stale frames | Sink sync, RTSP output port reuse, downstream backlog | Use fakesink; change output ports; check sink queue | sink `sync=0`; unique RTSP/UDP ports; leaky queue only if frame loss is acceptable |

## RTSP Setting Rules

- Use `latency` to size the jitterbuffer. Higher values absorb jitter but add delay and can hide downstream slowness.
- Use `drop-on-latency=true` when latency or memory grows because late packets/backpressure would otherwise let buffers accumulate. This may drop late frames.
- Use `drop-on-latency=false` only as a forensic no-drop test or when the user accepts added delay and memory risk.
- Compare transports:
  - `rtspsrc protocols=tcp` or DeepStream `select-rtp-protocol=4` for TCP-only.
  - DeepStream `select-rtp-protocol=0` for UDP + UDP multicast + TCP negotiation.
- Increase UDP buffers when UDP is required and drops are visible; do not run system `sysctl -w` changes without explicit approval.
- For RTSP over TCP with slowly growing end-to-end latency, check whether `rtspsrc` exposes `tcp-timestamp`; use only when `gst-inspect-1.0 rtspsrc` shows the property.
- Set reconnect intervals for real cameras that reboot or stop sending data; do not use infinite reconnect attempts without telling the user operational trade-offs.
- When the user provides RTSP URLs, give commands with a placeholder or environment variable and do not echo credentials.

## Decoder And Mux Rules

- For decoder starvation, increase DeepStream source or `nvv4l2decoder` `num-extra-surfaces`.
- For `nvurisrcbin`, `dec-skip-frames=1` can skip non-reference frames and `dec-skip-frames=2` can decode key frames only; use only when frame loss is acceptable. Verify the installed plugin properties before applying either value.
- `drop-frame-interval=N` reduces decoded output rate by emitting every Nth frame; use as a load-shedding diagnostic or explicit trade-off.
- Use `low-latency-mode=true` for IPPP streams when the decoder property exists and low latency matters.
- For default `nvstreammux`, set `live-source=1` for live RTSP input and match width/height to the stream. For newer mux behavior, verify current docs because some legacy properties are deprecated.
- Set `batched-push-timeout` near one frame period in microseconds for the highest input FPS, for example 40000 for 25 FPS or 33333 for 30 FPS.
- For dynamic source churn, treat `NVSTREAMMUX_ADAPTIVE_BATCHING=yes` as forum-sourced guidance that should be verified against the installed DeepStream version. Explain that `batched-push-timeout` controls how long mux waits for a batch and that `batch-size` changes require a state transition.
- For display, RTSP output, or fakesink diagnostics, set sink `sync=0`/`sync=false` when checking whether downstream timing is causing drops or stale frames.

## Evidence Standards

Prefer evidence that identifies the first failing stage:

- `rtspsrc ! fakesink` fails: camera/network/RTSP session problem.
- depay/parse fails but raw RTP does not: RTP payload/caps/codec problem.
- decode fails or slows: codec/profile/decoder capacity problem.
- mux/fakesink fails but decode-only does not: streammux batching/timestamp problem.
- full pipeline fails only: inference, tracker, analytics, OSD, sink, or app lifecycle problem.

Always label unverified claims as provisional and give one command that would confirm or falsify the diagnosis.
