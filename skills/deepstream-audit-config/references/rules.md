# DeepStream Config Validation Rules

Use this reference when the helper script is insufficient or when reviewing a
proposed config/design change.

## Use Case Comes First

Always validate against intent. The same key can be correct in one use case and
wrong in another.

Examples:

- `filesrc`: file/URI input, usually `streammux.live-source=0`.
- `filesrc + latency`: file input paced as live input, usually
  `source-list` / `nvmultiurisrcbin` plus
  `[source-attr-all].simulate-fps-interval-ms`.
- `rtsp`: live camera input, usually `streammux.live-source=1`, explicit RTSP
  reconnect policy, and explicit timestamp policy.
- `dynamic-rtsp`: RTSP plus `nvmultiurisrcbin`, REST endpoint, and sensor-id
  metadata lists.
- `perf`: disable visual overhead and measure throughput.
- `display`: enable tiler/OSD/display sink.
- `msgbroker` / `test5`: ensure message payload timestamps match the chosen NTP
  policy.

Prefer running:

```bash
python3 scripts/validate_config.py <config> --intent "<conversation or generated-config request>"
```

The validator infers use cases from the config and intent. Add explicit
`--usecase` only when you need to force or clarify an intent.

## Static And Approved Runtime Readiness

Static config correctness is not enough. A config can be valid and still fail at
runtime because the host lacks display, plugins, GPU access, media files, or
reachable streams.

Default validation is offline/static and does not launch subprocesses or open
network connections:

```bash
python3 scripts/validate_config.py <config>
```

After explicit approval, `--check-runtime` runs bounded local-host readiness
checks:

```bash
python3 scripts/validate_config.py <config> --check-runtime --timeout-sec 5
```

These checks include:

- `deepstream-app` / `deepstream-test5-app` availability.
- `gst-inspect-1.0` and required plugin availability.
- dGPU/Jetson hints.
- EGL/display environment for display sinks.
- output directory writeability for file sinks and smart record.

For agent-driven validation, start with offline/static checks. Add
`--check-runtime` or runtime probes only after explicit user approval (users
should not need to know these internal flags). Manual CLI equivalents:

```bash
python3 scripts/validate_config.py <config> --check-runtime
python3 scripts/validate_config.py <config> --probe-sources
python3 scripts/validate_config.py <config> --probe-rtsp
python3 scripts/validate_config.py <config> --probe-sinks
python3 scripts/validate_config.py <config> --probe-groups
python3 scripts/validate_config.py <config> --probe-pipeline --confirm-pipeline-probe
python3 scripts/validate_config.py <config> --probe-sources --probe-ntp
```

- File source existence, regular-file/readability/empty-file checks, Git LFS
  pointer detection, and media inspection via GStreamer/ffprobe when available.
- HTTP(S) source reachability and media inspection for URI sources.
- RTSP host/port TCP reachability and optional `gst-discoverer-1.0` probe.
- `--probe-rtsp` runs RTSP source reachability/media checks and RTCP/NTP checks
  together for project sign-off that the stream itself is valid.
- Functional sink smoke tests for enabled sinks:
  - fakesink: minimal GStreamer pipeline.
  - EglSink: display/EGL render smoke test when display is available.
  - file sink: encode/mux/write smoke test to the output directory.
  - RTSP out: encoder/payloader/UDP construction smoke test.
  - msgbroker: library/config path sanity.
- Active group prerequisite checks:
  - PGIE/SGIE plugin and file prerequisites.
  - tracker plugin/library/config prerequisites.
  - preprocess, analytics, dewarper, message converter, and msgbroker paths.
- Exact-config smoke run:
  - `--probe-pipeline` plus `--confirm-pipeline-probe` runs
    `deepstream-app -c <config>` or `deepstream-test5-app -c <config>` for a
    bounded interval after operator approval (GPU/network/broker side effects).
  - This is the strongest check that the configured groups work together with
    the exact values in the supplied config.
- RTSP RTCP Sender Report/NTP sync probe via `scripts/ntp_probe.py` (DeepStream
  `pyds` + `nvds-ntp-sync`) when `--probe-ntp` / `--probe-rtsp` is requested.
- V4L2 device existence/readability and `v4l2-ctl` capability probe when
  available.
- CSI warning when not validating on Jetson.

`--probe-ntp` is the strongest runtime check for camera event-time. It tries to
observe non-zero `ntp-time-epoch-ns` and `frame-timestamp` values generated from
RTCP Sender Reports. If those values are not observed within the timeout, do not
claim that camera NTP is valid; recommend either fixing the camera/server RTCP
Sender Report behavior or using system timestamps (`attach-sys-ts-as-ntp=1`).

## RTSP And NTP Timestamp Policy

`[streammux].attach-sys-ts-as-ntp` controls the timestamp placed in
`NvDsFrameMeta.ntp_timestamp`:

- `1`: attach system time as NTP timestamp. This is the parser default.
- `0`: use RTSP sender NTP from `rtspsrc` / RTCP sender reports when available.

Decision rule:

- Latency test, CI, localhost nvstreamer, or synthetic RTSP: prefer `1`.
- Camera event-time, cross-camera ordering, or downstream message timestamp
  correctness: prefer `0` only if cameras provide trustworthy RTCP/NTP.
- Missing policy on RTSP configs: warn. Defaults are behavior, but not design
  intent.

For `deepstream-test5-app`, this matters because message metadata uses
`frame_meta->ntp_timestamp`.

Related keys to inspect together:

- `[streammux] attach-sys-ts-as-ntp`
- `[streammux] sync-inputs`
- `[streammux] sync-inputs-ntp`
- `[streammux] max-latency`
- `[streammux] frame-duration`
- RTSP source `latency`

## Source Type Contract

- `type=2` URI/file source requires `uri`.
- `type=4` RTSP source requires `uri=rtsp://...` or `rtsps://...`.
- Camera sources need width, height, FPS, and device/sensor fields.
- `num-sources` expands a source group; ensure it matches streammux/GIE batch
  assumptions.

## RTSP Stability

For RTSP in CI or unstable networks, prefer:

- `latency` explicitly set.
- `rtsp-reconnect-interval-sec`.
- `init-rtsp-reconnect-interval-sec`.
- `rtsp-reconnect-attempts`.
- `select-rtp-protocol=4` (TCP) unless UDP is a deliberate choice.

Low latency and stability pull in opposite directions: very low source latency
can reduce delay but increases drop/jitter risk.

## Streammux

Check:

- `live-source=1` for RTSP/v4l2/csi/live `nvmultiurisrcbin`.
- `batch-size` matches active source count unless intentionally over/under
  batching.
- PGIE `batch-size` matches streammux batch.
- `batched-push-timeout` aligns with the goal:
  - latency: around one frame interval, e.g. `33333` us for 30 fps.
  - throughput: commonly `40000` us.
- `cache-buffer`, `sort-batch`, and `align-first-buffer` are usually disabled
  for strict latency.
- `buffer-pool-size` is not too small for the source count.
- `width`, `height`, `enable-padding`, and model input dimensions make sense.

## nvmultiurisrcbin / source-list

Use `nvmultiurisrcbin` when the design needs dynamic add/remove, REST endpoints,
sensor-id metadata, or single-bin aggregation. For a static set of RTSP cameras,
plain `[source0..N] type=4` is often simpler.

When `use-nvmultiurisrcbin=1`, check:

- `num-source-bins`.
- `list`.
- `sensor-id-list` and `sensor-name-list` count match `num-source-bins`.
- `max-batch-size <= num-source-bins`.
- `http-ip` and `http-port`.
- `[source-attr-all]` contains shared source attributes.
- `simulate-fps-interval-ms` belongs in `[source-attr-all]`, not `[source-list]`.
- `low-latency-mode=1` for latency profiles.

## Decode And Queue Pressure

Review:

- `low-latency-decode`.
- `drop-frame-interval`.
- `dec-skip-frames`.
- `num-decode-surfaces`.
- `num-extra-surfaces`.
- `drop-on-latency`.
- `leaky`.
- `max-size-buffers`.

Warn when latency settings drop frames that analytics must preserve, or when
large multi-stream configs have too few decode/extra surfaces.

## GIE, Preprocess, And Tensor Meta

Check:

- Enabled PGIE has `config-file` or complete inline model fields.
- `gie-unique-id` values are unique.
- SGIE `operate-on-gie-id` references an enabled upstream GIE.
- `operate-on-class-ids` matches the upstream detector intent.
- `input-tensor-meta=1` aligns with an enabled preprocess group.
- Engine filename `_bN_` matches `batch-size`; mismatch causes silent rebuilds.

## Tracker

Check:

- `tracker-width` and `tracker-height` are multiples of 32 for NvDCF/NvDeepSORT
  style trackers.
- `ll-lib-file` and `ll-config-file` are valid for the target host.
- `sub-batches`, `sub-batch-err-recovery-trial-cnt`, and
  `user-meta-pool-size` are sane for stream count.
- `operate-on-class-ids` matches PGIE classes when configured.
- Avoid enabling tracker with `source type=3` URI_MULTIPLE RTSP plus
  `num-sources > 1`; this pattern can fail at runtime with
  `motFrame->numBuffers must be 1 when numTransforms is 1`. Prefer separate
  RTSP source groups (`type=4`) or `source-list` / `nvmultiurisrcbin`.

## Sinks And Demux

Validate required keys:

- File sink: `codec`, `bitrate`, `output-file`; container is strongly
  recommended.
- RTSP out: `codec`, `bitrate`, `rtsp-port`, `udp-port`.
- Msgbroker: `msg-conv-config`, `msg-broker-proto-lib`,
  `msg-broker-conn-str`, `topic`.

Warn when:

- User-facing RTSP/file output uses `sync=0`.
- Accuracy/completeness matters and `qos=1` may drop buffers.
- `[tiled-display] enable=2` has zero or multiple `link-to-demux=1` sinks.
- Per-source demux lacks sink coverage for some `source-id`.

## Smart Record

If `smart-record` is enabled, require or review:

- `smart-rec-dir-path`.
- `smart-rec-file-prefix`.
- `smart-rec-cache` or legacy `smart-rec-video-cache`.
- `smart-rec-container`.
- start/default/duration/interval values.

Warn if smart record is used without live/RTSP sources or with incoherent
cache/duration settings.

## SEI, Simulation Time, Dewarper

If using SEI/simulation timestamp keys, document the timestamp semantics:

- `extract-sei-type5-data`.
- `sei-uuid`.
- `extract-sei-sim-time`.
- `drop-backward-sei`.

For dewarper or multi-surface pipelines, align:

- `[dewarper] num-surfaces-per-frame`.
- `[streammux] num-surfaces-per-frame`.
- Downstream GIE/tracker expectations.

## Platform And Paths

- Jetson commonly uses `nvbuf-memory-type=4`; dGPU should avoid surface-array
  memory unless there is a reason.
- EglSink (`type=2`) is dGPU-oriented; Jetson usually uses `type=7` or `type=5`.
- Resolve relative paths from the config file directory.
- Flag absolute DeepStream install-directory paths if the config is intended for
  Jenkins, Docker image portability, or another host.
