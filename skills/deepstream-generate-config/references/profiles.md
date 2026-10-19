# Profile Reference

These are the deltas the `render.py` overlay applies on top of user-supplied spec values. Spec values always win — overlays only fill `auto`/missing keys.

## Profile: `latency`

Goal: measure or operate at the lowest steady-state per-frame latency. Mirrors `deepstream-dev-test-suite/latency-perf/test5app_config_simulate_fps.txt`.

### Overlay

| Group | Key | Value | Rationale |
|-------|-----|-------|-----------|
| `streammux` | `live-source` | 1 | live data; no batching pile-up |
| `streammux` | `batched-push-timeout` | 33333 | ~30 fps push cadence |
| `streammux` | `batch-size` | `num_sources` | one slot per source, no over-allocation |
| `streammux` | `cache-buffer` | 0 | disable input caching |
| `streammux` | `sort-batch` | 0 | skip per-batch reordering work |
| `streammux` | `align-first-buffer` | 0 | don't wait to align the first buffer across sources |
| `streammux` | `drop-pipeline-eos` | 1 | individual streams ending shouldn't stall the pipeline |
| `streammux` | `attach-sys-ts-as-ntp` | 1 | use sys time so latency math is consistent |
| `tiled-display` | `enable` | 0 | tiler adds frames of latency |
| `osd` | `enable` | 0 (default) | OSD adds GPU work; opt-in only |
| `tests` | `file-loop` | 1 | keep file sources running indefinitely |
| `sink0` (default) | `type` | 1 (FakeSink) | no display path |
| `sink0` | `sync` | 0 | drop frames rather than wait |
| `sink0` | `nvdslogger` | 1 | per-component latency dump |
| `sink0` | `show-latency` | 1 | print stage latency in logs |
| `sink0` | `show-fps` | 1 | print fps in logs |

### Conditional add-ons

- If `sources.kind == source_list`: also set `source-list.use-nvmultiurisrcbin=1`, `source-list.low-latency-mode=1`, `source-list.max-batch-size = num_sources` (≠ greater — `max-batch-size > num-source-bins` causes nvstreammux to wait for slots that never fill).
- If `sources.kind == source_list` and any URI uses the `file://` scheme: auto-set `[source-attr-all].simulate-fps-interval-ms=33` unless the user already supplied a value via `sources.simulate_fps_interval_ms`. **Rationale**: a plain file source decodes as fast as the GPU can go, so a measurement done without pacing under-reports true pipeline latency. `simulate-fps-interval-ms=33` paces the file at ~30 fps, mirroring what a real RTSP camera would deliver. Not applied for live source kinds (`rtsp`/`v4l2`/`csi`) — the device paces the stream for free, and not applied for `perf`/`display` profiles (they don't measure latency).
- **`kind=source_list` for RTSP is opt-in, not the default**: when the user wants latency from RTSP, default to `kind=rtsp` (one `[source0..N]` per camera, `type=4`). Only use `kind=source_list` + `use_nvmultiurisrcbin: true` if the user explicitly asked for dynamic add/remove via the REST endpoint, sensor-id tagging in downstream metadata, or single-bin multi-RTSP aggregation. RTSP through `nvmultiurisrcbin` opens `http-ip`/`http-port` and forces a shared `[source-attr-all]` — extra surface area that's wasted if you just have a static stream set. `simulate-fps-interval-ms` is irrelevant for RTSP either way (the camera paces).
- If `sources.kind == rtsp`: set `source.latency=100`, `source.rtsp-reconnect-interval-sec=10`, `source.init-rtsp-reconnect-interval-sec=5`, `source.rtsp-reconnect-attempts=4`, `source.select-rtp-protocol=4` (TCP).
- If `platform == jetson`: `nvbuf-memory-type=4` everywhere, switch any `egl` sink to `nv3dsink` (`type=7`).

### What the user can still override

`streammux.overrides`, sink-level keys, `primary_gie.interval` (set to 0 by default; raise to 1–2 to skip inferences and trade accuracy for lower latency).

---

## Profile: `perf` (throughput)

Goal: maximum FPS across N streams. Mirrors `samples/configs/deepstream-app/source30_1080p_dec_infer-resnet_tiled_display.txt`.

### Overlay

| Group | Key | Value | Rationale |
|-------|-----|-------|-----------|
| `streammux` | `live-source` | 1 if any source kind is live (rtsp/v4l2/csi), else 0 | matches source nature |
| `streammux` | `batched-push-timeout` | 40000 | ~25 fps cadence; gives muxer time to fill batches |
| `streammux` | `batch-size` | `num_sources` | one slot per source |
| `streammux` | `buffer-pool-size` | 4 | |
| `tiled-display` | `enable` | 0 | tiler costs GPU we want for inference |
| `osd` | `enable` | 0 | same reason |
| `primary_gie` | `interval` | 0 | every frame |
| `primary_gie` | `batch-size` | `streammux.batch-size` | match — enforced by validator |
| `primary_gie` | engine `_bN_` | matches `batch-size` | enforced by validator (warn-only) |
| `tests` | `file-loop` | 1 | sustained measurement |
| `sink0` (default) | `type` | 1 (FakeSink) | no display path |
| `sink0` | `sync` | 1 | render at media rate; perf number is the achievable fps |

### Conditional add-ons

- Same Jetson and source-kind overlays as the latency profile.
- If `secondary_gies` present, validator warns when SGIE `batch-size < num_sources × 4` (a rough lower bound for object density).

---

## Profile: `display`

Goal: human watching the output. Mirrors `samples/configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display.txt`.

### Overlay

| Group | Key | Value | Rationale |
|-------|-----|-------|-----------|
| `streammux` | `live-source` | based on source kind | |
| `streammux` | `batched-push-timeout` | 40000 | smoother for tiler |
| `streammux` | `batch-size` | `num_sources` | |
| `tiled-display` | `enable` | 1 | grid |
| `tiled-display` | `rows`, `columns` | auto-grid: `rows = ceil(sqrt(n))`, `columns = ceil(n / rows)` | |
| `tiled-display` | `width`, `height` | 1280, 720 | |
| `osd` | `enable` | 1 | draw boxes/labels |
| `sink0` (default) | `type` | 2 (EglSink) on dGPU, 7 (nv3dsink) on Jetson | |
| `sink0` | `sync` | 1 | media-rate playback |
| `tests` | `file-loop` | 0 | display once unless user asks |

### Conditional add-ons

- Tracker auto-on with `NvDCF_perf` if not specified (display almost always wants tracker IDs).
- On Jetson, additionally allow `type=5` (nvdrmvideosink) when user states "no X server".

---

## Profile: `custom`

No overlay. The spec must specify:
- `streammux.batch_size`, `streammux.batched_push_timeout` (explicit ints — no `auto`)
- At least one entry under `sinks` with all required keys for that sink type
- `tiled_display.enable` explicitly 0 or 1

The validator still runs and enforces cross-group consistency. Use `custom` only when you're cloning a known-good config one of our customers shipped.

---

## Decision tree (which profile to pick)

```text
Is the goal "measure or operate at the lowest latency"?            → latency
Is the goal "max FPS over many streams, no human watching"?        → perf
Is the goal "show this on a screen / encode an output video"?      → display
Are you porting a known-good config the team already validated?    → custom
```

## Demux is profile-agnostic

`tiled_display.enable: demux` (= `[tiled-display].enable=0`) works under any profile. The profile picks the streammux/sink defaults; demux only changes where the post-inference frames go (split per-source vs tiled vs single shared).

| Profile | Typical demux use |
|---------|-------------------|
| `perf`    | per-source recording or per-source RTSP republish during a multi-stream benchmark |
| `latency` | per-source latency printout (one fakesink per source with `show-latency=1`) so you don't average across streams |
| `display` | rarely combined with demux — display almost always wants the tile. If needed, use `parallel_demux` to tap one source out while still rendering the tile |
| `custom`  | anything goes |

`parallel_demux` (= `[tiled-display].enable=2`) deserves its own note: deepstream-app only honours the **first** sink with `link_to_demux: true` and silently drops the rest. Document this when the user asks for it; do not pretend it scales like the N-sinks demux.

## Combining profiles with sink overlays

You pick **one** profile and **one or more** sinks. The profile sets `sink0` to its default (fakesink for latency/perf, egl/nv3dsink for display) only when the spec's `sinks:` list is empty. If you list sinks, only those appear, in order, as `sink0`, `sink1`, …

Examples:
- `profile: perf` + `sinks: [{type: file}]` → drops the default fakesink, emits a file sink.
- `profile: display` + `sinks: [{type: egl}, {type: file}]` → display plus on-disk recording.
- `profile: latency` + `sinks: [{type: msgbroker, ...}]` → realtime inference with downstream Kafka — typical edge-AI deployment.
