# Natural-language → spec mapping

Use when building a YAML spec from prose (`SKILL.md` Mode C). Schema fields live
in `scripts/spec_schema.yaml`, which the skill entrypoint references directly.

| Phrase | Spec change |
|--------|-------------|
| "RTSP" / "live cameras" / "from cameras" | `sources.kind: rtsp`, `sources.live: true`. **Do NOT auto-promote to `source_list` / `nvmultiurisrcbin`** — for RTSP, the default `kind: rtsp` (one `[source0..N]` per camera, `type=4`) is simpler, has no HTTP port, and is fine for any static set of cameras. |
| **"nvmultiurisrcbin for RTSP" / "REST API" / "dynamic add/remove" / "add cameras at runtime" / "sensor-id tagging" / "dynamic streams"** | `sources.kind: source_list`, `sources.use_nvmultiurisrcbin: true`, URIs go in `sources.uris`. Opt-in only. Costs: HTTP REST port (`http-ip` / `http-port`), all sources share `[source-attr-all]`. |
| "lowest latency" / "real time" / "low latency" | `profile: latency` |
| "max fps" / "throughput" / "perf" / "benchmark" | `profile: perf` |
| "display" / "show it on screen" / "monitor" | `profile: display`, `sinks: [{type: egl}]` |
| "record to file" / "save mp4" | append `{type: file, output_file: out.mp4}` to `sinks` |
| "kafka" / "msgbroker" / "publish events" | append `{type: msgbroker, ...}` to `sinks` |
| "30 streams" / "many streams" / "lots of streams" | `sources.num_sources: 30` (and `kind: source_list` if user said multistream) |
| "jetson" / "orin" / "nano" | `deployment.platform: jetson` |
| "test5" / "ota" / "consumer" | `deployment.app: test5`, msgbroker consumer enabled |
| **"latency test from a file" / latency profile + `file://` URI** | force `kind: source_list`, `use_nvmultiurisrcbin: true` so latency profile injects `simulate-fps-interval-ms=33`. Tell the user why: unpaced file decode makes latency look artificially low. Not for RTSP/v4l2/csi. |
| **"demux" / "per-source recording" / "one MP4 per camera" / "per-source RTSP-out"** | `tiled_display.enable: demux` (`[tiled-display].enable=0`). One `sinks:` entry per source with `source_id: 0..N-1`. No `link_to_demux` for this pattern. |
| **"parallel demux" / "tap one source while still rendering tile"** | `tiled_display.enable: parallel_demux` (`enable=2`). Exactly ONE sink with `link_to_demux: true` and a `source_id`. |

Write the spec to a writable path (for example `./ds_spec.yaml`) and proceed to render.
