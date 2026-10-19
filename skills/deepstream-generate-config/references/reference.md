# deepstream-app Config Reference

Validated for DeepStream SDK 9.2.x against
`src/apps/common/includes/deepstream_config_file_parser.h` and
`src/apps/common/src/deepstream_config_file_parser.c`. Verify those parser sources before
using this table with another DeepStream release. Use this as the lookup table
when filling in `streammux.overrides` or writing `custom` profiles.

`.txt` (GKeyFile) uses `[group]` headers + `key=value`. `.yml` uses `group:\n  key: value` with the same key names.

## Groups

| Group | Multiplicity | Purpose |
|-------|--------------|---------|
| `application` | 1 | App-level perf measurement, kitti dumps, global gpu-id |
| `tiled-display` | 1 | nvmultistreamtiler — grid of streams |
| `source0` … `source9` | many (classic) | Per-stream source bins (use either this OR source-list) |
| `source-list` | 1 | nvmultiurisrcbin / REST-driven sources (alternative to per-source groups) |
| `source-attr-all` | 1 | Defaults applied to every entry in `source-list` |
| `streammux` | 1 | nvstreammux — the most important block for latency/perf |
| `pre-process` | 1 | nvdspreprocess before PGIE |
| `secondary-pre-process` | 1 | nvdspreprocess before SGIE |
| `primary-gie` | 1 | Primary nvinfer / nvinferserver |
| `secondary-gie0` … | many | SGIEs operating on PGIE output |
| `tracker` | 1 | nvtracker |
| `osd` | 1 | nvdsosd |
| `sink0` … `sink9` | many | Per-sink branches (fakesink/egl/file/rtsp_out/msgbroker) |
| `message-converter` | 1 | nvmsgconv (only when using broker across multiple sinks, or in test5) |
| `message-consumer0` … | many | nvmsgbroker subscriber (test5 only) |
| `nvds-analytics` | 1 | nvdsanalytics ROI/line crossing |
| `tests` | 1 | `file-loop`, `pipeline-recreate-sec` |

## `[application]`

| Key | Type | Notes |
|-----|------|-------|
| `enable-perf-measurement` | 0/1 | Always 1 for our profiles |
| `perf-measurement-interval-sec` | int | 5 is typical |
| `gie-kitti-output-dir` | path | Dump detector output as KITTI |
| `kitti-track-output-dir` | path | Tracker KITTI output |
| `reid-track-output-dir` | path | ReID dumps |
| `terminated-track-output-dir` | path | NvDCF terminated tracks |
| `shadow-track-output-dir` | path | NvDCF shadow tracks |
| `global-gpu-id` | int | Override gpu-id for the whole app |

## `[tiled-display]`

| Key | Type | Notes |
|-----|------|-------|
| `enable` | 0/1/2 | **0 = no tile, demuxer auto-created** when sinks declare distinct `source-id` (the N-sources→N-sinks pattern). **1 = nvmultistreamtiler** (grid into one output buffer). **2 = parallel demux + tile** (tiler runs and ONE sink with `link-to-demux=1` taps one source out — additional `link-to-demux=1` sinks are silently dropped). Spec aliases: `enable: demux` ⇒ 0, `enable: parallel_demux` ⇒ 2. |
| `rows`, `columns` | int | Auto-grid: `rows=ceil(sqrt(n))`, `columns=ceil(n/rows)`. Ignored when `enable=0`. |
| `width`, `height` | int | 1280×720 typical |
| `gpu-id` | int | |
| `nvbuf-memory-type` | 0–4 | 0 dGPU default; 4 (surface-array) recommended on Jetson |
| `square-seq-grid` | 0/1 | Forces a perfect square grid |

## `[source*]` (classic per-source) and `[source-attr-all]`

| Key | Type | Notes |
|-----|------|-------|
| `enable` | 0/1 | |
| `type` | 1=CameraV4L2, 2=URI, 3=MultiURI, 4=RTSP, 5=CSI, 6=Audio | |
| `uri` | string | `file://...` or `rtsp://...` |
| `num-sources` | int | For `type=3` MultiURI replicates the same URI N times |
| `gpu-id` | int | |
| `cudadec-memtype` | 0/1/2 | 0=device, 1=pinned, 2=unified |
| `live-source` | 0/1 | Set per-source live flag (also on streammux) |
| `latency` | int µs | RTSP jitter buffer; 100 typical for RTSP |
| **drop-frame-interval** | int | Decoder-level frame skip |
| **drop-on-latency** | 0/1 | rtspsrc drop-on-latency |
| `select-rtp-protocol` | 0=UDP+TCP, 4=TCP-only | TCP is more reliable |
| `rtsp-reconnect-interval-sec` | int | 10 typical |
| `rtsp-reconnect-attempts` | int | Default unlimited |
| `init-rtsp-reconnect-interval-sec` | int | Initial backoff |
| `buffer-mode` | 0–4 | rtspsrc buffer mode |
| `intra-decode-enable` | 0/1 | Decode only I-frames |
| `dec-skip-frames` | 0/1/2 | 0=all, 1=non-ref+B, 2=non-ref |
| `num-decode-surfaces` | int | nvv4l2decoder surfaces |
| `num-extra-surfaces` | int | |
| `smart-record` | 0/1/2 | 0=off, 1=event, 2=continuous |
| `smart-rec-dir-path`, `smart-rec-file-prefix`, `smart-rec-cache`, `smart-rec-container`, `smart-rec-default-duration`, `smart-rec-duration`, `smart-rec-interval`, `smart-rec-start-time` | various | |
| `camera-width`, `camera-height`, `camera-fps-n`, `camera-fps-d` | int | v4l2/CSI |
| `camera-v4l2-dev-node` | int | `/dev/video<N>` |
| `camera-csi-sensor-id` | int | Argus CSI sensor id |
| `nvbuf-memory-type` | 0–4 | |

## `[source-list]` (nvmultiurisrcbin)

When `use-nvmultiurisrcbin=1`, five keys are **hard-required at init time** — missing any of them aborts before the first frame. The render script fills them automatically; the validator (`validate.py`) errors out if they're missing or mis-counted.

| Key | Type | Notes |
|-----|------|-------|
| `num-source-bins` | int | Must equal length of `list` |
| `list` | semicolon-separated URIs | `uri1;uri2;...` |
| `sensor-id-list` | semicolon | **REQUIRED when `use-nvmultiurisrcbin=1`.** Semicolon-count MUST equal `num-source-bins` — mismatch aborts with `Mismatch in Sensor IDs provided and num-source-bins` (`parse_source_list:188`). |
| `sensor-name-list` | semicolon | **REQUIRED when `use-nvmultiurisrcbin=1`.** Same count rule as `sensor-id-list`. |
| `use-nvmultiurisrcbin` | 0/1 | 1 enables the bin |
| `max-batch-size` | int | **Must match `num-source-bins`** or you'll accumulate latency |
| `stream-name-display` | 0/1 | Overlay stream name |
| `dummy-payload` | 0/1 | |
| `custom-ts-to-rfc` | 0/1 | |
| `batched-sensor-name` | string | |
| `http-ip` | string | **REQUIRED when `use-nvmultiurisrcbin=1`.** nvmultiurisrcbin always starts its REST endpoint for dynamic add/remove — there's no way to disable. Missing value becomes `std::string(nullptr)` and aborts with `basic_string: construction from null is not valid`. Default in the template: `localhost`. |
| `http-port` | int | **REQUIRED when `use-nvmultiurisrcbin=1`.** Same crash mode as `http-ip`. Default in the template: `9000`. |
| `sgie-batch-size` | int | Override SGIE batch when set |
| `extract-sei-type5-data`, `sei-uuid`, `low-latency-mode` | various | `low-latency-mode=1` for live |
| `enable-error-propagation`, `proto-lib`, `conn-str`, `topic` | error msg bus | |

> **Do NOT put `simulate-fps-interval-ms` in `[source-list]`.** It is a `[source-attr-all]` key; the `[source-list]` parser rejects it with `Unknown key 'simulate-fps-interval-ms'` and the keep-going parser then misaligns subsequent keys, producing cascading `parse_source_list` errors. `validate.py` errors out on this.

## `[source-attr-all]` extras for nvmultiurisrcbin file sources

| Key | Type | Notes |
|-----|------|-------|
| `simulate-fps-interval-ms` | int (ms) | Paces FILE URIs at one frame every N ms (e.g. 33 = ~30 fps). **Required for honest end-to-end latency from a file**; otherwise the decoder races and the latency you'd measure is artificially low. Ignored for live sources (RTSP/v4l2/CSI — the device paces). render.py auto-injects `33` when `profile=latency` AND `kind=source_list` AND at least one URI uses `file://`. Override via `sources.simulate_fps_interval_ms`. |
| `sensorID-padID-mapping` | 0/1 | `0` (default) lets nvmultiurisrcbin assign pad-ids implicitly. Keep at `0` unless you've already mapped sensor-id→pad-id elsewhere in the downstream metadata. |

## `[streammux]` — the latency/perf hot spot

| Key | Type | Notes |
|-----|------|-------|
| `gpu-id` | int | |
| `batch-size` | int | **Must equal num_sources (and PGIE batch-size)** |
| `batched-push-timeout` | int µs | **33333 for realtime (~30 fps), 40000 for file/perf, -1 to wait forever** |
| `width`, `height` | int | Muxer output resolution |
| `live-source` | 0/1 | **1 for RTSP/cameras; 0 for file** |
| `enable-padding` | 0/1 | Aspect-ratio padding |
| `nvbuf-memory-type` | 0–4 | |
| `attach-sys-ts-as-ntp` | 0/1 | 1 = sys ts; 0 = rtspsrc ts |
| `frame-num-reset-on-stream-reset` | 0/1 | |
| `frame-num-reset-on-eos` | 0/1 | |
| `frame-duration` | int ns | Force fixed frame duration |
| `config-file-path` | path | `nvstreammux` new-API config |
| `sync-inputs` | 0/1 | Real-time sync across inputs |
| `sync-inputs-ntp` | 0/1 | NTP-based sync |
| `max-latency` | int ns | |
| `async-process` | 0/1 | |
| `drop-pipeline-eos` | 0/1 | **1 in latency profile** |
| `num-surfaces-per-frame` | int | |
| `extract-sei-type5-data`, `extract-sei-sim-time` | 0/1 | |
| `align-first-buffer` | 0/1 | **0 in latency profile** |
| `cache-buffer`, `cache-buffer-timeout` | 0/1, int | **`cache-buffer=0` in latency profile** |
| `sort-batch` | 0/1 | **0 in latency profile** |
| `failsafe-flush-count` | int | |
| `drop-backward-sei` | 0/1 | |
| `buffer-pool-size` | int | 4 typical |

## `[primary-gie]` / `[secondary-gie*]`

| Key | Type | Notes |
|-----|------|-------|
| `enable` | 0/1 | |
| `gpu-id` | int | |
| `batch-size` | int | **PGIE = streammux.batch-size; SGIE = max objects per frame × num streams** |
| `interval` | int | Skip N frames between inferences (PGIE only) |
| `gie-unique-id` | int | 1 for PGIE, 4/5/… for SGIEs |
| `operate-on-gie-id` | int | SGIE: PGIE id whose output it operates on |
| `operate-on-class-ids` | semicolon | SGIE: classes to process |
| `model-engine-file` | path | TensorRT engine; **filename should contain `_b{batch-size}_`** |
| `config-file` | path | nvinfer config (the inner `config_infer_primary.txt`) |
| `labelfile-path` | path | Required if not in config-file |
| `bbox-border-color0` … | r;g;b;a | Per-class BBOX color (OSD) |
| `nvbuf-memory-type` | 0–4 | |
| `input-tensor-meta` | 0/1 | When pre-process attaches tensors |

## `[tracker]`

| Key | Type | Notes |
|-----|------|-------|
| `enable` | 0/1 | |
| `tracker-width`, `tracker-height` | int | Multiples of 32 for NvDCF/NvDeepSORT |
| `gpu-id` | int | |
| `ll-lib-file` | path | `/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so` |
| `ll-config-file` | path | One of `config_tracker_{IOU,NvSORT,NvDCF_perf,NvDCF_accuracy,NvDeepSORT,MaskTracker}.yml` |
| `display-tracking-id` | 0/1 | |
| `tracking-id-reset-mode` | 0–3 | 0=never, 1=on stream reset, 2=on EOS, 3=always |

## `[osd]`

| Key | Type | Notes |
|-----|------|-------|
| `enable` | 0/1 | |
| `gpu-id` | int | |
| `border-width` | int | |
| `text-size` | int | |
| `text-color`, `text-bg-color`, `clock-color` | `r;g;b;a` | |
| `font` | string | `Serif`, `Arial` |
| `show-clock` | 0/1 | |
| `clock-x-offset`, `clock-y-offset`, `clock-text-size` | int | |
| `display-text` | 0/1 | |
| `display-bbox` | 0/1 | |
| `display-mask` | 0/1 | |
| `nvbuf-memory-type` | 0–4 | |

## `[sink*]`

Sink `type` values:
- `1`: FakeSink
- `2`: EglSink (`nveglglessink`) — **dGPU only**
- `3`: File (encoder + filesink)
- `4`: RTSPStreaming (encoder + udpsink + rtspserver)
- `5`: nvdrmvideosink (Jetson, no X)
- `6`: MsgConvBroker (`nvmsgconv + nvmsgbroker` combined)
- `7`: nv3dsink (Jetson display)

Common subkeys (per type):

| Key | Used by | Notes |
|-----|---------|-------|
| `enable` | all | |
| `type` | all | See above |
| `sync` | all | **0 in latency profile**; 1 in display/perf; RTSP-out template defaults to 1 (media-rate pacing) |
| `source-id` | all | Which stream (after demux); meaningful when `tiled-display.enable=0` (auto-demux) or `enable=2` (parallel demux). With `enable=1` (tile), all sinks see the tiled output regardless of `source-id`. |
| `link-to-demux` | all | Set to 1 only when `tiled-display.enable=2`. Picks which single source-id is tapped out of the demuxer running parallel to the tiler. Only the first sink with this flag is honoured. Spec key: `sinks[*].link_to_demux: true`. |
| `gpu-id` | all | |
| `nvbuf-memory-type` | all | 4 on Jetson |
| `show-fps`, `show-latency`, `nvdslogger` | fakesink | Latency profile sets `show-fps=1 show-latency=1 nvdslogger=1` |
| `container` | file | 1=mp4, 2=mkv |
| `codec` | file/rtsp_out | 1=h264, 2=h265, 3=mpeg4 (sw only) |
| `enc-type` | file/rtsp_out | 0=HW, 1=SW |
| `sw-preset` | file/rtsp_out (SW) | 1=ultrafast … 10=placebo |
| `bitrate` | file/rtsp_out | bits/sec |
| `iframeinterval` | file/rtsp_out | int |
| `profile` | file/rtsp_out | H264: 0=Baseline, 2=Main, 4=High; H265: 0=Main, 1=Main10 |
| `output-file` | file | `out.mp4` |
| `rtsp-port`, `udp-port` | rtsp_out | 8554, 5400 |
| `msg-conv-config` | msgbroker | `dstest5_msgconv_sample_config.txt` |
| `msg-conv-payload-type` | msgbroker | 0=DEEPSTREAM, 1=DEEPSTREAM_MINIMAL, 256=RESERVED, 257=CUSTOM |
| `msg-conv-msg2p-lib`, `msg-conv-comp-id` | msgbroker | Custom payload lib |
| `msg-broker-proto-lib` | msgbroker | `libnvds_kafka_proto.so` / `_amqp_proto.so` / `_azure_proto.so` / `_redis_proto.so` |
| `msg-broker-conn-str` | msgbroker | `host;port;topic` |
| `topic` | msgbroker | |
| `msg-broker-config` | msgbroker | Optional proto-specific cfg (`cfg_kafka.txt`) |
| `disable-msgconv` | msgbroker | 1 when using a separate `[message-converter]` |

## `[message-converter]` / `[message-consumer*]`

| Key | Group | Notes |
|-----|-------|-------|
| `enable` | both | |
| `msg-conv-config` | converter | |
| `msg-conv-payload-type` | converter | |
| `msg-conv-msg2p-lib`, `msg-conv-comp-id` | converter | |
| `proto-lib` | consumer | |
| `conn-str` | consumer | `host;port` |
| `config-file` | consumer | Broker config |
| `subscribe-topic-list` | consumer | semicolon |
| `sensor-list-file` | consumer | Maps named sensor → index |

## `[nvds-analytics]`

| Key | Notes |
|-----|-------|
| `enable` | 0/1 |
| `config-file` | `config_nvdsanalytics.txt` |

## `[pre-process]` / `[secondary-pre-process]`

| Key | Notes |
|-----|-------|
| `enable` | 0/1 |
| `config-file` | `config_preprocess.txt` |

## `[tests]`

| Key | Notes |
|-----|-------|
| `file-loop` | 0/1 — loop file sources back to start |
| `pipeline-recreate-sec` | int — periodic restart for soak |

## Jetson vs dGPU cheatsheet

| Concern | dGPU | Jetson |
|---------|------|--------|
| Default display sink | `type=2` EglSink (`nveglglessink`) | `type=7` nv3dsink, or `type=5` nvdrmvideosink (no X) |
| `nvbuf-memory-type` | 0 (default = device) | 4 (surface-array) |
| `cudadec-memtype` | 0 (device) | n/a (NVMM) |
| Hardware encoder | `enc-type=0` available | `enc-type=0` available |
| `gie-unique-id` numbering | same | same |

## Numeric constants quick-reference (copy/paste)

- Realtime ~30 fps: `batched-push-timeout=33333`
- Realtime ~60 fps: `batched-push-timeout=16667`
- File/perf: `batched-push-timeout=40000`
- Block-forever: `batched-push-timeout=-1` (only for offline analysis runs)
- RTSP jitter: `latency=100` (`source.latency`)
- RTSP reconnect: `rtsp-reconnect-interval-sec=10`, `init-rtsp-reconnect-interval-sec=5`, `rtsp-reconnect-attempts=4`
