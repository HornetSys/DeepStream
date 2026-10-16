# Worked Examples

End-to-end input → output examples. Each one shows the **input spec**, the **invocation**, and the **expected resolved values** (the post-overlay spec) so you can compare with the rendered config.

> Paths use `${DS_ROOT}` as shorthand for `/opt/nvidia/deepstream/deepstream`. `render.py` substitutes the literal value.
>
> The examples below use `./out_*.txt` or `./quickstart_generated.txt` so they
> copy-paste without privileged commands. Generated configs use absolute paths
> internally and run from any cwd.

---

## Example 0 — Quickstart (zero-input default)

User intent: "I'm new. Just give me something that runs."

Use **Mode A** from the skill entrypoint: render with `--out ./quickstart_generated.txt`, validate, then run `deepstream-app -c` on that path. Do not duplicate the command block here — the canonical steps live in `SKILL.md`.

What it produces:

- 4 streams from the bundled `samples/streams/sample_1080p_h264.mp4`
- `trafficcamnet` PGIE with `_b4_` engine file
- `NvDCF_perf` tracker
- 2×2 tiled display with OSD bounding boxes
- EglSink (or `nv3dsink` if you re-render with `platform: jetson` in a copied spec)

The bundled spec lives at `assets/templates/quickstart_spec.yaml` — copy and edit it to customise (`cp` it locally before editing so it stays editable across re-renders).

---

## Example 1 — 4-stream file perf

User intent: "Measure peak FPS for 4 file streams with trafficcamnet + NvDCF + the 2 vehicle SGIEs."

### Input spec

```yaml
deployment:
  app: classic
  format: txt
  platform: dgpu
  gpu_id: 0
  output_path: ./out_perf_4file.txt
profile: perf
sources:
  kind: multi_uri
  uris: ["file://${DS_ROOT}/samples/streams/sample_1080p_h264.mp4"]
  num_sources: 4
primary_gie:
  enable: true
  preset: trafficcamnet
tracker:
  enable: true
  preset: NvDCF_perf
secondary_gies: [vehicletypes, vehiclemake]
osd: { enable: false }
sinks: []   # use profile default (fakesink)
```

### Invocation

```bash
python3 scripts/render.py --spec spec1.yaml --out ./out_perf_4file.txt
python3 scripts/validate.py ./out_perf_4file.txt
deepstream-app -c ./out_perf_4file.txt
```

### Key resolved values

- `streammux.batch-size=4`, `streammux.batched-push-timeout=40000`, `streammux.live-source=0`
- `primary-gie.batch-size=4`, engine `_b4_`
- `tiled-display.enable=0`, `osd.enable=0`
- `sink0.type=1`, `sink0.sync=1`

---

## Example 2 — 4-stream RTSP latency

User intent: "Realtime latency test against 4 RTSP cameras."

### Input spec

```yaml
deployment:
  app: classic
  format: txt
  platform: dgpu
  gpu_id: 0
  output_path: ./out_latency_4rtsp.txt
profile: latency
sources:
  kind: rtsp
  uris:
    - rtsp://10.0.0.10:8554/cam0
    - rtsp://10.0.0.11:8554/cam1
    - rtsp://10.0.0.12:8554/cam2
    - rtsp://10.0.0.13:8554/cam3
  num_sources: 4
primary_gie:
  enable: true
  preset: trafficcamnet
tracker: { enable: false }
osd: { enable: false }
sinks: []   # use profile default (fakesink with show-latency)
```

### Key resolved values

- `streammux.live-source=1`, `batched-push-timeout=33333`, `batch-size=4`, `cache-buffer=0`, `align-first-buffer=0`, `drop-pipeline-eos=1`
- Per-source: `type=4`, `latency=100`, `rtsp-reconnect-interval-sec=10`, `select-rtp-protocol=4`
- `sink0.type=1`, `sync=0`, `nvdslogger=1`, `show-latency=1`, `show-fps=1`
- `tests.file-loop=1`, `tiled-display.enable=0`

---

## Example 3 — 30-stream perf (source-list)

User intent: "30 streams via nvmultiurisrcbin for a throughput benchmark."

### Input spec

```yaml
deployment:
  app: classic
  format: txt
  platform: dgpu
  gpu_id: 0
  output_path: ./out_perf_30.txt
profile: perf
sources:
  kind: source_list
  uris: ["file://${DS_ROOT}/samples/streams/sample_1080p_h264.mp4"]
  num_sources: 30
  use_nvmultiurisrcbin: true
primary_gie:
  enable: true
  preset: trafficcamnet
tracker: { enable: false }
osd: { enable: false }
sinks: []
```

### Key resolved values

- `source-list.num-source-bins=30`, `use-nvmultiurisrcbin=1`, `max-batch-size=30`
- `source-attr-all.type=3`, `num-sources=1`
- `streammux.batch-size=30`, `batched-push-timeout=40000`, `live-source=0`
- `primary-gie.batch-size=30`, engine `_b30_`
- `sink0.type=1`, `sync=1`

---

## Example 4 — single USB camera display on Jetson

User intent: "Show the V4L2 camera output with detection on my Jetson — I have a screen attached."

### Input spec

```yaml
deployment:
  app: classic
  format: txt
  platform: jetson
  gpu_id: 0
  output_path: ./out_display_v4l2.txt
profile: display
sources:
  kind: v4l2
  num_sources: 1
  v4l2_dev_node: 0       # /dev/video0
  camera_width: 1920
  camera_height: 1080
  camera_fps_n: 30
  camera_fps_d: 1
primary_gie:
  enable: true
  preset: trafficcamnet
tracker:
  enable: true
  preset: NvDCF_perf
osd: { enable: true }
sinks:
  - type: egl     # auto-mapped to nv3dsink (type=7) on jetson
```

### Key resolved values

- `source0.type=1` (CameraV4L2), `camera-v4l2-dev-node=0`
- `streammux.batch-size=1`, `batched-push-timeout=40000`, `live-source=1`, `nvbuf-memory-type=4`
- `tiled-display.enable=1`, `rows=1`, `columns=1`
- `sink0.type=7` (nv3dsink), `sync=1`, `nvbuf-memory-type=4`

---

## Example 5 — msgbroker (Kafka) with consumer (test5)

User intent: "Pipe detections to Kafka and listen for OTA commands."

### Input spec

```yaml
deployment:
  app: test5      # required because consumer is enabled
  format: txt
  platform: dgpu
  gpu_id: 0
  output_path: ./out_msgbroker_test5.txt
profile: latency
sources:
  kind: multi_uri
  uris: ["file://${DS_ROOT}/samples/streams/sample_1080p_h264.mp4"]
  num_sources: 4
primary_gie:
  enable: true
  preset: trafficcamnet
tracker:
  enable: true
  preset: NvDCF_perf
sinks:
  - type: fakesink          # keep a latency-friendly sink alongside the broker
  - type: msgbroker
    msg_conv_config: ${DS_ROOT}/sources/apps/sample_apps/deepstream-test5/configs/dstest5_msgconv_sample_config.txt
    proto_lib: ${DS_ROOT}/lib/libnvds_kafka_proto.so
    conn_str: "kafka.local;9092;ds-events"
    topic: ds-events
    payload_type: 0
    consumer:
      enable: true
      proto_lib: ${DS_ROOT}/lib/libnvds_kafka_proto.so
      conn_str: "kafka.local;9092"
      config_file: cfg_kafka.txt
      subscribe_topics: [cmd]
```

### Key resolved values

- Latency overlay applied → `streammux.live-source=1`, `batched-push-timeout=33333`, `batch-size=4`, etc.
- `sink0.type=1` fakesink with latency overlay
- `sink1.type=6` msgbroker with all 4 required keys
- `message-converter.enable=0` (since broker sink bundles converter; would only set 1 if a second broker sink were added with `disable-msgconv=1`)
- `message-consumer0.enable=1`, `subscribe-topic-list=cmd`
- Run command: `deepstream-test5-app -c ./out_msgbroker_test5.txt`

---

## Example 6 — Latency test from a file (source_list + nvmultiurisrcbin)

User intent: "Measure end-to-end latency from one of the bundled sample mp4s. I want the number to reflect what a real RTSP camera would give me, not the decoder racing ahead."

This is the case where the skill auto-promotes `kind=file` to `kind=source_list` and adds `simulate-fps-interval-ms=33`. Spec stays small because all four `nvmultiurisrcbin` hard-requirements (`sensor-id-list`, `sensor-name-list`, `http-ip`, `http-port`) are auto-filled by `render.py`.

### Input spec

```yaml
deployment:
  app: classic
  format: txt
  platform: dgpu
  gpu_id: 0
  output_path: ./out_latency_file.txt

profile: latency

sources:
  kind: source_list
  num_sources: 1
  uris:
    - file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
  use_nvmultiurisrcbin: true

primary_gie:
  enable: true
  preset: trafficcamnet

tracker: { enable: false }
osd: { enable: false }
tiled_display: { enable: false }
sinks: []  # profile default fakesink with show-latency=1
tests:
  file_loop: true
```

### Invocation

```bash
python3 scripts/render.py --spec spec6.yaml --out ./out_latency_file.txt
python3 scripts/validate.py ./out_latency_file.txt
export NVDS_ENABLE_LATENCY_MEASUREMENT=1
export NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1
deepstream-app -c ./out_latency_file.txt
```

### Key resolved values

- `source-list.use-nvmultiurisrcbin=1`, `num-source-bins=1`, `max-batch-size=1`, `low-latency-mode=1`
- `source-list.sensor-id-list=UniqueSensorId1`, `sensor-name-list=UniqueSensorName1` (auto-filled, count = `num-source-bins`)
- `source-list.http-ip=localhost`, `http-port=9000` (auto-filled — missing => nvmultiurisrcbin crashes)
- `source-attr-all.simulate-fps-interval-ms=33` (auto-injected: latency + source_list + `file://` URI)
- `source-attr-all.sensorID-padID-mapping=0`
- `streammux.live-source=1`, `batched-push-timeout=33333`, `batch-size=1`
- `sink0.type=1`, `sync=0`, `nvdslogger=1`, `show-latency=1`, `show-fps=1`

`render.py` will also print to stderr: `[render] latency profile + file source -> auto-set simulate-fps-interval-ms=33 in [source-attr-all]`. Surface that line to the user — it's why the measured latency is honest instead of artificially low.

---

## Example 7 — Per-source RTSP-out (auto-demux)

User intent: "I have 4 file streams. Run detection on all of them, then re-stream each one back out on its own RTSP port (8554/8555/8556/8557) so my downstream consumers can pick them up independently."

This is the **N sources → N sinks** demux pattern. The trick is **`tiled-display.enable=0`** (alias: `enable: demux`) — deepstream-app then auto-creates `nvstreamdemux` and binds each sink to a source by `source-id`. No `link-to-demux` flag involved.

### Input spec

```yaml
deployment:
  app: classic
  format: txt
  platform: dgpu
  gpu_id: 0
  output_path: ./out_demux_rtsp_out.txt

profile: perf

sources:
  kind: multi_uri
  uris:
    - file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
  num_sources: 4

primary_gie: { enable: true, preset: trafficcamnet }
tracker: { enable: true, preset: NvDCF_perf }
osd: { enable: true }

tiled_display:
  enable: demux   # alias for 0; deepstream-app auto-creates nvstreamdemux

sinks:
  - type: rtsp_out
    source_id: 0
    codec: 1
    enc_type: 0
    bitrate: 4000000
    rtsp_port: 8554
    udp_port: 5400
  - type: rtsp_out
    source_id: 1
    codec: 1
    enc_type: 0
    bitrate: 4000000
    rtsp_port: 8555
    udp_port: 5401
  - type: rtsp_out
    source_id: 2
    codec: 1
    enc_type: 0
    bitrate: 4000000
    rtsp_port: 8556
    udp_port: 5402
  - type: rtsp_out
    source_id: 3
    codec: 1
    enc_type: 0
    bitrate: 4000000
    rtsp_port: 8557
    udp_port: 5403

tests: { file_loop: true }
```

### Key resolved values

- `tiled-display.enable=0` (the demux alias)
- `sink0..3` each with `type=4` (RTSPStreaming), `source-id=0..3`, sequential `rtsp-port`/`udp-port`
- No `link-to-demux=N` on any sink — that's only for `enable=2` parallel-demux
- `streammux.batch-size=4`, `live-source=0`

`render.py` prints to stderr:

```text
[render] tiled-display.enable=0 + per-sink source_id => deepstream-app
  auto-creates nvstreamdemux. Each [sink<N>] binds to one demuxed source
  by source-id. No link-to-demux property needed (that is for enable=2
  'parallel demux' only).
```

Run + consume:

```bash
deepstream-app -c ./out_demux_rtsp_out.txt
# in 4 separate terminals (or one mosaic):
ffplay rtsp://<host>:8554/ds-test
ffplay rtsp://<host>:8555/ds-test
ffplay rtsp://<host>:8556/ds-test
ffplay rtsp://<host>:8557/ds-test
```

### Variant: parallel demux + tile (`enable: parallel_demux`)

If instead the user wants the **tile on screen AND** one source tapped out to file/RTSP simultaneously, use `tiled_display.enable: parallel_demux` (maps to `[tiled-display].enable=2`) and set `link_to_demux: true` on exactly the one sink that should tap a source:

```yaml
tiled_display: { enable: parallel_demux }
sinks:
  - type: egl                       # consumes the tiled output
  - type: file
    source_id: 0
    link_to_demux: true             # taps source 0 out of the demuxer
    codec: 1
    bitrate: 4000000
    output_file: source0.mp4
```

deepstream-app honours **only the first** `link-to-demux=1` sink and silently drops any additional ones — `validate.py` warns when it sees more than one. For the N→N case prefer plain `enable: demux` (the example above).

---

## Anti-examples (don't do this)

1. **Mismatched batch**: `streammux.batch-size=4` but `primary_gie.batch_size=1`. The validator will fail; PGIE accumulates a queue and you get a latency cliff.
2. **`live-source=1` with file URIs**: streammux drops the EOS and the pipeline never finishes. Validator warns.
3. **`max-batch-size=10` with 4 URIs in source-list**: nvstreammux waits for 10 slots forever. Validator fails.
4. **Engine `_b1_` with `batch-size=4`**: DeepStream silently rebuilds the engine for batch 4 every launch — slow startup, no error. Validator warns.
5. **Jetson + `sink type=2`**: EglSink is dGPU-only. Validator fails; pick `egl` in spec and let the overlay map to `nv3dsink` for Jetson.
6. **`simulate-fps-interval-ms` in `[source-list]`**: wrong section. deepstream-app's parser rejects with `WARN parse_source_list: Unknown key 'simulate-fps-interval-ms' for group [source-list]` and the keep-going parser then misaligns subsequent keys, cascading into `parse_source_list failed`. Move the key to `[source-attr-all]`. Validator errors out.
7. **`use-nvmultiurisrcbin=1` without `sensor-id-list` / `sensor-name-list`** (or with the wrong count): aborts at init with `ERROR parse_source_list:188: Mismatch in Sensor IDs provided and num-source-bins`. Both lists must have exactly `num-source-bins` semicolon-separated entries. `render.py` auto-fills them; validator errors out if missing or mis-counted.
8. **`use-nvmultiurisrcbin=1` without `http-ip` / `http-port`**: aborts with `terminate called after throwing an instance of 'std::logic_error' what(): basic_string: construction from null is not valid`. nvmultiurisrcbin always starts its REST endpoint — there's no way to disable it. Defaults: `http-ip=localhost`, `http-port=9000`. Validator errors out.
9. **`tiled-display.enable=2` (parallel demux) without `link-to-demux=1` on any sink**: deepstream-app errors at `create_pipeline` with `Error : sink for demux (use link-to-demux-only property) is not provided in the config file`. For the N→N pattern use `enable=0` (alias `enable: demux`) instead — demuxer is auto-created and `link-to-demux` is not needed. Validator errors out.
10. **`tiled-display.enable=2` with multiple `link-to-demux=1` sinks**: deepstream-app silently keeps the first and drops the rest (known limitation). For per-source N→N sinks use `enable=0` instead. Validator warns.
