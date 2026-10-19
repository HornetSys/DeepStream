# DeepStream Config Snippets

Use these snippets as targeted changes, not as a complete application config. Preserve the user's existing model, tracker, analytics, tiler, and sink settings unless the symptom points to them.

## RTSP Source Group

For `deepstream-app` RTSP sources:

```ini
[source0]
enable=1
type=4
uri=rtsp://camera.example/stream
latency=300
udp-buffer-size=2000000
rtsp-reconnect-interval-sec=30
rtsp-reconnect-attempts=-1
select-rtp-protocol=4
num-extra-surfaces=4
```

Notes:

- `select-rtp-protocol=4` means TCP-only. Use it to test whether UDP loss is causing frame drops.
- `select-rtp-protocol=0` allows UDP, UDP multicast, and TCP negotiation. Use it when UDP is required or to compare against TCP.
- `latency` is in milliseconds for RTSP jitter buffering.
- `udp-buffer-size` helps only UDP receive buffering. It does not fix decoder or downstream backpressure.
- `rtsp-reconnect-interval-sec=0` disables reconnect. Nonzero values force reconnect after no data is received for that many seconds.
- `rtsp-reconnect-attempts=-1` retries indefinitely; call out operational trade-offs before recommending it for production.

## nvurisrcbin Properties

When the user constructs GStreamer pipelines around `nvurisrcbin`, map the same concepts:

```bash
nvurisrcbin uri="$RTSP_URL" latency=300 drop-on-latency=true select-rtp-protocol=4 \
  rtsp-reconnect-interval=30 rtsp-reconnect-attempts=-1 num-extra-surfaces=4
```

Useful properties:

- `latency`: jitterbuffer size in milliseconds for RTSP streams.
- `drop-on-latency`: cap jitterbuffer growth at the configured latency.
- `udp-buffer-size`: UDP buffer size for RTSP streams.
- `select-rtp-protocol`: `0` for multi transport, `4` for TCP-only.
- `num-extra-surfaces`: add decoder surfaces when downstream elements are starved.
- `drop-frame-interval`: reduce decoded frame output rate when load shedding is acceptable.
- `dec-skip-frames`: skip-frame diagnostic available on `nvurisrcbin`; verify the installed plugin properties.
- `low-latency-mode`: use only when the stream structure and plugin support it.

## Streammux

For the default/legacy streammux with live RTSP sources:

```ini
[streammux]
batch-size=1
width=1920
height=1080
live-source=1
batched-push-timeout=40000
```

Set `batch-size` to the number of sources in the batch unless the application intentionally uses another batching strategy. Use a frame-period timeout in microseconds:

| FPS | `batched-push-timeout` |
| --- | --- |
| 15 | 66667 |
| 25 | 40000 |
| 30 | 33333 |
| 60 | 16667 |

For newer `nvstreammux`, verify current docs with `gst-inspect-1.0 nvstreammux` because some legacy properties such as width, height, and live-source may be deprecated or moved.

For dynamic source add/remove latency, first tune `batched-push-timeout` to the input frame period and make dummy sources match the camera FPS/resolution when they are used to keep the pipeline alive. If the installed DeepStream version supports it, test this forum-sourced setting:

```bash
export NVSTREAMMUX_ADAPTIVE_BATCHING=yes
```

Do not tell the user to change `nvstreammux batch-size` at runtime as a no-state-change fix; changing `batch-size` requires a state transition.

## Sinks

For live diagnostics and low-FPS display issues:

```ini
[sink0]
sync=0
qos=0
```

Use `sync=0` to prevent wall-clock rendering from masking upstream performance. Use `qos=0` when decodebin/sink QoS is dropping frames during startup or heavy load; make the trade-off explicit.

For RTSP output stale frames, use unique ports per running pipeline:

```ini
[sink0]
type=4
rtsp-port=8660
udp-port=5500
sync=0
```

## Decoder Load Shedding

Use these only when the user accepts reduced frame rate or quality:

```ini
[source0]
drop-frame-interval=2
```

For `nvurisrcbin`:

```ini
[source0]
dec-skip-frames=1
```

Never present load shedding as a no-cost fix. It is a diagnostic or an explicit product trade-off.
