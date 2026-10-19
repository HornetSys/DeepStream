# Validator pitfalls

`scripts/validate.py` catches these common deepstream-app config mistakes:

- `streammux.batch-size != primary-gie.batch-size` (silent perf cliff).
- Model engine filename `_bN_` doesn't match `batch-size` (DeepStream silently rebuilds).
- `live-source=1` with `batched-push-timeout > 40000` (latency tax).
- `sink type=4` (RTSPStreaming) missing `codec` / `bitrate` / `rtsp-port` / `udp-port`.
- `sink type=6` (MsgConvBroker) missing `msg-conv-config` / `msg-broker-proto-lib` / `msg-broker-conn-str` / `topic`.
- Jetson + `sink type=2` (EglSink — dGPU only) — should be `nv3dsink`.
- `source-list.use-nvmultiurisrcbin=1` with `max-batch-size > num-source-bins`.
- `[tiled-display] enable=2` without exactly one sink having `link-to-demux=1`. For N→N sinks use `enable=0` (demux) with per-sink `source-id` instead.
- `[tiled-display] enable=2` with multiple `link-to-demux=1` sinks — only the first is kept.

Patch the **spec** and re-render; do not hand-edit generated configs.
