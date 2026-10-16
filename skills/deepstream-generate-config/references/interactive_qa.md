# Interactive Q&A script (Mode D)

Use the `AskQuestion` tool. Batch questions where possible. Stop early if the user
already answered. In **Q1**, include:

> *"Reply 'quickstart' at any time if you'd rather skip all questions and get a known-good default config."*

## Question order

1. **App + format + platform** (one `AskQuestion`, three questions):
   - `app`: classic (`deepstream-app`) or test5 (`deepstream-test5-app`)
   - `format`: txt or yml
   - `platform`: dgpu or jetson
2. **Profile**: `latency` | `perf` | `display` | `custom` only. Reject any
   other value (for example `turbo-max`) and ask for at most 1 profile clarification.
   If the response is still invalid or absent, abort without rendering.
3. **Sources**: `kind` ∈ {file, rtsp, v4l2, csi, multi_uri, source_list}, `num_sources`, URIs/devices. If `kind=rtsp`, ask about `rtsp-reconnect-interval-sec`.
   - **Auto-promote `kind=file` → `source_list` + `use_nvmultiurisrcbin: true` when `profile=latency`.** Explain simulate-fps pacing. Do **not** auto-promote for `perf`/`display`.
   - **For `kind=rtsp`, do NOT auto-promote to `source_list` / `nvmultiurisrcbin`** unless the user explicitly needs REST add/remove, sensor-id tagging, or a single bin for many RTSPs.
4. **Sinks** (multi-select): `fakesink`, `egl`, `file`, `rtsp_out`, `msgbroker`. Profile sets defaults; this overrides.
   - **Demux** (only if `num_sources > 1`): tiled vs per-source vs shared sink. Per-source → `tiled_display.enable: demux` and `source_id` per sink.
5. **Primary GIE**: `trafficcamnet` (default), `peoplenet`, or `custom`.
5b. **Generate `config_infer_primary.txt`?** Default no. Yes for custom ONNX/engine or threshold tuning.
6. **Tracker** on/off; preset ∈ {NvDCF_perf, NvDCF_accuracy, NvSORT, IOU, NvDeepSORT, MaskTracker}.
6b. **Generate tracker YAML?** Default no. Yes for NvDCF/NvSORT/IOU tuning overrides.
7. **Secondary GIEs**: offer `vehicletypes` + `vehiclemake` when relevant.
8. **MsgBroker** (if chosen): `proto_lib`, `conn_str`, `topic`, `payload_type`. `message-consumer` → force `app=test5`.
9. **Output path**: default `./generated_config.txt` via `--out`. `/opt/nvidia/deepstream/...` only if user explicitly wants stock samples dir (root-owned).

Do not ask about `batched-push-timeout`, `live-source`, `cache-buffer`, tiled-display grid, or `nvbuf-memory-type` — the profile overlay documented directly by `SKILL.md` sets those. Override via `streammux.overrides` if needed.

**Escape hatch**: "just use defaults" / "skip the rest" → fill from `assets/templates/quickstart_spec.yaml` and render.
