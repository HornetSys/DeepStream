---
name: deepstream-inspect-rtsp
description: "Use when diagnosing DeepStream or GStreamer RTSP ingest reliability issues: packet loss, latency, decoder saturation, artifacts, disconnects, frame drops, source churn, and camera or network faults."
metadata:
  author: "NVIDIA DeepStream Team <deepstream-sdk@nvidia.com>"
  version: "1.0.0"
  team: deepstream-sdk
  domain: deepstream
  languages:
    - en
  tags:
    - deepstream
    - rtsp
    - gstreamer
    - diagnostics
    - networking
    - latency
owner: NVIDIA CORPORATION
service: deepstream-sdk
reviewed: "2026-07-22"
license: CC-BY-4.0 AND Apache-2.0
compatibility: >
  Diagnostic guidance for DeepStream and GStreamer RTSP ingest. Use the
  installed DeepStream release and target platform as the authority for
  GStreamer properties and DeepStream configuration details before applying
  guidance in production.
allowed-tools: Bash(cat:/proc/device-tree/model) Bash(cat:/etc/nv_tegra_release) Bash(ls:*) Bash(which:*) Bash(gst-inspect-1.0:*) Bash(nvidia-smi:*) Bash(tegrastats:*) Bash(ip:*) Bash(ethtool:*) Bash(ss:*) Bash(ping:*) Bash(tracepath:*) Bash(uname:*)
---
# DeepStream Inspect RTSP

## Purpose

Diagnose RTSP ingest reliability in DeepStream and GStreamer pipelines. Keep v1 focused on triage, commands, and configuration changes that isolate packet loss, latency, decoder saturation, disconnects, frame drops, artifacts, source churn, and camera/network faults.

## Boundaries

This is a diagnostic skill, not a production pipeline builder.

- Do not write a complete DeepStream application, Service Maker app, model config, inference parser, tracker setup, or deployment manifest.
- Do not claim a DeepStream, GStreamer, camera firmware, or driver bug unless the evidence isolates it from network, camera, decode, mux, and sink behavior.
- Do not run write actions, installs, service restarts, sysctl writes, packet captures, camera changes, `deepstream-app`, or `gst-launch-1.0` automatically. Provide commands for the user to run instead.
- If the request is mainly version compatibility, collect the installed DeepStream/GStreamer/container versions and direct the user to the DeepStream SDK compatibility documentation or their release owner; this mono-repo package has no dedicated stack-inspection sibling skill.
- If the request is full pipeline construction, reroute to `deepstream-dev` or `deepstream-generate-pipeline`; if it is model onboarding, reroute to `deepstream-import-vision-model`.

## Intake

Collect or infer these fields. If at least symptom, platform, source count, and current pipeline/config are present, give a provisional diagnosis and put missing items under Verify.

| Field | Examples |
| --- | --- |
| Symptom | packet loss, latency growth, disconnects, EOS/recreate memory growth, stale frames, low FPS, decoder warnings |
| RTSP source count | 1 camera, 30 cameras, dynamic add/delete |
| FPS/resolution | 25 FPS 1920x1080, 30 FPS 4K |
| Codec | H.264, H.265, MJPEG, unknown |
| Platform | Jetson Orin, dGPU model, container/bare metal |
| DeepStream/GStreamer version | DS 7.1, DS 9.1, container tag, `gst-launch-1.0 --version` |
| Current pipeline/config | `gst-launch` string, deepstream-app `[source#]`, `[streammux]`, `[sink#]` snippets |
| Network/camera context | wired/Wi-Fi, switch/NVR path, camera model, bitrate/GOP, TCP/UDP, NAT/VPN |
| Logs and counters | GStreamer warnings, late-buffer messages, FPS output, RSS/memory trend, NIC drops |
| Latency target | best effort, under 100 ms, under 1 second, no-drop recording |
| Decoder utilization | `nvidia-smi dmon`, `tegrastats`, NVDEC percentage, CPU decode suspicion |
| Packet trace availability | none, can run packet capture with approval, existing pcap/log excerpt |

Tell the user to redact RTSP credentials, customer hostnames, private IP ranges if needed, and tokens before pasting logs or URLs. Keep examples credential-free.

## Instructions

1. Read `references/diagnostic-playbook.md` for the symptom matrix and source-backed tuning rules.
2. Read `references/command-templates.md` when the answer needs copy-paste checks or minimal pipelines.
3. Read `references/deepstream-config-snippets.md` when the user uses deepstream-app, nvurisrcbin, nvstreammux, or sink config.
4. Classify the issue by first failing stage: camera/network/RTP, jitterbuffer, decoder, streammux, inference/processing, sink/output, or lifecycle/EOS.
5. Start with the smallest reproducible path:
   - RTSP session only: `rtspsrc ... ! fakesink`
   - RTP depay/parse only
   - decode only: depay/parse -> `nvv4l2decoder` -> `fakesink`
   - DeepStream mux only: source/decode -> `nvstreammux` -> `fakesink`
   - full pipeline only after the smaller stage reproduces or clears
6. Compare transport, latency, buffering, and decoder behavior before changing application logic.
7. Return the mandatory report shape.

## Scripts

No deterministic helper scripts are bundled. Do not call `run_script(...)` for this skill; use the references and read-only shell commands only when autonomous mode is explicitly selected.

| Script | Purpose |
| --- | --- |
| None | This skill is reference-driven RTSP ingest diagnostic triage. |

## Mode

Default to client-mediated diagnosis: provide commands for the user to run on the target DeepStream machine or container and ask for pasted output.

Use autonomous mode only when the current environment is the target machine and the user clearly opts in with "auto-detect", "run the checks", "I am on the box", or similar. In autonomous mode, run only read-only commands covered by `allowed-tools`; do not modify system state. Packet capture and pipeline execution remain client-mediated even in autonomous mode.

## Report Shape

Use exactly these headers for diagnostic answers:

```markdown
**Risk:**
**Likely Cause:**
**Evidence:**
**Minimal Test Pipeline:**
**RTSP Source Settings:**
**Decoder/Network Checks:**
**DeepStream Config Changes:**
**Next Step:**
```

Risk values:

| Risk | Use when |
| --- | --- |
| `LOW` | The likely fix is a bounded tuning/config change and evidence is clear. |
| `MEDIUM` | Missing data or multiple plausible stages remain. |
| `HIGH` | The current setup can accumulate latency/memory, drop frames heavily, or exceed decoder/network capacity. |
| `BLOCKED` | The request cannot be diagnosed without basic source details, or the requested action is outside this skill. |

## Mandatory Fast Paths

- Growing RTSP latency or memory while the pipeline keeps running: suspect jitterbuffer/backpressure first. Recommend a minimal `rtspsrc ... ! fakesink` isolation test, then tune `latency` and `drop-on-latency` with the trade-off made explicit.
- Packet drops/choppy remote RTSP viewing: compare TCP and UDP transport. For DeepStream RTSP output viewed in VLC, tell the user to force TCP on the client before changing the pipeline.
- Low FPS with low CPU/GPU utilization: suspect buffer starvation or decodebin/sink dropping. Verify component latency, then consider `num-extra-surfaces`, sink `qos=0`, and sink `sync=0`.
- Many cameras with one GPU/Jetson: compute aggregate decode load from stream count x resolution x FPS x codec and verify against the actual platform; do not quote hardware capacity from memory.
- Pixel corruption, smeared snapshots, or artifacts under high stream count: treat packet loss and decoder saturation as linked hypotheses. Verify packet loss and decoder utilization before changing inference settings.
- Repeated disconnects: add reconnect settings for DeepStream source bins, verify camera/network path, and compare `select-rtp-protocol=4` TCP-only against UDP/multi transport.
- EOS/recreate growth that appears only after replacing file/videotest input with RTSP: isolate rtspsrc and jitterbuffer behavior before attributing the growth to Python garbage collection or a DeepStream leak.
- Dynamic add/remove source latency with `nvstreammux`: keep this skill at diagnostic/config guidance. Check `batched-push-timeout`, active sink-pad count, dummy-source FPS/resolution, and forum-sourced `NVSTREAMMUX_ADAPTIVE_BATCHING=yes`; explain that changing `nvstreammux batch-size` requires a state change.

## Execute

1. If the request asks for full app construction, model onboarding, inference config authoring, tracker tuning, deployment, or broad capacity planning, enforce the boundary and reroute rather than writing code.
2. If the request is an RTSP ingest reliability problem, collect or infer the intake fields.
3. Read the relevant reference files before returning settings or commands.
4. Give the mandatory report with one minimal test pipeline and one next step that can confirm or falsify the likely cause.

## Quality Bar

- Prefer the smallest reproducible test over broad advice.
- State trade-offs for every buffering, transport, load-shedding, or reconnect setting.
- Do not quote hardware stream capacity from memory; tell the user how to verify the actual platform.
- Keep RTSP credentials out of commands and output.
- Treat `BENCHMARK.md` as NVCARPS-generated evaluation evidence; never edit it manually.

## Examples

```text
My DS 7.1 app on Orin NX ingests one H.265 RTSP camera. After every EOS/recreate cycle RSS grows. What should I test?
```

```text
Thirty 1080p30 H.264 RTSP cameras randomly disconnect and recover slowly in deepstream-app. Suggest source config and checks.
```

```text
My RTSP cameras show high jitter and frame drops. Current source config has latency=100 and UDP transport. What should I change?
```

```text
Use autonomous mode on this Jetson container and check whether the RTSP decode path is saturated.
```
