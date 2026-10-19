# DeepStream Inspect RTSP

Diagnostic skill for NVIDIA DeepStream and GStreamer RTSP ingest reliability issues.

## What It Does

- Triage packet loss, latency growth, jitter, disconnects, frame drops, stale frames, artifacts, decoder saturation, camera/network faults, EOS/recreate memory growth, and dynamic RTSP source churn.
- Produce likely cause, minimal test pipeline, RTSP source settings, decoder/network checks, and targeted DeepStream config changes.
- Keep troubleshooting focused on diagnostics and commands instead of production application generation.

## Main Files

- `SKILL.md` - trigger, workflow, boundaries, report shape.
- `references/diagnostic-playbook.md` - symptom matrix and source-backed decision rules.
- `references/command-templates.md` - copy-paste diagnostic commands.
- `references/deepstream-config-snippets.md` - targeted DeepStream config snippets.
- `evals/evals.json` - automated behavior checks.

## Safety

Ask users to redact RTSP credentials, tokens, private hostnames, customer names, and internal registry paths. Packet capture is optional and requires explicit approval because it can expose camera and network details.
