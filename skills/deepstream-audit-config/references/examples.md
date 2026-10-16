# DeepStream Config Validator Examples

## RTSP Config With Missing Timestamp Policy

Finding:

```text
WARN [timestamp]: RTSP sources found but streammux.attach-sys-ts-as-ntp is not
explicitly set. Parser default is 1 (system time). That is OK for latency/local
RTSP, but production event-time/cross-camera ordering usually needs 0 with valid
RTCP sender reports.
```

Suggested response:

```markdown
For this use case, choose one timestamp policy explicitly:

- Latency test / localhost nvstreamer: add `attach-sys-ts-as-ntp=1`.
- Real camera event-time: add `attach-sys-ts-as-ntp=0` and verify the camera
  sends valid RTCP sender reports.
```

## Static RTSP Using nvmultiurisrcbin

Finding:

```text
WARN [source-list]: use-nvmultiurisrcbin=1 with RTSP inputs. This is useful for
dynamic add/remove or sensor-id metadata, but static RTSP sets are simpler as
plain [sourceN] type=4 groups.
```

Suggested response:

```markdown
Keep `nvmultiurisrcbin` if you need REST add/remove or sensor ID metadata.
Otherwise use one `[sourceN] type=4` group per camera to reduce REST-port and
shared-attribute complexity.
```

## SGIE References Missing Upstream GIE

Finding:

```text
ERROR [gie]: secondary-gie0.operate-on-gie-id=4 does not reference an enabled
GIE unique ID.
```

Suggested response:

```markdown
Set `operate-on-gie-id` to the enabled PGIE's `gie-unique-id`, typically `1`, or
enable the upstream GIE with ID 4.
```

## Demux Misuse

Finding:

```text
ERROR [demux]: tiled-display.enable=2 requires exactly one enabled sink with
link-to-demux=1. For N sources to N sinks, use tiled-display.enable=0 and per-sink
source-id instead.
```

Suggested response:

```markdown
`enable=2` is the parallel-demux path and only one demux-linked sink is honored.
For per-source output, set `[tiled-display] enable=0` and create sinks with
`source-id=0..N-1`.
```

## Default User-Facing Report

Keep the default final response short:

```markdown
Go ahead with conditions: this config is usable for the stock display sample, but the current host is headless so EglSink will fail unless DISPLAY/X/Xvfb is available.
For production RTSP camera use, set live-source=1, choose attach-sys-ts-as-ntp explicitly, and add RTSP reconnect/TCP settings before deployment.
```

For a blocked config:

```markdown
Do not use yet: source0 points to a missing media file, so deepstream-app will fail before decode.
Fix the URI or copy the media file to the target host, then re-run validation.
```

When fixes are available:

```markdown
Go ahead with conditions: the config has no blocking errors, but production RTSP use should set live-source=1 and add reconnect/TCP settings.
I can apply those safe RTSP config updates and re-run validation. Should I update the config file?
```

Only provide detailed group-by-group findings when the user asks for details.

## Applying Updates After Approval

If the user approves, edit only the supplied config file. Keep the patch minimal.

Example approved edits for a production RTSP camera config:

```ini
[source0]
latency=100
select-rtp-protocol=4
rtsp-reconnect-interval-sec=10
init-rtsp-reconnect-interval-sec=5
rtsp-reconnect-attempts=4

[streammux]
live-source=1
attach-sys-ts-as-ntp=0
```

If timestamp intent is unclear, do not choose automatically. Ask:

```text
Should timestamps use system time for local latency testing (`attach-sys-ts-as-ntp=1`) or camera event-time via RTCP Sender Reports (`attach-sys-ts-as-ntp=0`)?
```

## Detailed Report Template

```markdown
## Result
Validation found N error(s), M warning(s), K info note(s).

## Errors
- ...

## Warnings
- ...

## Design Concerns
- ...

## Suggested Fixes
- ...

## Verification
Run:
`deepstream-app -c <config>`

Check:
- RTSP reconnect logs.
- `frame_meta->ntp_timestamp` behavior if test5/msgbroker is enabled.
- FPS/latency logs if this is a latency or perf config.
```

## Explicit Independent Audit After Generation

A creation-only request belongs to `deepstream-generate-config`; do not invoke
this audit skill or `validate_config.py` in the same turn. Use the commands below
only when the user explicitly asks for an independent audit or later supplies the
generated config for review.

If `deepstream-generate-config` previously produced `<generated-config>` after
the user asked for "4 RTSP cameras, low latency, Kafka msgbroker, run with test5",
audit it with the conversation text:

```bash
python3 scripts/validate_config.py <generated-config> \
  --intent "4 RTSP cameras, low latency, Kafka msgbroker, run with test5"
```

After explicit approval, add the bounded current-host readiness check:

```bash
python3 scripts/validate_config.py <generated-config> \
  --intent "4 RTSP cameras, low latency, Kafka msgbroker, run with test5" \
  --check-runtime --timeout-sec 5
```

Only probe RTSP sources when the current host/VPN can reach them:

```bash
python3 scripts/validate_config.py <generated-config> \
  --intent "4 RTSP cameras, low latency, Kafka msgbroker, run with test5" \
  --probe-rtsp --timeout-sec 5
```

To functionally test enabled sinks on the current host/container:

```bash
python3 scripts/validate_config.py <generated-config> \
  --intent "display config using EglSink" \
  --probe-sinks --timeout-sec 5
```

Expected EglSink failure on a headless host:

```text
ERROR [sink-probe]: sink0: EglSink cannot be functionally tested because no DISPLAY/WAYLAND_DISPLAY is set.
SUGGEST [sink-probe]: Run with X/Wayland/Xvfb or change this sink to fakesink for headless validation.
```

To check active groups such as PGIE, tracker, preprocess, and msgbroker
prerequisites:

```bash
python3 scripts/validate_config.py <generated-config> \
  --intent "trafficcamnet with tracker and display" \
  --probe-groups --timeout-sec 5
```

To run an exact-config bounded smoke test (after explicit approval):

```bash
python3 scripts/validate_config.py <generated-config> \
  --intent "trafficcamnet with tracker and display" \
  --probe-pipeline --confirm-pipeline-probe --timeout-sec 10
```

Expected failure when one active group cannot initialize:

```text
ERROR [pipeline-probe]: deepstream-app reported fatal/error lines during exact-config smoke run
SUGGEST [pipeline-probe]: Fix the reported runtime errors, then re-run validation.
```

Static prevention for a known tracker/runtime failure:

```text
ERROR [tracker]: source0: tracker is enabled with URI_MULTIPLE (type=3) RTSP and num-sources=4; this can fail at runtime with motFrame->numBuffers must be 1 when numTransforms is 1.
SUGGEST [tracker]: Use separate RTSP source groups (type=4 per source), use source-list/nvmultiurisrcbin for dynamic RTSP, or disable tracker for this URI_MULTIPLE RTSP pattern.
```

To verify whether RTSP cameras send usable RTCP Sender Reports for camera NTP
timestamps, run the stronger NTP probe on the target host/container:

```bash
python3 scripts/validate_config.py <generated-config> \
  --intent "4 RTSP cameras, event-time timestamps, Kafka msgbroker, run with test5" \
  --probe-rtsp --probe-ntp --timeout-sec 10
```

Shortcut for project validation that an RTSP source is reachable and carries
camera NTP/RTCP Sender Report data:

```bash
python3 scripts/validate_config.py <generated-config> \
  --intent "validate RTSP source and camera NTP timestamp flow" \
  --probe-rtsp --timeout-sec 10
```

Expected success:

```text
INFO [ntp-probe]: source0: RTCP Sender Report/NTP sync observed
```

Expected failure when the server does not expose sender-report NTP:

```text
WARN [ntp-probe]: source0: RTSP was probed but no valid RTCP Sender Report/NTP sync was observed
SUGGEST [ntp-probe]: If camera event-time is required, configure the camera/server to send RTCP Sender Reports or keep attach-sys-ts-as-ntp=1
```

## Structured Output For Patch Suggestions

```bash
python3 scripts/validate_config.py app.txt --intent "filesrc latency" --json
```

The JSON includes `summary.usecases` plus findings with optional `suggestion`
fields. Use it to prepare a config patch, but ask the user before modifying
their config.
