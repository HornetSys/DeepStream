# RTSP / NTP timestamp policy

Do not blindly require `attach-sys-ts-as-ntp=0` or `1`. Match intent:

| Intent | Typical policy |
|--------|----------------|
| Latency measurement / local synthetic RTSP | `attach-sys-ts-as-ntp=1` (consistent local latency math) |
| Production event-time / cross-camera alignment | `attach-sys-ts-as-ntp=0` when cameras send trustworthy RTCP SR/NTP |
| Missing or commented in RTSP config | **WARN** — ask user to choose explicitly |

DeepStream parser defaults `attach-sys-ts-as-ntp` to true. `deepstream-test5`
serializes `frame_meta->ntp_timestamp` into message metadata.

When the user asks whether the camera sends valid RTCP Sender Reports, run
`--probe-ntp` (with `--probe-rtsp`) on the target host and report whether
non-zero `nvds-ntp-sync` values were observed.
