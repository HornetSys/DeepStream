---
name: deepstream-audit-config
description: >-
  Use only when auditing or sanity-checking a supplied or pre-existing
  deepstream-app/deepstream-test5 config, performing an explicitly requested
  independent post-generation audit, or reviewing a supplied design/code change
  that affects DeepStream config parsing or runtime behavior. For config audits,
  run scripts/validate_config.py statically first; runtime and network probes
  require explicit approval. Do not activate during a creation-only turn: use
  deepstream-generate-config. Not for merely running a validated config, ONVIF
  discovery, NVBug triage, or nsys profiling;
  use deepstream-inspect-rtsp for diagnosis of a known RTSP URI and
  deepstream-profile-pipeline for profiling.
metadata:
  author: "Kumar Abhishek <abhishek@nvidia.com>"
  version: "1.0.6"
  tags:
    - deepstream
    - deepstream-app
    - audit-config
    - rtsp
    - gstreamer
  languages:
    - python
    - bash
  domain: video-analytics
  team: deepstream-sdk
  permissions:
    - file_read
    - file_write
    - shell
    - network
    - env
owner: "Kumar Abhishek"
service: "deepstream"
reviewed: "2026-08-11"
license: "CC-BY-4.0 AND Apache-2.0"
compatibility: >
  Requires DeepStream SDK with deepstream-app or deepstream-test5-app on PATH,
  Python 3.8+, and PyYAML for YAML configs. scripts/validate_config.py uses
  stdlib only aside from optional PyYAML; scripts/ntp_probe.py needs GI + pyds
  when --probe-rtsp/--probe-ntp is approved. Complements deepstream-generate-config.
data_classification: "internal"
---

# DeepStream Config Auditor

## When to Use This Skill

Audits existing DeepStream configs and proposed design changes. Creation belongs
to `deepstream-generate-config`, including its render-time validation. Activate
this auditor afterward only when the user explicitly requests an independent
audit or supplies the generated config in a later request.

Typical triggers are:

- A supplied config needs correctness, use-case, or runtime-readiness review.
- A pasted config snippet needs the same static audit.
- A design or code change alters DeepStream config parsing or behavior.

Do not activate this auditor during a creation-only turn.

### Quickstart

From this skill directory, exercise the static path without DeepStream runtime
probes:

```bash
python3 scripts/validate_config.py evals/files/app_rtsp_bad.txt \
  --intent "4 production RTSP cameras"
```

Expected: exit `1` with an `ERROR` because this fixture intentionally contains a
batch-size mismatch. The command demonstrates finding and reporting a bad config;
it is not a setup smoke test.

## Inputs

Require either:

- one existing `.txt`, `.yml`, or `.yaml` config (path or pasted content) plus
  the user's intended use case; or
- a supplied design/code patch or proposal, the affected config semantics, and
  target DeepStream SDK/runtime context.

Do not invent an input or claim config-validation results without config content.

## Prerequisites

Static `.txt` parsing needs Python 3.8+. YAML needs PyYAML. Approved probes must
run on the target DeepStream host/container with the required binaries, GPU,
network access, and credentials supplied through the environment or config.
If PyYAML is missing, offer `python3 -m pip install -r requirements.txt`; do not
install it without approval. GStreamer GI bindings and `pyds` come from the
target DeepStream environment and are not installed by that manifest.

## When NOT to use

- User needs a **new** config from requirements -> `deepstream-generate-config`.
  Do not load this skill or run `validate_config.py` in that creation turn unless
  the user explicitly requests a separate independent audit.
- User needs ONVIF discovery or unknown-camera URI resolution -> no bundled skill
  currently owns that workflow. For diagnosis of a known RTSP URI, use
  `deepstream-inspect-rtsp`.
- User wants nsys profiling -> `deepstream-profile-pipeline`.
- User wants NVBug triage or custom GStreamer element implementation -> no
  bundled skill currently owns those workflows. For DeepStream application or
  pipeline development through supported APIs, use `deepstream-dev`.
- User only wants to **run** a config that already passed validation -> run `deepstream-app -c` directly.

## Runtime capabilities (approval required)

Static validation is the default and runs no runtime subprocess or network probe.
After **explicit user approval**, this skill may:

| Capability | How | Side effects |
|------------|-----|--------------|
| Plugin / host readiness | `gst-inspect-1.0`, optional `nvidia-smi` | Local subprocess only |
| Source reachability | `--probe-sources` / `--probe-rtsp` | Outbound RTSP/HTTP; may use URI credentials from the config |
| NTP / RTCP SR | `scripts/ntp_probe.py` via `--probe-rtsp` / `--probe-ntp` | Outbound RTSP; needs GI + pyds |
| Sink / group smoke | `--probe-sinks` / `--probe-groups` | Local gst-launch / path checks |
| Exact-config app smoke | `--probe-pipeline` **and** `--confirm-pipeline-probe` | Runs `deepstream-app` or `deepstream-test5-app`: GPU/inference, possible file writes, broker messages, network I/O |

`--probe-pipeline` is a **bounded smoke test**, not nsys profiling. Always warn about
side effects and require both flags after operator approval.

## Instructions

Users speak in plain language ("validate this config for 4 RTSP cameras and
Kafka"). For config audits, translate to `validate_config.py`; do not require
users to compose probe flags. The agent must still disclose each proposed probe,
target, timeout, and side-effect class and request approval in plain language.

## Workflow

```markdown
- [ ] Step 1: Get config path, snippet, or design proposal
- [ ] Step 2: Infer use case from conversation + config (--intent)
- [ ] Step 3: Run offline `validate_config.py` first (no `--check-runtime` or `--probe-*`)
- [ ] Step 4: If static validation reports any ERROR, STOP: report the blockers, skip runtime probes, and do not ask for probe approval. Otherwise, if runtime evidence is needed, disclose the exact flags, target, timeout, and side effects and ask for one-shot approval
- [ ] Step 5: STOP with the static result if approval is denied, absent, or ambiguous. Run only explicitly approved probes. For an approved exact-config smoke, state the timeout/output paths/broker topics, then pass BOTH `--probe-pipeline` and `--confirm-pipeline-probe`
- [ ] Step 6: Run approved probes only, then report Go ahead / Go ahead with conditions / Do not use yet (2-3 lines)
- [ ] Step 7: If the user approves edits, perform at most 1 edit/revalidation retry: apply 1 minimal edit pass and run static validation 1 more time. If that revalidation still reports any ERROR, abort further edits and report the remaining errors. In all cases, do not repeat the edit/re-validation cycle automatically
```

For a design/code review without config content, do not run
`validate_config.py`; inspect only the supplied artifact using
[references/design_review.md](references/design_review.md) and
[references/rules.md](references/rules.md). The validator steps above apply only
to config audits.

1. **Input**: config file (`.txt`/`.yml`), pasted snippet, or design/code change.
2. **Run** (from this skill directory):

```bash
python3 scripts/validate_config.py <config-file> --intent "<user request>"
```

Run static/offline validation by default. It parses files and evaluates rules
without launching DeepStream/GStreamer/GPU tools or opening sockets. Add
`--check-runtime` or probes from
[references/probe_flags.md](references/probe_flags.md) only after the user
explicitly approves them. If the user explicitly requests an independent audit
of a generated config, pass the original request as `--intent`.

### Exact-config pipeline smoke gate

Do not infer approval from words such as "thorough" or "will this run." Before
executing, name all side-effect classes that apply (GPU/model loading, outbound
sources, output files, broker topics), choose a finite `--timeout-sec`, and get
an explicit yes. Then and only then run:

```bash
python3 scripts/validate_config.py <config> --intent "<request>" \
  --probe-pipeline --confirm-pipeline-probe --timeout-sec <N>
```

Afterward, report created files and broker/network effects. Do not delete output
or attempt rollback without approval; published broker messages cannot be
recalled.

3. **Report**: start with **Go ahead**, **Go ahead with conditions**, or **Do not
use yet**. Lead with the top 1-2 blockers, but state the total ERROR count and
include every remaining ERROR in a compact list. Every ERROR/WARN needs a
concrete fix suggestion.

4. **Edits**: follow [references/config_edits.md](references/config_edits.md).

### Design / code review

See [references/design_review.md](references/design_review.md) and [references/rules.md](references/rules.md).

## Available Scripts

| Script | Purpose | Key arguments |
|--------|---------|----------------|
| `scripts/validate_config.py` | Config audit + approval-gated runtime probes | `<config>`, `--intent`, `--usecase`, `--target-platform`, `--json`, approved `--check-runtime` / `--probe-*`, `--confirm-pipeline-probe`, `--timeout-sec` |
| `scripts/ntp_probe.py` | Bounded RTSP NTP / RTCP SR probe (called by validate_config) | `<timeout-sec>` (RTSP URI on stdin) |

### `validate_config.py` machine-readable output and exit codes

`--json` emits one object with `summary` (`errors`, `warnings`, `info`,
`usecases`) and `findings`; every finding contains `severity`, `area`, `message`,
and nullable `suggestion`. `--target-platform` accepts `auto`, `jetson`, or
`dgpu`.

| Exit | Meaning |
|------|---------|
| `0` | Validation completed with no `ERROR` findings; warnings may remain |
| `1` | Validation completed and found at least one `ERROR` |
| `2` | CLI/input error, including a missing config path or invalid probe confirmation |

## Guardrails

- Do not invent config keys; use parser-backed rules in [references/rules.md](references/rules.md).
- Do not treat sample configs as universal for every use case.
- Do not conflate static RTSP with `nvmultiurisrcbin` -- use source-list only for dynamic add/remove, sensor-id, or REST.
- Default to static validation first; run `--check-runtime` or runtime/network probes only after explicit user approval.
- Never pass `--probe-pipeline` without the Step 5 approval gate, `--confirm-pipeline-probe`, and a finite timeout.
- Do not run privileged commands; do not embed credentials in output; redact RTSP passwords in reports.

## Output format

Return one verdict line, the total error/warning counts, all ERRORs (top 1-2
first), and concrete fixes. After approved probes, also report the exact command,
timeout, side effects observed, and pass/fail evidence.

## Limitations

Static validation cannot prove source reachability, camera RTCP/NTP quality,
model loadability, display availability, broker delivery, or end-to-end
pipeline success. Those claims require the corresponding approved probe on the
target host.

## Troubleshooting

- Missing config: ask for a path or pasted snippet; do not invent a result.
- Missing runtime tools: report a condition and validate on the target host.
- Probe timeout: report bounded timeout as inconclusive; do not claim success.
- Credentialed URI: redact user/password in all output.

## Examples

See [references/examples.md](references/examples.md) for RTSP timestamp policy,
runtime probe, msgbroker/test5, demux, tracker, and default reporting examples.

## Additional resources

- Validation rules: [references/rules.md](references/rules.md)
- Probe flags: [references/probe_flags.md](references/probe_flags.md)
- RTSP/NTP policy: [references/rtp_ntp_policy.md](references/rtp_ntp_policy.md)
- Example findings: [references/examples.md](references/examples.md)
- Design review guidance: [references/design_review.md](references/design_review.md)
- Config generator (sibling): `deepstream-generate-config`
- Known-URI RTSP diagnostics (sibling): `deepstream-inspect-rtsp`
- Pipeline profiling (sibling): `deepstream-profile-pipeline`
