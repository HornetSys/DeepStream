---
name: deepstream-generate-config
description: >-
  Use when generating or repairing DeepStream deepstream-app or
  deepstream-test5-app configs for RTSP/GStreamer sources, streammux, inference,
  or tracker pipelines. Not for independent audits.
metadata:
  author: "Kumar Abhishek <abhishek@nvidia.com>"
  version: "1.0.10"
  tags: "deepstream, deepstream-app, generate-config, rtsp, gstreamer"
  languages: "python, yaml, bash"
  domain: video-analytics
  team: deepstream-sdk
  owner: "Kumar Abhishek"
  service: "deepstream"
  reviewed: "2026-10-01"
license: "CC-BY-4.0 AND Apache-2.0"
---

# deepstream-app Config Generator

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

## Purpose

Generate validated `deepstream-app` and `deepstream-test5-app` configuration
files from a bounded requirements spec, with explicit output-path safeguards.

## When to Use This Skill

Use this skill when a user needs a new `deepstream-app` or
`deepstream-test5-app` config, wants a generated config repaired through its
source spec, or asks for creation-time tuning of sources, streammux, inference,
tracking, display, latency, RTSP, or message-broker settings.

## Prerequisites

- DeepStream SDK 9.2.x with `deepstream-app` or `deepstream-test5-app` on `PATH`.
- Python 3.10+ and the PyYAML version pinned in `scripts/requirements.txt`.
- A user-writable directory for the spec and generated output.

Verify parser keys before targeting another DeepStream release.

## Inputs

Accept one of: `--quickstart`, a YAML/JSON spec path, detailed prose that can be
mapped to a spec, or answers to the interactive questions. For non-quickstart
generation, source kind/location and intended sink are required; profile,
platform, inference, and tracker settings are optional.

When values conflict, use this precedence: supplied spec file, explicit user
arguments, unambiguous agent context, then documented defaults. This skill has
no persistent state file. Never infer credentials or retain secrets in a spec.

## When NOT to use

- User supplies a pre-existing config and asks to **audit** or sanity-check it →
  use deepstream-audit-config. The bundled `scripts/validate.py` is only the
  render-time linter for this skill's own generation/repair workflow.
- User already has a working config and only wants to **run** or **debug** it →
  do not re-render or claim an independent audit.
- User asks to fix an NVBugs ticket, profile with Nsight, or write custom GStreamer/C code → other skills.
- User names an invalid `profile` (anything other than `latency`, `perf`, `display`, `custom`)
  → reject and list valid values. Ask for at most 1 profile clarification; if
  the response is still invalid or absent, abort without rendering. Never pass
  an invalid value through to `render.py`.

## Instructions

- Select the operating mode from the inputs and workflow below.
- Write `--out` only to a user-writable path such as
  `./generated_config.txt` or `./quickstart_generated.txt`; generated configs
  use absolute paths internally.
- Mention the root-owned stock samples directory under
  `/opt/nvidia/deepstream/...` only when the user explicitly asks, and never
  run privileged commands on the user's behalf.
- On `PermissionError`, make at most 1 writable-path retry with an explicit
  alternate `--out` after surfacing `render.py`'s recovery message. If that
  retry fails, abort generation and report the error.

## Workflow

```markdown
- [ ] Step 1: Detect mode (A quickstart / B spec / C prose / D Q&A)
- [ ] Step 2: Gather requirements
- [ ] Step 3: Profile overlay (automatic in render.py)
- [ ] Step 4: Render via scripts/render.py
- [ ] Step 5: Validate via scripts/validate.py
- [ ] Step 6: Print deepstream-app -c <path>
```

When the host provides a `run_script` helper, invoke the bundled scripts rather
than reproducing their logic. Equivalent calls are:

```text
run_script("scripts/render.py", ["--spec", spec_path, "--out", output_path])
run_script("scripts/validate.py", [output_path])
```

Otherwise, use the `python3` commands shown below.

**Default: Mode D (interactive Q&A).** Do not ask "which mode?" — infer from the invocation:

| Signal | Mode |
|--------|------|
| "quickstart", "starter", "just run", "I'm new" | **A** — skip Q&A |
| `--spec`, pasted YAML, spec file path | **B** — skip Q&A |
| Detailed prose with ≥3 inferable fields | **C** — build spec, confirm assumptions |
| Anything else | **D** — [references/interactive_qa.md](references/interactive_qa.md) |

### Mode A — Quickstart

```bash
python3 scripts/render.py --quickstart --out ./quickstart_generated.txt
python3 scripts/validate.py ./quickstart_generated.txt
deepstream-app -c ./quickstart_generated.txt
```

Bundled spec: `assets/templates/quickstart_spec.yaml`. Worked examples: [references/examples.md](references/examples.md).

### Mode B — Spec file

User supplied a spec → Step 4.

### Mode C — Natural language

Build YAML per [scripts/spec_schema.yaml](scripts/spec_schema.yaml) and [references/natural_language_mapping.md](references/natural_language_mapping.md). List assumptions before rendering.

### Mode D — Interactive Q&A

Follow [references/interactive_qa.md](references/interactive_qa.md). First question must include the quickstart escape hatch.

### Step 3: Profile overlay

Automatic in `render.py`. Explain choices from [references/profiles.md](references/profiles.md).

### Step 4: Render

```bash
python3 scripts/render.py --spec <spec.yaml> --out <writable-output-path>
```

### Step 5: Validate

```bash
python3 scripts/validate.py <output_path>
```

On validation errors, make at most 1 targeted correction to the spec, re-render,
and validate 1 more time. If validation still fails, abort further corrections
and report the remaining errors; do not loop or hand-edit generated configs.

### Step 6: Run command

```bash
deepstream-app -c <output_path>          # deployment.app == classic
deepstream-test5-app -c <output_path>    # deployment.app == test5
```

Relative paths inside the config resolve relative to the config file's directory.

## Output Format

Return the generated `.txt`, `.yml`, or `.yaml` config path, any explicitly
requested inference/tracker sidecar paths, the validation result, and the exact
`deepstream-app -c` or `deepstream-test5-app -c` command. Do not claim success
unless validation passes. Before writing to an already-present output path,
show its path and obtain explicit confirmation; commands are proposals for
review, not permission to execute them.

## Examples

```bash
# Starter config
python3 scripts/render.py --quickstart --out ./generated_config.txt
python3 scripts/validate.py ./generated_config.txt

# Spec → config
python3 scripts/render.py --spec ./my_spec.yaml --out ./generated_config.txt
python3 scripts/validate.py ./generated_config.txt
```

More worked examples: [references/examples.md](references/examples.md).

## Available Scripts

| Script | Purpose | Arguments |
|--------|---------|-----------|
| `scripts/render.py` | Spec → config | `--spec <yaml>`, `--quickstart`, `--out <path>`, `--print-resolved-spec` |
| `scripts/render_infer.py` | Render a standalone nvinfer detector/classifier config; also imported by `render.py` | `--network-type`, `--batch-size`, model paths, `--out <path>` |
| `scripts/render_tracker.py` | Render a standalone NvDCF/NvSORT/IOU tracker YAML; also imported by `render.py` | `--algorithm`, supported tracker overrides, `--out <path>` |
| `scripts/validate.py` | Config linter | `<config-path>`, `--help` |

`--print-resolved-spec` still writes the config and, after a successful
render, emits the normalized/default-resolved spec to standard error.

## Limitations

- Parser keys and templates are validated for DeepStream 9.2.x only.
- `validate.py` checks generated structure; it cannot prove that models,
  cameras, brokers, drivers, or plugins will work at runtime.
- This workflow does not independently audit arbitrary pre-existing configs.

## Troubleshooting

| Error | Cause | Resolution |
|---|---|---|
| `PyYAML` import failure | Pinned dependency is absent | Install `scripts/requirements.txt` in the active environment. |
| `SyntaxError: f-string expression part cannot include a backslash` | An older skill revision used Python 3.12-only f-string syntax | Update to the compatibility-fixed skill revision; if updating is not possible, use Python 3.12+ for that older revision. |
| Permission denied | Output directory is not writable | Retry once with an explicit user-writable `--out`; then stop. |
| Invalid profile | Value is outside the four supported profiles | Ask once for `latency`, `perf`, `display`, or `custom`. |
| Validation failure | The spec rendered an inconsistent config | Correct the spec once, re-render and revalidate; report remaining errors. |

## Guardrails

- **`profile` must be** `latency`, `perf`, `display`, or `custom` only. Reject unknown values before render; `render.py` exits 2 on invalid profiles.
- Always render through `scripts/render.py` — never hand-write configs in chat.
- Use [scripts/spec_schema.yaml](scripts/spec_schema.yaml) for accepted spec fields and [references/reference.md](references/reference.md) for authoritative DeepStream config keys; do not invent keys.
- Never skip `validate.py`.
- Do not auto-promote `msgbroker` to test5 unless the user wants `message-consumer`.
- Default `--out ./generated_config.txt`; no privileged commands.

## Additional resources

- Spec schema: [scripts/spec_schema.yaml](scripts/spec_schema.yaml)
- Quickstart spec: [assets/templates/quickstart_spec.yaml](assets/templates/quickstart_spec.yaml)
- Config key reference: [references/reference.md](references/reference.md)
- Profile deltas: [references/profiles.md](references/profiles.md)
- Worked examples: [references/examples.md](references/examples.md)
- Validator pitfalls: [references/validator_pitfalls.md](references/validator_pitfalls.md)
