<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Bounded adapter development

This is a development path for a routed `ADAPTER_REQUIRED` model with a
complete type-level contract. It does not change the production route. Use the
same selected GPU, pinned model, fixture, and semantic oracle throughout; do
not promote a candidate using a diagnostic visual run.

Allow at most three candidate attempts, including the initial candidate. One
attempt is one candidate generation or repair, config generation, and
validation cycle (backend, runtime, structured, semantic, and visual gates).
A failed or aborted cycle still consumes an attempt; reruns and code edits do
not reset the counter. Stop after the third attempt and retain the evidence.
The detection KITTI path has its own stricter one-repair limit; it is not an
additional adapter retry budget.

## 1. Generate a candidate

Run `generate_adapter.py /data/route.json /data/adapter-candidate`. Its
`manifest-with-adapter.json` records `implementation_status=implemented` and
`validation_status=generated-untested`. Generated code is not runtime-ready.

## 2. Generate DeepStream configs — mandatory hard gate

**Run `generate_configs.py` (mandatory).** Pass the generated manifest,
`--allow-implemented-adapter`, an explicit `--engine` or `--onnx` input, and a
fresh output directory. The flag is allowed only in this bounded development
path; it never changes the route to `READY`. Verify that `nvinfer.txt` and
`pipeline-plan.json` exist before any task runner. The latter records the
validated graph contract, not an executable GStreamer configuration; the
task runner materializes that graph. Missing configs stop this
development attempt as `BLOCKED`. Do not skip this step even when the adapter
source exists.

## 3. Execute and validate the candidate

For classification, use the shipped runner with the configs from Step 2:

```bash
CAMPAIGN_ID="${CAMPAIGN_ID:-classification-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
build/.venv_optimum/bin/python \
  .claude/skills/deepstream-import-vision-model/scripts/multitask/deepstream/run_type_adapter_pipeline.py \
  --manifest /data/adapter-candidate/manifest-with-adapter.json \
  --nvinfer-config /data/generated/nvinfer.txt \
  --input /data/fixture.mp4 --oracle /data/oracle.json \
  --output-dir /data/evidence --campaign-id "$CAMPAIGN_ID"
```

Before the semantic or rendered-output gate, confirm that caller-supplied
media and oracle passed intake. Omit `--oracle` only for an explicitly
diagnostic run on an existing local model and real media; keep semantic
`UNTESTED` and do not promote. If the runner exits nonzero, times out, or
omits required evidence, retain its logs and `ADAPTER_REQUIRED` status.
Failed intake for certification is `BLOCKED`. After an executable runner
succeeds, gate backend parity, DeepStream runtime, structured output,
semantic oracle, and model-derived rendered output against the same fixture.

The [NCHW full-frame logits renderer](segmentation-logits-runner.md) is a
concrete mask/overlay path only for its exact tensor and resize contract. It
does not provide an oracle or automatic readiness. Other segmentation layouts,
pose, and depth remain `ADAPTER_REQUIRED` until an equivalent real-input
runner is implemented and tested. For those supported categories, a
goal-based development request may add a type-level runner after inspecting
its tensor, upstream-stage, and oracle contract within the three-attempt budget
above. An unshipped runner, a generated plan,
or a raw-tensor smoke cannot establish readiness.

On any failed gate, preserve completed and failed-gate evidence, remain
`ADAPTER_REQUIRED`, and stop without promotion. A Step 8 **status** report
may be produced from partial evidence, including an early stop before media
or oracle intake completes. Missing inputs and unexecuted gates remain
`BLOCKED`/`UNTESTED`; the report is not a functional or semantic `PASS`.
Record visual `PASS` only if an actual model-derived overlay passed its visual
gate on available media; otherwise visual remains `UNTESTED` or `FAIL` as
observed. If report tooling itself is unavailable, record that report failure
as an additional blocker; do not claim a PDF was produced. Follow
[report-generation.md](report-generation.md) for evidence and filenames.

## 4. Promote only passed evidence

Only after every applicable backend, DeepStream, structured, semantic, and
visual gate passes, run `validate_adapter_readiness.py` to author a tested
manifest. Rerun `route_task.py` against it. Resume production only when the
reroute returns `READY` with exit `0`; a generated or partly tested candidate
remains `ADAPTER_REQUIRED`.
