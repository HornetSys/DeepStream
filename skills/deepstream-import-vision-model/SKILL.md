---
name: deepstream-import-vision-model
description: >
  Import detection, classification, segmentation, pose, and depth models into
  DeepStream with task routing and evidence-gated validation.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: "Tushar Khinvasara <tkhinvasara@nvidia.com>"
  owner: "Tushar Khinvasara <tkhinvasara@nvidia.com>"
  service: "deepstream"
  version: "1.6.0"
  reviewed: "2026-09-16"
  team: deepstream-sdk
  tags:
    - deepstream
    - tensorrt
    - object-detection
    - classification
    - segmentation
    - pose
    - depth
    - import-vision-model
  languages:
    - bash
    - python
    - cpp
  domain: computer-vision
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# DeepStream Import Vision Model

## Use case

Detection has the established end-to-end path; classification, semantic
segmentation, 2D pose, and monocular
depth require model-and-fixture-specific runtime, semantic, and rendered-output
evidence before claiming readiness.

This category-readiness distinction must remain in regenerated publication
metadata; a planning evaluation is not runtime certification of all categories.

## Purpose

Import one supported visual model reproducibly. Preserve the native detector
path; require evidence-backed type-level adapters for other categories.

## Prerequisites

- Docker, NVIDIA Container Toolkit, a visible GPU, and DeepStream registry access.
  The caller must provide its verified immutable `@sha256:` repository digest
  as `DS_IMAGE`; record that exact digest in the evidence bundle. In execution
  mode, stop `BLOCKED` before container setup if it is absent or malformed;
  a dry run may leave it as an unverified prerequisite. Screen the declared
  category first: an unsupported task returns `route_status=UNSUPPORTED`
  regardless of whether `DS_IMAGE` is present.
- A local model artifact, or a public/versioned Hugging Face or NGC source.
- Credentials are optional for local artifacts and public downloads; private
  model or registry access requires caller-provided credentials via environment only.
- A declared supported task and its tensor/preprocess/postprocess contract.
- Real media and a semantic oracle conforming to
  [references/semantic-oracles.md](references/semantic-oracles.md) before any
  certified semantic or overall `PASS`. See Step 7 below for the diagnostic lane.
- Explicit caller approval before deserializing a supplied TensorRT engine.

## Limitations

Use the phase table below as the reference checklist: read each linked contract
before its phase. If a required reference is absent from the installed package,
stop `BLOCKED`; do not reconstruct commands or validation rules from memory.

**Current scope:** route exactly object detection, image classification,
semantic segmentation, 2D pose/keypoints, and monocular depth. Reject other
categories before conversion, without running setup, cleanup, or model tools.
Detection keeps its parser, KITTI, benchmark, and report path. Classification
may use native parsing or the embedded classification reference runner.
NCHW full-frame segmentation logits may use
the narrow shipped renderer; other segmentation layouts, pose, and depth use
generated type-level hooks and remain `ADAPTER_REQUIRED` until an equivalent
concrete DeepStream runner exists and real-input runtime, semantic, and visual
validation passes. Only the readiness validator may mark an adapter `tested`;
caller metadata and environment variables cannot override this gate.

## Instructions

### Route the task first

There are two distinct gates, in this order: screen the confirmed declared
category without tools or writes; only for a supported category, validate
`DS_IMAGE`, start the persistent container, and complete setup/preflight.
Then run Step 0 to inspect the actual local artifact and its contract before
conversion. A supported declared category is not a `READY` artifact route.

```bash
# Inside the persistent container after setup.sh and preflight have passed.
PYTHONDONTWRITEBYTECODE=1 build/.venv_optimum/bin/python \
  .claude/skills/deepstream-import-vision-model/scripts/model/route_task.py \
  /data/model.onnx --task semantic-segmentation \
  --metadata /data/task-metadata.json --output /data/route.json
```

These exit values belong only to `route_task.py`:

- Exit `0`: `READY`; continue the production workflow.
- Exit `2`: invalid input; stop and report the validation error.
- Exit `3`: unsupported category; stop without conversion. Return final status
  `BLOCKED`, `route_status=UNSUPPORTED`, all execution gates `UNTESTED`, and
  identify the unsupported category in `next_action`.
- Router exit `4` (`ROUTER_EXIT_4`): inspect `capability_status`. For `BLOCKED`,
  stop and report the
  missing source, task, or contract. For `ADAPTER_REQUIRED`, stop the production
  workflow; enter only the bounded adapter-development path below when the
  manifest contains a complete type-level contract.
  Missing or unknown `capability_status` is `FAIL`, never permission to proceed.
  Report that router failure as `route_status=FAIL`.

Treat every other router exit as unexpected: stop, preserve stdout/stderr, and
report `FAIL` with `route_status=FAIL`. Never interpret an undocumented exit as a usable route. Keep
phase-specific statuses in their referenced phase: engine build uses exit `5`
only when pinned metadata cannot resolve dynamic spatial dimensions. Preserve
that diagnostic as `BLOCKED`; all other phase statuses follow their directly
linked reference.

| Category | Execution family | Initial readiness |
|---|---|---|
| Object detection | Native parser, KITTI/structured detections, box rendering, and multi-stream benchmark | `READY` after source preflight |
| Image classification | Embedded runtime; primary frame or declared detector-owned ROI | `READY` only for a verified native classifier contract; otherwise `ADAPTER_REQUIRED` |
| Semantic segmentation | Embedded runtime; mask decoder, frame remap, and overlay | `ADAPTER_REQUIRED` until its adapter is tested |
| 2D pose/keypoints | Embedded runtime; full frame or declared person-detector ROI, keypoint decoder, and skeleton renderer | `ADAPTER_REQUIRED` until its adapter is tested |
| Monocular depth | Embedded runtime; depth decoder, numeric oracle contract, and heatmap overlay | `ADAPTER_REQUIRED` until its adapter is tested |

### Bounded adapter-development path

Require a complete input/output, preprocessing, postprocessing, upstream-stage,
media, and semantic-oracle contract. Follow [adapter-development.md](references/adapter-development.md):
generate the candidate; **run `generate_configs.py` (mandatory)** and verify
`nvinfer.txt`/`pipeline-plan.json`; execute backend, DeepStream, structured,
semantic, and visual gates; promote using `validate_adapter_readiness.py`;
then reroute Step 0. Readiness JSON must bind the same model/fixture to all
passed gates. Missing configs stop the run; failed gates never promote.
Resume production only at `READY` (exit `0`). Allow at most three candidate
attempts, including the initial candidate. One attempt is one candidate
generation or repair, config generation, and validation cycle; a failed or
aborted cycle still consumes an attempt. Reruns and code edits do not reset
the counter. Keep KITTI's separate one-repair limit. An early stop may produce
a Step 8 **status** report
from partial evidence without complete media/oracle intake; retain missing
gate statuses and never claim a functional or semantic `PASS`.

Audio, instance/prompt/memory segmentation, SAM2, stereo, optical flow, 3D/6D
pose, OCR/LPR, Re-ID/retrieval, and generic custom tensors are outside this
integration scope and must return `BLOCKED`/unsupported without conversion.
The [fixed-pair diagnostic](references/stereo-fixed-pair.md) is experimental;
it does not synchronize changing stereo video or permit a stereo `READY` claim.

## Inputs

Required:

- Local artifact or pinned HF/NGC source, supported task, and tensor/preprocess/
  postprocess contracts.
- Real media and a validated [semantic oracle](references/semantic-oracles.md)
  for certification; Step 7 permits diagnostics without an oracle.

Optional: labels, batch/profile limits, ROI stages, environment constraints,
and adapters. Tested adapters require validator-authored readiness evidence;
never invent contracts or treat fields as execution evidence.

## Model choice

Read [references/model-choice.md](references/model-choice.md) before any
mutation; retain default/custom choice and intake gates. Dry runs make no tool
calls or writes: return only `model_choice`, `source`, `task`, `steps_to_run`,
and `estimated_outputs`; supplied paths remain unverified.

## Pipeline Overview

Execution order:

1. Confirm model/task; dry runs emit only the plan and stop. Custom remote
   metadata inspection is read-only and requires final confirmation.
2. Screen the declared category before setup, cleanup, downloads, or writes.
   Unsupported tasks stop; an undetermined task is `BLOCKED` until confirmed.
3. For supported categories, validate `DS_IMAGE`, start the persistent
   container, and pass setup/preflight before Step 0.
4. Route the actual artifact in Step 0 before export/build. Remote sources
   first pin an immutable HF commit/NGC version and materialize artifact/config
   in Steps 1–2; never select a task from its name.
5. Complete Step 3A before deserializing any supplied engine.
6. Continue only through passed execution gates. Step 8 reports the observed
   result, including blocked/failed cases; it never substitutes for Steps 0–7.

The order for all sources is **declared-category screen → setup/preflight →
Step 0 artifact routing → Step 3A for supplied engines → deserialization**.
Neither `trtexec --loadEngine` nor DeepStream `nvinfer` may load a supplied
engine before Step 3A approval and artifact validation pass.

| Step | Phase | Reference | What it does |
|------|-------|-----------|--------------|
| Pre-route | Category screen, then setup/preflight | [references/runtime-environment.md](references/runtime-environment.md) | Reject unsupported declared categories before setup; create the shared venv only for supported requests |
| 0 | Artifact Route | [references/multitask/task-registry.md](references/multitask/task-registry.md) | After declared-category screening and setup/preflight, route a local artifact; remote sources first materialize in Steps 1–2 |
| 1–3 | Model Acquire | [references/model-acquire.md](references/model-acquire.md) | Resolve/browse/download a remote source, return through Step 0, then perform only the routed export or final artifact placement |
| 3A | Supplied Engine Approval | [references/engine-approval.md](references/engine-approval.md) | Approve and validate before deserialization; verify planned shapes/batches, then skip engine builds, not required measurements |
| 4–5 | Engine Build  | [references/engine-build.md](references/engine-build.md) | Build dynamic TRT engine, run trtexec BS=1 and BS=MAX_BS |
| 6 | DS Pipeline | Detection: [references/pipeline-run.md](references/pipeline-run.md); other categories: [references/multitask/pipeline-graph.md](references/multitask/pipeline-graph.md) | Build the category-aware graph, decoder, metadata writer, and renderer |
| 7 | Validation and adapter promotion | [references/multitask/validation.md](references/multitask/validation.md) and [references/adapter-readiness.md](references/adapter-readiness.md) | Validate parity, runtime, structured and rendered output; promote only a fully tested adapter |
| 8 | Evidence-backed report and 13-case campaign index | [references/report-generation.md](references/report-generation.md) | Retain the detection benchmark; report each executed or blocked application case and verify all 13 suite entries without treating report generation as a model test |

Proceed only between passed gates; never bypass model choice, supplied-engine
approval, or a KITTI/parity/semantic/visual stop. `ADAPTER_REQUIRED` stops
production; a complete contract permits only bounded development. At other
stops, abort and report the missing contract, hook, or evidence. Generated code
cannot claim runtime `PASS` without promotion evidence.
At Step 7, confirm that caller-supplied real media and a reviewed semantic
oracle passed intake before claiming certified semantic or overall `PASS`.
If the model and suitable media exist but the oracle does not, a bounded
diagnostic backend, DeepStream, structured-output, and rendered-output attempt
is allowed; record its actual gate outcomes, leave semantic `UNTESTED`, issue
only a Step 8 status report, and do not promote the adapter or claim functional `PASS`.
Do not attempt an application run with a missing model, absent required
upstream component, or invalid stereo pair/calibration.

## Container runtime and preflight

Read [references/runtime-environment.md](references/runtime-environment.md)
completely before setup or execution. Use one persistent, digest-pinned
container and one selected GPU for every phase. Preflight is a hard stop on
missing bindings, selected media, CUDA support, or the required visual encoder.
Credentials enter through environment variables, never command arguments.
Windows mounts: [references/windows.md](references/windows.md).

## Output Format

Artifacts include KITTI detections for detection runs, category-specific
structured outputs for other tasks, rendered frames/video, hashed evidence,
and Step 8 Markdown/HTML/PDF reports. Publication metadata must preserve
this output scope and the category-readiness disclaimer in Use case.

For a dry run, return the five-field plan object defined under Model choice
instead of the production result below; a dry run has no execution statuses.

Return the final result as one JSON object with these fields in order:

```json
{
  "status": "PASS | FAIL | BLOCKED | ADAPTER_REQUIRED | UNTESTED",
  "task": "detection | classification | segmentation | pose | depth | unsupported",
  "model": "resolved model identifier or local artifact",
  "route_status": "READY | BLOCKED | ADAPTER_REQUIRED | UNSUPPORTED | FAIL",
  "pipeline_status": "PASS | FAIL | BLOCKED | UNTESTED",
  "semantic_status": "PASS | FAIL | UNTESTED",
  "visual_status": "PASS | FAIL | UNTESTED",
  "capacity_status": "PASS | FAIL | BLOCKED | UNTESTED",
  "evidence": {"manifest": "path", "structured": "path", "rendered": "path", "report": "path"},
  "next_action": "empty on PASS; otherwise the required corrective action"
}
```

Use listed enums and empty strings for unavailable evidence paths; explain
gaps in `next_action`. Never omit fields or mark unexecuted gates `PASS`.
A diagnostic run with semantic `UNTESTED` and all
other applicable gates passing has final `status=UNTESTED` and
`next_action=requires_semantic_oracle_for_certification`; a failure, missing
input, or unpromoted adapter retains `FAIL`, `BLOCKED`, or `ADAPTER_REQUIRED`
respectively rather than being hidden by that diagnostic status.
Capacity is a separate measured claim. Detection retains its multi-stream
benchmark; use the [task-capacity contract](references/benchmark-evidence.md)
for other categories. Report `UNTESTED` unless a task-compatible runner
measures at least two logical streams (paired inputs count as one stereo stream)
against a declared FPS target. Never infer capacity from TensorRT throughput or
one-stream EOS.

After final model/task/media/oracle confirmation, validate `MODEL_NAME` as a
single safe path component, then create the per-model tree using the quoted
commands in [references/artifact-layout.md](references/artifact-layout.md).
Never dump files flat or interpolate an unvalidated model name into a path.

## Critical Rules

1. **Engine naming** — always `{model}_dynamic_b{MAX_BS}.engine`. Never bare `model_dynamic.engine`.
2. **batch_size == num_streams** — in applicable DS benchmark runs, `batch-size` and stream count are always equal.
3. **Fixed benchmark logs** — `trtexec_b1.log`, `trtexec_b${MAX_BS}.log`, `ds_s${N}_run1.log`, `ds_s${N}_run2.log`; no timestamps except documented engine-build diagnostics.
4. **Detection parser zero-init** — require `NvDsInferObjectDetectionInfo obj = {};` before compilation; bare `obj;` leaves `rotation_angle` uninitialized.
5. **Detection KITTI gate** — zero frames or <90% detection rate stops benchmarks. Apply at most one evidenced repair and one rerun, in the order and with the audit record specified in [pipeline-run.md](references/pipeline-run.md#6g-kitti-dump--verify-detections-programmatically). Preserve failures; never change model code, labels, thresholds, or media to force a pass.
6. **Shared venv** — `build/.venv_optimum` reused across all models. Never create per-model venvs.
7. **trtexec `--noDataTransfers`** — GPU-only compute matches DeepStream's GPU-to-GPU data flow.
8. **Report HTML+PDF** — use [references/report-generation.md](references/report-generation.md) and the canonical converter for benchmark, validation, or status reports. Never write a custom HTML generator or call `wkhtmltopdf` directly.
9. **Category boundary** — accept only the five routed categories. Apply parser/KITTI rules only to object detection.
10. **Encoder fallback (MANDATORY)** — prohibit `x264enc`/`openh264enc`; use NVENC or `theoraenc + oggmux` (`.ogv`). Follow the ordered, one-retry procedure in [runtime-environment.md](references/runtime-environment.md#encoder-fallback). An unavailable encoder returns `ENCODER_EXIT_4`, pipeline `BLOCKED`, visual `UNTESTED`; never claim `PASS` from a skipped run.
11. **Video source (MANDATORY)** — use the caller's explicit `DS_VIDEO`, otherwise `sample_720p.mp4` (1280×720). Preflight must verify the selected file exists. If absent, block and request an explicit input; never autonomously substitute another file.
12. **Engine scaling bound** — allow at most three rebuilds and batch 512. A successful, validated build at either bound may continue normally. If the next required build would exceed either bound, retain successful artifacts, report `BLOCKED`, and return engine exit `4` (`ENGINE_EXIT_4`); never continue the loop.

These phase tags describe the existing numeric exit `4`, not environment
variables or new exit codes: `ROUTER_EXIT_4` reads `capability_status`,
`ENCODER_EXIT_4` records a skipped visual run, and `ENGINE_EXIT_4` records the
scaling bound. Never treat one phase's exit code as another phase's approval.

## Examples

- “Use `deepstream-import-vision-model` to import
  `PekingU/rtdetr_r50vd` at commit `<sha>` and validate it on `sample_720p.mp4`.”
- “Use `deepstream-import-vision-model` to plan a semantic-segmentation import
  for `/data/model.onnx` with `/data/task-metadata.json`; do not execute until
  the generated adapter is implemented and evidence-ready.”

Read [references/multitask/examples.md](references/multitask/examples.md) for
the remaining category prompts and expected gate behavior.

## Pipeline Timing

Record each step's and campaign's wall-clock timing in the report; calculate
with Python, not unavailable `bc`.

## Report output

Step 8 consumes `case-evidence.json` and writes Markdown/HTML/PDF to
`models/$MODEL_NAME/reports/`; follow [report-generation.md](references/report-generation.md)
for schema, names, and campaign procedure. A 13/14-case suite is a reporting
registry, not expanded routing permission. Any stereo or unsupported entry
gets `BLOCKED` at Step 0 without conversion/build/inference. Partial-evidence
status PDFs retain every unexecuted gate as `UNTESTED`, never functional `PASS`.
Report generation is not a model test.

## Scripts

Installed into `.claude/skills/deepstream-import-vision-model/scripts/` by `install.sh`.
Use phase-reference commands, or pass script path and arguments separately to
`run_script()` if available. [script-index.md](references/script-index.md)
lists installed scripts, phases, and purposes.
