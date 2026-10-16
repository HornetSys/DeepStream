<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

## Model choice

For a production execution request (not a dry run), collect and validate the exact media
fixture and category-specific semantic oracle during intake, before model
download, conversion, or build. The validated default model choice does not
waive these inputs. If either is absent, return `BLOCKED` before acquisition and
identify the missing input; do not defer the production stop to a later phase.
An explicitly diagnostic run against an existing local artifact and real media
may proceed without a reviewed oracle, with semantic `UNTESTED` and only a
status report. Do not download a new remote model merely to bypass this gate.

Before state-changing preflight, downloads, or file creation, present exactly
these two choices when no supported model is selected. If the request already
selects one, state whether it maps to the default or custom choice and confirm
that mapping; do not re-present both choices. A local artifact maps to Custom:
confirm its task and media, record whether the oracle is validated or absent,
then run Step 0 without presenting choices.

### 1. Default model (recommended)

Use the validated Hugging Face RT-DETR model:

```yaml
model_id: PekingU/rtdetr_r50vd
source: huggingface
task: object-detection
precision_preference: fp16
```

### 2. Custom supported vision model

Ask for one supported source:

- Hugging Face model ID (`organization/model`) or full model URL.
- NVIDIA NGC catalog model URL including its version.

Require one of the five supported categories plus its tensor, preprocessing,
postprocessing, media, and semantic-oracle contract. Do not invent or silently
substitute a model when the custom source or task contract is missing.

After the caller selects Custom, metadata-only browsing of `config.json`, the
model card, or an NGC version manifest is allowed to discover the task and
tensor contract. Make no download or file change during this inspection. Show
the discovered contract and obtain final model/task confirmation before
materializing an artifact or executing a pipeline.
Within an approved model/task choice, deterministic FP16-first variant
selection does not require another confirmation. If FP16 is unavailable,
confirm the documented fallback precision before download.

Treat a request as a dry run only when the caller asks for a dry run or a
plan-only response; otherwise it is an execution request. For a dry run,
apply the same choice rules above (confirm a supplied model; otherwise present
the two choices) and simulate discovery, build,
benchmark, and report stages without browsing, downloading, launching Docker,
writing files, or starting processes. Emit one JSON object to stdout with
exactly these top-level fields: `model_choice`, `source`, `task`,
`steps_to_run`, and `estimated_outputs`. Commands may appear only as inert plan
strings. Do not inspect the filesystem to decide whether a local artifact or
fixture exists. Represent caller-supplied paths as `unverified` and include a
planned source/media/oracle preflight in `steps_to_run`; never claim that an
artifact is present or missing during a dry run. Stop after emitting the plan.
