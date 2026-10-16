<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepStream Import Vision Model

Task-aware visual-model import: model source → TensorRT validation → native or
tested-adapter DeepStream pipeline → structured and rendered evidence → report.

## Overview

This self-contained skill routes five visual categories through a shared model
acquisition and validation workflow. Object detection uses the established
native parser, KITTI, benchmark, and report path. Classification can use native
DeepStream when its contract is compatible or the shipped one-stream reference
runner for adapter testing. Segmentation, pose, and depth stop at
`ADAPTER_REQUIRED` until an equivalent concrete runner and real-input evidence
are available.

Supported input formats: ONNX (direct), SafeTensors (auto-exported via `torch.onnx.export`).

**Current scope:** object detection, image classification, semantic
segmentation, 2D pose/keypoints, and monocular depth. Other categories fail
closed in the unified task router.

## Prerequisites

**Host: only Docker + the NVIDIA driver.** Everything else runs **inside the DeepStream container** —
DeepStream, TensorRT/`trtexec`, the Python export venv (`torch`/`onnx`/`onnxruntime`), `wkhtmltopdf`,
`deepstream-app`/`gst-launch-1.0` — and is bootstrapped by `setup.sh`. Nothing is installed on the host.
Runs identically on Linux and **Windows** (Docker Desktop + WSL2 backend, required for GPU).

- **Docker** — Docker Desktop with the WSL2 backend on Windows; Docker Engine + NVIDIA Container
  Toolkit on Linux.
- **NVIDIA GPU + driver** (on Windows, the WSL2 GPU driver — no host CUDA/TensorRT/DeepStream needed).
- Resolve the DeepStream tag externally, set `DS_IMAGE` to the verified
  `nvcr.io/nvidia/deepstream@sha256:<digest>`, then follow `SKILL.md` to create
  one named campaign container and run setup, preflight, and all phases in it.

See **[references/windows.md](references/windows.md)** for the cross-platform run model and the
per-shell `docker create` mount token.

## Installation

```bash
bash <path-to-deepstream-import-vision-model>/install.sh --target <your-project-path>
```

Preview what will be installed first with `--dry-run`:

```bash
bash <path-to-deepstream-import-vision-model>/install.sh --target <your-project-path> --dry-run
```

Where `<path-to-deepstream-import-vision-model>` is the location of this skill in your repo, e.g.:
- In **team-mind-hub**: `team-skills/deepstream-sdk/deepstream-import-vision-model`
- In **ds-copilot**: `team-skills/deepstream-sdk/deepstream-import-vision-model` (same path)

The script copies the complete skill into the target project for Claude Code, Codex, and Cursor. Re-running it safely refreshes an existing installation.

## Usage

**Claude Code:**
```text
Use deepstream-import-vision-model to run this model: https://huggingface.co/onnx-community/yolov8n
```

**Codex:**
```text
Use $deepstream-import-vision-model to deploy and benchmark https://huggingface.co/onnx-community/yolov8n
```

**Cursor:**
```text
@deepstream-import-vision-model run this model: https://huggingface.co/onnx-community/yolov8n
```

The skill proceeds autonomously through applicable gates. It stops on
`BLOCKED`; an `ADAPTER_REQUIRED` route may generate and test its executable
type adapter but is never reported production-ready until evidence-backed
promotion.

## Pipeline Steps

| Step | Phase reference | Action |
|------|-----------|--------|
| 0 | `references/multitask/task-registry.md` | Route one of five categories and determine readiness |
| 1–3 | `references/model-acquire.md` | Browse the source, download ONNX, or export SafeTensors |
| 4–5 | `references/engine-build.md` | Build dynamic TRT engine, run trtexec benchmarks |
| 6–7 | `references/pipeline-run.md` or `references/multitask/validation.md` | Run the native detection path or an evidence-backed tested adapter |
| 7 | `references/adapter-readiness.md` | Promote a candidate only after real-input runtime, semantic, and visual gates pass |
| 8 | `references/report-generation.md` | Shared detection/non-detection Markdown, HTML, PDF, evidence, and campaign index |

## Output Structure

Per-model outputs are written to `models/<model_name>/` in your project:

```text
models/<model_name>/
  model/          ONNX file(s)
  config/         nvinfer config, DS app config, labels.txt
  scripts/        Model-specific run helpers
  adapters/       Non-detection adapter source and contract when required
  structured/     Frame/ROI-keyed task output
  benchmarks/     TRT engines, trtexec logs
  reports/        Category report, evidence JSON, and applicable charts
  samples/        Output videos and test frames

Detection additionally creates `parser/` and `samples/kitti_output/`.
```

## Files in this package

```text
deepstream-import-vision-model/
├── SKILL.md                    Top-level skill definition
├── README.md                   This file
├── references/                 Phase-specific runbooks
├── scripts/                    Utility scripts by pipeline phase
└── tests/                      Self-contained developer regression tests
```

Developer regression tests live in `tests/` inside this skill, alongside its
scripts and references. From the repository root, run
`python3 -m unittest discover -s skills/deepstream-import-vision-model/tests -v`.
From an installed skill directory, use `python3 -m unittest discover -s tests -v`.
