<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Installed script index

`install.sh` installs these under
`.claude/skills/deepstream-import-vision-model/scripts/`. Use the command in
the relevant phase reference rather than inferring arguments from this index.

| Script | Phase | Purpose |
|--------|-------|---------|
| `model/route_task.py` | 0 | Unified five-category router and fail-closed scope gate |
| `model/hf-list-files.sh` | 1–3 | List HuggingFace repo files |
| `model/hf-curl.sh` | 1–3 | Shared HF transport; credential header via stdin, not process arguments |
| `model/hf-download-config.sh` | 1–3 | Download config.json from HF |
| `model/hf-download-file.sh` | 1–3 | Download any safe repo-relative HF file from a pinned revision |
| `model/hf-resolve-revision.sh` | 1–3 | Resolve a Hugging Face branch/tag to an immutable commit SHA |
| `model/safetensors-to-onnx.sh` | 1–3 | Export SafeTensors → ONNX via `torch.onnx.export` (wrapper) |
| `model/safetensors_to_onnx.py` | 1–3 | The exporter — dynamo backend, TorchScript fallback, verifies dynamic batch |
| `model/inspect-onnx.py` | 1–5 | Inspect ONNX input/output shapes |
| `engine/benchmark-trtexec.sh` | 4–5 | Run trtexec with standard flags |
| `deepstream/ds-single-stream.sh` | 6–7 | Single-stream visual validation; NVENC or theoraenc+oggmux fallback |
| `deepstream/parsers/yolo_dense_xywh.cpp` | 6–7 | Optional checked parser for an inspected dense YOLO XYWH/COCO head |
| `deepstream/ds-kitti-dump.sh` | 6–7 | KITTI detection dump via deepstream-app |
| `deepstream/ds-perf-run.sh` | 7 | Two-run DeepStream benchmark with fixed-name logs |
| `deepstream/extract-frame.sh` | 6–7 | Extract sample frames from `.mp4` or `.ogv` |
| `report/generate-benchmark-charts.py` | 8 | Generate five benchmark PNG charts |
| `report/collect-report-vars.sh` | 8 | Collect measured all-source sustained FPS; safe to source |
| `report/write-benchmark-data.py` | 8 | Write finite benchmark JSON to a new output file |
| `report/write-benchmark-report.py` | 8 | Compatibility entry point into the shared evidence-manifest report generator |
| `report/verify-charts.sh` | 8 | Check the five required detection benchmark charts |
| `report/md-to-html-pdf.py` | 8 | Canonical Markdown → HTML → PDF conversion |
| `report/generate-model-report.py` | 8 | Emit per-case benchmark, validation, or status report |
| `report/generate-campaign-index.py` | 8 | Index generated per-case PDFs |
| `report/prepare-campaign.py` | 0 | Stage caller inputs and hashes; never runs inference |
| `multitask/model/route_model.py` | 0–5 | Non-detection manifest and conversion planner |
| `multitask/deepstream/generate_adapter.py` | 6 | Generate executable category adapter from complete contracts |
| `multitask/deepstream/generate_configs.py` | 6 | Emit DeepStream configs after adapter implementation |
| `multitask/deepstream/run_type_adapter_pipeline.py` | 6–7 | Classification pipeline and evidence runner |
| `multitask/deepstream/run_task_capacity.py` | 7 | Complete caller-owned task graph, all-source FPS, separate frame/ROI batches; see [contract](benchmark-evidence.md) |
| `multitask/deepstream/run_secondary_classification.py` | 6–7 | Detector-owned ROI → secondary classifier; requires Service Maker and PyYAML |
| `multitask/deepstream/render_segmentation_logits.cpp` | 6–7 | NCHW FP32 segmentation-logit renderer; see [contract](segmentation-logits-runner.md) |
| `multitask/deepstream/static_extra_input.cpp` | 6–7 | Static FP32 input for fixed-pair diagnostic; see [contract](stereo-fixed-pair.md) |
| `multitask/validation/validate_artifact.py` | 3–6 | Validate ONNX or approval-gated supplied TensorRT engine |
| `multitask/validation/validate_semantic_oracle.py` | 7 | Validate oracle type, provenance, samples, metric, threshold |
| `multitask/validation/compare_outputs.py` | 7 | Compare reference, ONNX Runtime, TensorRT outputs |
| `multitask/visual/validate_rendered_output.py` | 7 | Gate model-derived rendered evidence |
| `multitask/validation/validate_adapter_readiness.py` | 7 | Bind passing evidence and author tested readiness |
