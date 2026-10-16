<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Task registry and tensor contracts

Task selection is metadata-driven. `auto` may only resolve when the model
metadata identifies the task; otherwise the route is `generic` and the skill
stops `BLOCKED` because generic models are outside this release's five
supported categories.

| Task | Required metadata | Typical outputs | Minimum assertion |
|---|---|---|---|
| Classification | label map, logits/probability semantics, top-k, threshold | `[N,C]` logits/probabilities | stable top-1/top-k and score tolerance |
| Detection | class map, box format, coordinate space, score/NMS policy | boxes, scores, classes | delegate to the detection skill and its parser/KITTI gate |
| Semantic segmentation | class count, mask layout, resize/alignment, excluded-label ID | `[N,C,H,W]` logits or `[N,H,W]` labels | mask shape, label range, IoU/pixel agreement |
| Pose/keypoints | keypoint names/order, skeleton edges, coordinate space, optional confidence threshold | `[N,K,2/3]` + scores or `[N,K,H,W]` heatmaps | keypoint count, coordinate range, PCK/OKS or fixture error |
| Depth | units, inverse-depth flag, invalid value, scale/shift policy, fixed render min/max | `[N,1,H,W]` | finite ratio, range, direct numeric MAE/RMSE |

For pose, coordinate space must match the tensor representation: direct
coordinates may declare pixel or normalized image coordinates, while heatmaps
must declare `heatmap` or `output`. The optional confidence threshold filters
low-confidence points before skeleton rendering. Depth contracts must declare
finite, ascending `render_min` and `render_max`; visual colorization uses that
fixed range and never per-frame min/max normalization.

The manifest must record input name, rank, layout, dtype, channel order, color
format, resize/crop/pad policy, scale, mean/std, output names, stable fixture
identities, optional fixture hashes, and the task's numeric `parity_thresholds`. If the model
uses multiple inputs, list all inputs in the task adapter rather than silently
discarding them; the basic generated image config is intentionally rejected for
non-single-image contracts.

The router normalizes catalog-specific task names to the execution families
above and records the original name in `task_variant`. This is only routing
metadata: the variant's concrete output schema remains mandatory in
`task_contract` before conversion, config generation, or parity execution.

## Output metadata modes

- Native `nvinfer` parsing is appropriate only where the model output matches
  DeepStream's supported classifier/detector contract and the parser semantics
  are verified.
- `output-tensor-meta=1` exposes raw inference tensors for custom decoding.
- A custom parser/library or application probe must own postprocessing for
  pose, depth, and segmentation outputs unless a
  tested plugin/library explicitly implements that schema.
- Detection stays on this skill's established native parser/KITTI route; it
  never enters the non-detection adapter generator.


## Capability contract fields

The router emits capability fields separately from source-format `status`:

- `input_domain`: `image` for this release.
- `upstream_stages`: ordered semantic inputs such as person detection, ROI
  crop, tracking, prompt, memory, or stereo synchronization.
- `adapter_contract`: paths or identifiers for input adapter, config
  generator, pipeline builder, input-tensor preprocessor, decoder, renderer,
  structured-output schema, and `implementation_status`. Generated supported
  adapters are `implemented`, which permits candidate config generation for
  testing but still routes `ADAPTER_REQUIRED`;
  only real-input DeepStream and semantic visual validation may promote it to
  `tested`. A tested contract must also populate
  all eight `validation_evidence` fields for input fixture, backend parity,
  DeepStream runtime, structured output, semantic validation, overlay evidence,
  rendered output, and visual validation.
- `capability_status`: `READY`, `DELEGATED`, `ADAPTER_REQUIRED`, or
  `BLOCKED`, with `capability_reasons`.
- `precision_policy`: ordered, recorded fallback attempts.

A complete task contract describes tensor meaning. A complete adapter contract
describes executable code that turns those tensors into task input/output.
Neither may substitute for the other.

## Type-level topology rules

| Input or task type | Required topology |
|---|---|
| Single image classification | decode, image preprocess, primary inference, classifier metadata, label renderer |
| Secondary classification | upstream detector, class filter, ROI ownership, crop preprocess, secondary inference |
| Segmentation or depth | full-frame or declared ROI owner, raw tensor metadata, semantic decoder, frame remap, overlay |
| 2D pose | full frame only when the model contract says so; otherwise person detector, ROI crop, pose decoder, skeleton renderer |

If any required node is absent, keep the graph `ADAPTER_REQUIRED` or
`BLOCKED`; do not flatten the task to a single full-frame `nvinfer` stage.
