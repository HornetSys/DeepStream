<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Small working examples

These examples describe the contract and commands; they are not claims that a
particular remote model or dataset is installed in the current workspace.

## Minimal local one-stream benchmark

Use one caller-supplied image-classification bundle containing a local ONNX
model, labels, task metadata, one fixed parity fixture, one real video, and a
reviewed label oracle. Keep the bundle and environment identical for both runs.

1. In the skill-assisted run, provide the model source, one of the five
   supported categories, tensor contract, media, and semantic oracle.
2. Route, generate the executable type adapter, and inspect the candidate plan:

   ```bash
   build/.venv_optimum/bin/python \
     .claude/skills/deepstream-import-vision-model/scripts/multitask/model/route_model.py /data/case/model.onnx \
     --task classification --metadata /data/case/task-metadata.json \
     --output /data/case/route.json
   build/.venv_optimum/bin/python \
     .claude/skills/deepstream-import-vision-model/scripts/multitask/model/emit_commands.py /data/case/route.json \
     --output-dir /data/case/generated
   build/.venv_optimum/bin/python \
     .claude/skills/deepstream-import-vision-model/scripts/multitask/deepstream/generate_adapter.py \
     /data/case/route.json /data/case/generated/adapter
   ```

3. Execute the approved conversion plan, validate the ONNX or generated engine,
   generate candidate configs, and run the concrete classification reference:

   ```bash
   CAMPAIGN_ID="${CAMPAIGN_ID:-classification-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
   build/.venv_optimum/bin/python \
     .claude/skills/deepstream-import-vision-model/scripts/multitask/deepstream/generate_configs.py \
     /data/case/generated/adapter/manifest-with-adapter.json \
     /data/case/generated/configs --engine /data/case/model.engine \
     --allow-implemented-adapter
   build/.venv_optimum/bin/python \
     .claude/skills/deepstream-import-vision-model/scripts/multitask/validation/compare_outputs.py \
     /data/case/reference.npz /data/case/tensorrt.npz \
     --manifest /data/case/generated/adapter/manifest-with-adapter.json \
     --input-fixture /data/case/fixture.mp4 \
     --campaign-id "$CAMPAIGN_ID" \
     --reference-backend onnxruntime --candidate-backend tensorrt \
     --output /data/case/evidence/backend_parity.json
   build/.venv_optimum/bin/python \
     .claude/skills/deepstream-import-vision-model/scripts/multitask/deepstream/run_type_adapter_pipeline.py \
     --manifest /data/case/generated/adapter/manifest-with-adapter.json \
     --nvinfer-config /data/case/generated/configs/nvinfer.txt \
     --input /data/case/fixture.mp4 --oracle /data/case/oracle.json \
     --output-dir /data/case/evidence --campaign-id "$CAMPAIGN_ID"
   ```

4. Accept the run only when the runner records startup, nonzero frames, tensor
   metadata, numeric output, EOS, structured output, decoded video, changed
   pixels, and SHA-256-bound model-aware overlay evidence.
5. Repeat the same request and bundle in a clean task without this skill. Score
   both runs with the same gates, repair limit, evidence checklist, elapsed
   time, and tool/token record. Never reuse generated artifacts between runs.

This is a local smoke benchmark, not a replacement for the published multi-case
evaluation.

## Image classification: ONNX → TensorRT → nvinfer

1. Start with a ResNet/MobileNet ONNX model and a label file.
2. Record `NCHW`, RGB/BGR, resize/crop, scale and mean/std.
3. Route with `--task classification`, validate ONNX, and build FP16.
4. Generate native classifier `nvinfer` config with `batch-size=1`.
5. Run one frame and assert class id/score against the reference output.

Use `nvdspreprocess` instead of native preprocessing when a shared ROI transform
or a nonstandard crop is required; then set `input-tensor-from-meta=1` in the
`nvinfer` configuration (`input-tensor-meta=1` on the GStreamer element).

## Semantic segmentation: ONNX → raw tensor → custom postprocess

Use a SegFormer/UNet-style `[1,C,H,W]` output. Set task `segmentation`,
`postprocess=custom`, `output-tensor-meta=1`, and `network-type=100`. Decode
argmax and resize masks using the model's documented alignment policy. Validate
mean IoU and output shape before drawing a colorized mask.

`nvdspostprocess` may host a custom library if its config/library ABI is tested;
the plugin loading by itself is not a segmentation validation.

## Pose/keypoints: PyTorch/SafeTensors → ONNX

Use a task-aware wrapper that returns heatmaps/offsets or final keypoint tensors,
with fixed example input `[1,3,256,192]`. Use `torch.onnx.export` and
record output names. Optimum is optional only after its route-specific
environment passes the documented torch/Transformers/Optimum import check.
Generate `network-type=100`, `output-tensor-meta=1`, and
`postprocess-contract.json`. Decode the named tensors in a custom library or
app probe, preserving keypoint order, visibility/confidence, coordinate space,
and skeleton edges. Validate PCK/OKS >= 0.99 or a stricter model-owned
fixed-coordinate threshold.

## Monocular depth: ONNX → TensorRT

Record whether the output is depth or inverse depth, units, scale/shift, invalid
value, output resolution, and a fixed valid render range (`render_min` and
`render_max`). Use raw tensor metadata and a custom/app decoder.
Validate finite ratio and direct MAE/RMSE. Require a model-specific alignment
procedure only when the caller supplies it; never treat a visually pleasing
colormap as numeric validation.

## TFLite MobileNet: experimental conversion

Run `tf2onnx.convert --tflite`, inspect quantization and layout, compare TFLite
and ONNX outputs on the same uint8/float fixtures, then build TensorRT only if
operator and metric gates pass. Mark the route `experimental` in the report.

## Detection

Invoke `deepstream-import-vision-model` for detection. This companion only
records the delegation and does not create a second bbox parser workflow.

## Copy-ready prompts

- `Use deepstream-import-vision-model to import this local ResNet ONNX classifier, validate TensorRT parity, run one supplied video, and render class labels and scores.`
- `Use deepstream-import-vision-model to import this semantic-segmentation model, generate the mask adapter from its declared NCHW logits contract, and keep it ADAPTER_REQUIRED until real-input DeepStream and visual validation pass.`
- `Use deepstream-import-vision-model to route this 2D-pose model, preserve its declared person-detection ROI stage, and require structured keypoints plus a skeleton overlay.`
- `Use deepstream-import-vision-model to import this monocular-depth ONNX model and validate its units, direct numeric output, and model-derived heatmap.`
- `Use deepstream-import-vision-model on onnx-community/mobilenet_v2_0.35_96 at revision f37d9b91354e5a43dae384840f46119c67a76d58. Inspect its 96x96 ImageNet preprocessing, pin the ONNX and labels, build TensorRT, use FP32 if FP16 raw-logit parity fails, run my supplied video through DeepStream on one GPU, validate a reviewed top-1 label oracle, and include the rendered input/output frames in the Step 8 PDF. Treat a generated or looped dog image as a diagnostic fixture only.`

For exact vendor artifacts and suitable SDK/caller media, first consult
[model-zoo-media.md](../model-zoo-media.md). These concrete prompts plan or
attempt a run; they do not assert a passing result:

- `Use deepstream-import-vision-model with NGC CitySemSegFormer deployable_onnx_v1.0 (citysemsegformer.onnx, labels.txt, nvinfer_config.txt). Run the documented DeepStream sample_720p.mp4 segmentation path, save the mask rendered on a frame, and make a status PDF if a reviewed frame-aligned mask oracle is unavailable.`
- `Use deepstream-import-vision-model with NGC VehicleMakeNet pruned_onnx_v1.1.0 and the SDK sample_1080p_h264.mp4 clip. Run a vehicle detector first, classify its actual car ROIs, capture the overlaid make on the source frame, and report semantic UNTESTED unless a reviewed label oracle is supplied.`

For that vehicle example, after generating and auditing a primary detector
`nvinfer` config and a secondary classifier `nvinfer` config, run the reusable
ROI cascade in a Python environment containing DeepStream Service Maker and
PyYAML. Replace the paths with the exact caller-provided files; both configs
must declare distinct GIE IDs, `process-mode=1/2`, and the secondary config
must declare the detector GIE ID and vehicle class IDs. The runner writes a
DeepStream graph, structured per-ROI labels, and an overlaid video; those are
visual diagnostics until a reviewed make-label oracle passes.

```bash
/path/to/deepstream-python-with-pyyaml \
  .claude/skills/deepstream-import-vision-model/scripts/multitask/deepstream/run_secondary_classification.py \
  --video /data/case/vehicle.mp4 \
  --primary-config /data/case/detector-nvinfer.txt \
  --secondary-config /data/case/make-nvinfer.txt \
  --detector-labels /data/case/detector-labels.txt \
  --classifier-labels /data/case/make-labels.txt \
  --output-dir /data/case/evidence/make-cascade
```
