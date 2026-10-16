<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Preprocessing, inference, and postprocessing

Preprocessing is part of the model, not a cosmetic pipeline setting. Record and
test every operation in order.

## Preprocessing methods

1. **Native `nvinfer` preprocessing**: use when the input is a single image
   tensor and resize/color/normalization fit `nvinfer` properties. Verify
   `model-color-format`, `net-scale-factor`, `offsets`, `infer-dims`,
   `maintain-aspect-ratio`, and padding behavior against a reference tensor.
2. **`nvdspreprocess`**: use for custom per-frame/per-object ROI preprocessing,
   explicit tensor shapes, shared preprocessing for multiple GIEs, or
   preprocessing that must be represented as `NvDsPreProcessBatchMeta`. Set
   `input-tensor-from-meta=1` in the consuming `nvinfer` configuration and
   match its GIE id. In a pipeline string, use the equivalent element property
   `input-tensor-meta=1`.
3. **Application/custom CUDA preprocessing**: use for temporal stacks,
   multimodal inputs, unusual color spaces, letterbox policies, or operations
   not expressible in the standard plugin. Attach tensors only with a tested
   metadata contract.
4. **Offline preprocessing**: use only for fixture generation and parity
   debugging, never as a substitute for the live pipeline contract.

## Operations that must be specified

- decode color space and channel order: stock configs support RGB/BGR/GRAY;
  RGBA/NV12 require a tested custom preprocessing adapter;
- resize algorithm, target H/W, crop/letterbox, aspect-ratio and padding color;
- layout NCHW/NHWC and batch semantics;
- integer-to-float conversion, scale/range, mean/std or offsets;
- dtype (FP32/FP16/INT8/UINT8 and supported integer types), quantization
  scale/zero-point; when using `nvdspreprocess`, map it explicitly to
  `tensor-data-type`;
- frame versus ROI operation, temporal ordering, and missing-frame policy.

## Mean/std unit convention

The manifest supports two conventions via `preprocess_units`:

- `pixel` (default): `mean`/`std` are in the [0,255] pixel domain. DeepStream
  `nvinfer` uses them directly as `offsets` and `net-scale-factor`; with
  `nvdspreprocess` they become `offsets` and `pixel-normalization-factor`.
- `normalized`: `mean`/`std` are applied to a normalized image (`x * scale`).
  The generator converts them to the pixel domain before writing DeepStream
  properties so the live pipeline matches the reference implementation.

A scalar mean or std is expanded to the known input channel count before
writing config. Per-channel std is rejected for native nvinfer and the stock
custom2d library. The generated type adapter therefore owns the normalized
input tensor and declares `preprocess_strategy=adapter-input-tensor`; nvinfer
consumes its tensor metadata with `input-tensor-from-meta=1`. The adapter
implements exactly `(x-mean)/std` for pixel units or
`(x*scale-mean)/std` for normalized units. Compare its output tensor
numerically with the reference preprocessor before inference. A caller may
instead supply and test a native nvdspreprocess library, but the generator does
not claim such a binary exists.

## Generated adapter lifecycle

When routing returns `ADAPTER_REQUIRED`, run
`scripts/multitask/deepstream/generate_adapter.py <manifest.json> <output-dir>`. The
generator writes `adapter.py`, `adapter-contract.json`, `pipeline-plan.json`, and
`manifest-with-adapter.json` from task type, input domain, and declared upstream
stages. The plan includes source/decode ownership, synchronization or temporal
state where needed, preprocessing, upstream stages, inference, metadata probe,
structured output, renderer, and an output sink with a sink-only software
encoder fallback. It never guesses tensor semantics from a model name. The
generated hooks provide input ownership, decoder, serializer, renderer,
pipeline plan/runner contract, sink selection, and oracle logic for the four
supported non-detection task families. The generator writes
`implementation_status=implemented`; call `generate_configs.py` with
`--allow-implemented-adapter` to create test-only configs. Promote it to
`tested` only after reference/TensorRT parity, one-stream DeepStream metadata,
structured semantic assertions, and visible rendered output all pass on supplied
real input and every `validation_evidence` field contains the resulting evidence
path.

## `nvinfer` usage

Use `nvinfer` for TensorRT execution. A direct ONNX path may build an engine at
runtime; a prebuilt engine is preferred for reproducible profiling. Set the
network type and batch/profile deliberately. For raw tensors, enable
`output-tensor-meta=1` and use a custom/app decoder. For tensors prepared by
`nvdspreprocess`, enable `input-tensor-from-meta=1` in the `nvinfer`
configuration and ensure the preprocessor's target GIE id matches the
inference component.

Do not combine native classifier parsing with a second postprocessor for the
same tensor. Do not use `network-type=100` as evidence that a task is supported;
it is an explicit signal that native parsing is disabled and custom decoding is
required.

## `nvdspreprocess` usage

The element consumes NVMM video and a preprocessing config. Use it when native
properties cannot express the model transform or when several inference
components need a shared tensor. Validate the configured tensor shape, color
format, normalization, ROI mode, and batch order. The generated config in this
skill is a template; the installed DeepStream documentation and plugin version
are authoritative for release-specific keys.

Configure routing in the model manifest instead of editing generated IDs:

- `gie_unique_id`: the consuming `nvinfer` component's unique id;
- `preprocess_unique_id`: the `nvdspreprocess` unique id;
- `target_unique_ids`: one or more GIE ids receiving the generated tensor;
- `preprocess_scope`: `frame` for full-frame/ROI processing or `object` for
  secondary inference on upstream object metadata;
- `operate_on_gie_id` and `operate_on_class_ids`: upstream GIE and optional
  class filter for secondary/object processing;
- `maintain_aspect_ratio` and `symmetric_padding`: explicit resize policy;
- `roi_params`: source-id keys mapped to flattened `x, y, width, height`
  groups for frame ROI processing.

The router rejects duplicate/zero target ids, secondary or object processing
without an upstream GIE, upstream GIE routing on an unrelated primary frame
pipeline, and symmetric padding without aspect-ratio preservation.

## `nvdspostprocess` usage

`nvdspostprocess` consumes inference tensor metadata and calls a postprocess
library/config. On the local DeepStream 9.1 runtime, the documented native task
coverage is detection and classification. For pose, depth, segmentation,
embedding, and optical flow, use a custom postprocess library or application
probe and document the output schema. Always assert that tensor metadata is
present before postprocessing.

## Postprocessing methods by task

| Task | Method |
|---|---|
| Classification | native nvinfer classifier parser, or raw logits → softmax/top-k in a tested library |
| Segmentation | argmax/threshold, optional morphology/connected components, resize with the model's alignment rule |
| Pose | decode heatmaps/offsets/associative embeddings, confidence filtering, skeleton projection |
| Depth | activation/scale conversion, invalid mask, declared units and fixed render range, optional color visualization only after direct numeric validation |
| Embedding | flatten, dtype cast, L2 normalization if required, store exact vector contract |
| Optical flow | channel/order conversion, coordinate scaling, validity/occlusion handling |
| Detection | existing detection skill's parser and gates |
