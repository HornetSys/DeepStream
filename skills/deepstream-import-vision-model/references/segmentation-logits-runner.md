<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# NCHW semantic-segmentation logits: full-frame DeepStream renderer

Use `scripts/multitask/deepstream/render_segmentation_logits.cpp` only after
inspecting the exact model output. This type-level runner expects **one**
finite FP32 output tensor in NCHW order, with `C` channels and a spatial grid
that may be smaller than the model input. It decodes `argmax(C)`, writes every
native class map as little-endian signed `int32` in a raw file with frame-linked
offsets in JSONL, nearest-neighbor remaps the class map to the `nvstreammux`
frame geometry, and attaches DeepStream segmentation metadata for `nvsegvisual`.
The nvinfer config must set `network-type=100`, `output-tensor-meta=1`, the
inspected channel count, and **full-frame resize without letterbox/padding**.
Do not use this runner for NHWC, multiple outputs, secondary crops, instance
masks, or padded/aspect-ratio-preserving preprocessing without implementing
their explicit geometry contract. A class-map output is not logits.

Build against the installed DeepStream SDK, then use fresh paths for every run:

```bash
g++ -std=c++17 -O2 \
  scripts/multitask/deepstream/render_segmentation_logits.cpp \
  -I/opt/nvidia/deepstream/deepstream/sources/includes \
  $(pkg-config --cflags --libs gstreamer-1.0 glib-2.0) \
  -L/opt/nvidia/deepstream/deepstream/lib \
  -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib \
  -lnvds_meta -lnvdsgst_meta -o models/$MODEL_NAME/adapters/render_segmentation_logits

models/$MODEL_NAME/adapters/render_segmentation_logits \
  models/$MODEL_NAME/config/nvinfer.txt \
  models/$MODEL_NAME/samples/input.h264 \
  models/$MODEL_NAME/samples/rendered.ogv \
  models/$MODEL_NAME/structured/masks.jsonl \
  1280 720 "$CLASS_COUNT" \
  models/$MODEL_NAME/structured/native-masks.raw
```

The input is an existing H.264 elementary stream; remux a real video fixture
without re-encoding if necessary. The output dimensions must match the
`nvstreammux` frame size. `SEGMENTATION_EOS frames=N masks=N`, exit zero,
nonempty frame-linked raw masks, a decoded rendered video, and inspected
source/output frames are the minimum pipeline/visual evidence. Compare the raw
maps to a reviewed frame-aligned oracle before a semantic PASS. A missing
oracle leaves full functional readiness blocked even when the overlay is
visually valid. Record encoder mode and do not infer multi-stream FPS from EOS.

For a single NCHW FP32 scalar map, the optional `--scalar MIN MAX` suffix
instead writes raw FP32 maps and quantizes them into the supplied display-bin
count. This is only a diagnostic overlay; see
[fixed-pair stereo limits](stereo-fixed-pair.md) when the model has an
additional static input.

Diagnostic example: the SegFormer-B0 ADE20K 150-class, 128×128 logit head
completed 984 frames on a 1280×720 classroom clip after class-map remapping.
The older adapter attached a 128×128 map directly and painted only small
patches; this runner's output covered people, tables, walls, and floor at the
frame geometry. That visual observation is not a mask-accuracy score.
