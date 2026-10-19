<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# Experimental fixed-pair extra-input diagnostic

This path checks whether a two-input FP32 TensorRT engine can run inside
`nvinfer` for a **single repeated stereo pair**. The DeepStream SDK calls
`NvDsInferInitializeInputLayers` only once. The right tensor is static for the
entire run; a successful EOS is **not** synchronized stereo-video validation.
Do not route experimental stereo as `READY` or promote an adapter from this
diagnostic. A dynamic pair needs a separate synchronized-input runner (for
example a custom `nvinferserver` extra-input processor), calibration checks,
semantic oracle, and rendered-frame review.

## Exact input contract

- An inspected engine with one image input and exactly one extra FP32 input.
- `DS_STATIC_EXTRA_INPUT_NAME` equals the extra binding name exactly.
- `DS_STATIC_EXTRA_INPUT_FILE` is a regular, non-symlink binary file containing
  exactly one batch item's finite contiguous FP32 values in binding order.
- Left and right images are distinct and share a reviewed rectification and
  calibration contract. The left image can be repeated into an H.264 stream;
  the right tensor must use the **same** spatial crop, color order, and numeric
  preprocessing. Do not pass a compressed image as the extra tensor.
- Select exactly one GPU, then use `gpu-id=0` *inside* that container. The
  `native-pipeline` preflight profile is only a diagnostic; full preflight and
  all evidence gates remain required for certification.

Build `static_extra_input.cpp` against installed DeepStream/CUDA headers as a
shared library and set `custom-lib-path` in the `nvinfer` config. The config
also needs the inspected engine, `network-type=100`, `output-tensor-meta=1`,
`batch-size=1`, and matching input dimensions/color/scale. Export the two
environment variables above in the process that loads `nvinfer`.

For one NCHW FP32 scalar output `[1, 1, H, W]`, the existing
`render_segmentation_logits.cpp` runner accepts its optional
`--scalar <min> <max>` suffix. Use a fixed, documented physical-value range and
at least two display bins; it writes frame-linked raw FP32 maps and a decoded
overlay video. Check that every frame has a finite map, the output video
decodes, the overlay aligns with the input geometry, and the run reaches EOS.
The display bins are **not** a semantic oracle. Compare raw tensors to the
reference backend and run a reviewed metric against ground truth separately.
Neither a TensorRT build nor a colorized frame alone is a functional PASS.

For changing left/right video, temporal inputs, more than one extra binding,
non-FP32 tensors, or incompatible layouts, stop with `ADAPTER_REQUIRED`. Do
not silently reuse this initializer; it would feed a stale right frame.
