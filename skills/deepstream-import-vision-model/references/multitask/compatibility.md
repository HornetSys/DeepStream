<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compatibility and capability policy

The checked-out repository used to develop this companion skill is a DeepStream
9.1 baseline. The requested rollout target is DeepStream 9.2 / Skills_26.09.
Do not label local 9.1 execution as 9.2 validation. Before release, rerun the
**environment compatibility gate** and the **one-stream DeepStream
functional/parity gate** defined in SKILL.md's Mandatory gates section in the
target container.

## dGPU-first policy

The primary path assumes a CUDA-capable dGPU and TensorRT `trtexec`. Jetson
requires separate checks for architecture, JetPack/L4T, TensorRT, memory, and
plugin availability. A dGPU engine may not be portable to Jetson; rebuild and
validate on the Jetson target.

## Release-sensitive areas

- plugin ABI and config keys;
- TensorRT engine portability and timing cache;
- ONNX opset/operator support;
- `nvdspreprocess` tensor metadata layout;
- `nvdspostprocess` native task coverage and custom-library ABI;
- CUDA, driver, Python, Optimum, PyTorch, TensorFlow, and tf2onnx versions.

Store these versions beside every manifest. If a property is not accepted by
`gst-inspect-1.0` or the target release documentation, stop and update the
template rather than ignoring the error.
