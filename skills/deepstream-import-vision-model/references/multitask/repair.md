<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Bounded repair loop

Repair only after capturing the first causal error. Preserve the failed command,
manifest, environment, and log. Never hide an error by loosening validation or
switching silently to a different model/task.

| Failure | First check | Bounded repair |
|---|---|---|
| Missing exporter dependency | caller-owned route environment | use the explicit `torch.onnx.export` wrapper by default; install and import-check a compatible Optimum/Transformers pair only when that optional exporter supports the task |
| ONNX checker/operator failure | opset, custom ops, dynamic dimensions | choose a supported opset, simplify only with a recorded command, or stop for a custom plugin |
| TensorRT shape/profile error | names, rank, NCHW/NHWC, min/opt/max | resolve dimensions from metadata and rebuild once |
| TensorRT unsupported operator | first unsupported node and plugin availability | use the framework exporter/simplifier or a custom TensorRT plugin; do not delete the node |
| Wrong scores/empty masks | color, resize, scale, mean/std, layout | compare the first preprocessed tensor and correct one transform at a time |
| Missing tensor metadata | `output-tensor-meta`, GIE id, plugin order | fix the metadata contract and rerun one-stream smoke test |
| Wrong task output | output schema and decoder | select the correct task adapter; do not reinterpret tensors opportunistically |
| DeepStream startup failure | plugin/version/engine path | validate capabilities and absolute paths; do not claim runtime success |
| Devtest/catalog input is not materialized | expected URL and SHA-256 in the test harness | the harness resolves and verifies one pinned artifact before invoking the skill; normal skill execution never downloads or rechecks it |
| Per-channel standard deviation | preprocessing equation and channel order | use the generated adapter-owned input tensor and compare it with the reference preprocessor; do not average std values |
| FP16 drift or non-finite output | raw tensors, TF32, first divergent layer | retry once with TF32 disabled, then FP32; preserve FP16 FAIL and report fallback precision |
| Unsupported ARM media | codec, profile, chroma, bit depth | preserve source and create one hashed H.264/YUV420 derivative; do not change semantic content |
| TensorRT OOM | peak memory, workspace, shape profile | retry a smaller valid profile or higher-memory build host; otherwise report BLOCKED |
| Runtime succeeds but oracle is missing | fixture target and ground truth | report semantic UNTESTED; do not convert missing evidence to FAIL |
| Type adapter is missing | input domain, upstream stages, decoder, renderer | report ADAPTER_REQUIRED, generate the supported task-type adapter from a complete contract, and keep it unpromoted until real-input visual validation passes |

At most three repairs are attempted for one gate unless that gate declares a
stricter cap. The detection KITTI gate and encoder sink fallback permit exactly
one repair attempt total. After a repaired gate passes,
preserve the attempts and resume at the next applicable workflow gate. After
three unsuccessful repairs, preserve the route status, set that gate and overall
execution status to `FAIL`, emit the final manifest with the causal log excerpt,
deduplicated root causes, evidence paths, and required external decision, then
stop before downstream gates.
