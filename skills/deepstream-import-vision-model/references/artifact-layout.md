<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Per-model artifact layout

After model/task/media/oracle confirmation, use a single path component for
`MODEL_NAME`. Reject empty names, separators, whitespace, leading dots, and
traversal rather than trying to repair them. Use the validated name for every
later artifact path.

```bash
[[ "${MODEL_NAME:-}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "BLOCKED: invalid MODEL_NAME" >&2
  exit 4
}
MODEL_DIR="models/$MODEL_NAME"
mkdir -p -- "$MODEL_DIR"/{model,config,scripts,adapters,structured,benchmarks/engines,benchmarks/ds,reports/charts,samples}
# Object detection only:
mkdir -p -- "$MODEL_DIR"/{parser,samples/kitti_output}
```

Never dump files flat. The directories have these uses:

```text
models/{model_name}/
  model/           <- ONNX file(s)
  config/          <- nvinfer config, ds-app config, labels.txt
  scripts/         <- run helper scripts
  adapters/        <- non-detection adapter source and contract when required
  structured/      <- frame/ROI-keyed task output
  benchmarks/
    engines/       <- _dynamic_b{MAX_BS}.engine, timing.cache, build logs
    b1/            <- trtexec BS=1 log
    b{MAX_BS}/     <- trtexec BS=MAX_BS log
    ds/            <- DS benchmark logs
  reports/         <- Markdown, HTML, PDF, evidence JSON
    charts/        <- category-relevant charts; exactly five for detection
  samples/         <- output .mp4 or .ogv and test frames

Detection additionally creates:
  parser/          <- .cpp, Makefile, .so
  samples/kitti_output/ <- KITTI detection .txt files
```
