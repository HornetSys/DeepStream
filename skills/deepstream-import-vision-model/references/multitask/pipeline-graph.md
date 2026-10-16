<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pipeline graph manifest contract

Set `pipeline_graph` to a JSON object before command or config emission. The
schema is type-level and applies to the supported non-detection routes:
classification, semantic segmentation, 2D pose, and monocular depth. Reject
embeddings, optical flow, audio, stereo, temporal, prompt/memory, 3D/6D pose,
OCR/LPR, and generic custom tensor routes before graph generation.

```json
{
  "pipeline_graph": {
    "status": "PLANNED",
    "nodes": [
      {"id": "source", "role": "input"},
      {"id": "inference", "role": "model"},
      {"id": "decoder", "role": "task-decoder"},
      {"id": "sink", "role": "structured-and-rendered-output"}
    ],
    "edges": [
      {"from": "source", "to": "inference", "contract": "named input tensor"},
      {"from": "inference", "to": "decoder", "contract": "named output tensors"},
      {"from": "decoder", "to": "sink", "contract": "frame/ROI-keyed task record"}
    ],
    "input_ownership": "component that owns decode, resize, normalization, and tensor production",
    "coordinate_transforms": "forward and inverse resize/crop/pad/ROI mapping",
    "decoder": "library or application hook and its tensor-to-task contract",
    "renderer": "task-record-to-frame overlay implementation",
    "structured_output": "schema and writer for frame/ROI-keyed model results",
    "evidence": {
      "backend_parity": "/case/parity.json",
      "deepstream_runtime": "/case/deepstream.json",
      "structured_output": "/case/output.jsonl",
      "rendered_output": "/case/rendered.mp4"
    }
  }
}
```

Required fields are `status`, `nodes`, `edges`, `input_ownership`,
`coordinate_transforms`, `decoder`, `renderer`, `structured_output`, and
`evidence`. Every node requires non-empty `id` and `role`. Every edge requires
non-empty `from`, `to`, and `contract`. `evidence` must be a JSON object.

Allowed status values are `PLANNED`, `RENDERED_PASS`, `RUNTIME_ONLY`,
`ADAPTER_REQUIRED`, and `BLOCKED`. Use `PLANNED` before execution. Promote it
only from recorded evidence; a planned graph is not runtime-ready.

For multi-input, cascade, stereo, temporal, prompt/memory, or ROI workflows,
list every input owner, synchronization/state boundary, and upstream stage as
nodes and edges. Every manifest `upstream_stages` value must occur in a node
`id` or `role`; `pipeline_graph_errors()` enforces this before backend/config and
visual validation. Do not flatten those workflows into a single full-frame model
node.
