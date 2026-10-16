<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Supplied TensorRT Engine Approval

A caller-supplied TensorRT plan may contain unsafe or incompatible serialized
content. Never deserialize it merely because a path was supplied.

## Request and record approval

Ask the caller to approve deserialization of the exact engine inside one exact
trusted model directory. Record that decision in the routed manifest:

```json
{
  "execution_approval": {
    "approved": true,
    "approver": "caller identity or ticket",
    "approved_at": "2026-09-16T12:00:00Z",
    "trusted_root": "/absolute/path/to/models/model-name",
    "scope": "deserialize-supplied-tensorrt-engine",
    "engine_provenance": "/absolute/path/to/models/model-name/build.json"
  }
}
```

Approval is valid only when:

- `approved` is the boolean `true`;
- approver and timezone-qualified timestamp are present;
- `trusted_root` is absolute and exactly matches the validation argument;
- the engine resolves inside that root without symbolic-link components;
- provenance identifies the engine source, TensorRT/CUDA/GPU versions, and hash;
- scope is exactly `deserialize-supplied-tensorrt-engine`.

## Evaluate approval

Use only the approval-gated validator:

```bash
python3 .claude/skills/deepstream-import-vision-model/scripts/multitask/validation/validate_artifact.py \
  manifest.json --engine /absolute/path/model.engine --run-trtexec \
  --trusted-root /absolute/path/to/models/model-name \
  --engine-provenance /absolute/path/to/models/model-name/build.json
```

The command emitter generates this validator call and refuses incomplete
approval. Before this validator succeeds, do not emit or execute a direct
`trtexec --loadEngine` command for a supplied plan. Missing or mismatched
approval is `BLOCKED`, not a prompt to weaken validation.

## Shape and benchmark scope

`validate_artifact.py --run-trtexec` performs approval-gated deserialization
with `--skipInference`; it does not measure throughput or verify every planned
shape/batch. Before Step 6, verify the supplied engine's input bindings and
profiles cover the declared preprocessing shapes and every planned batch.
For a dynamic benchmark, require coverage of BS=1 through the declared
`MAX_BS` (at most 512), and record the measured BS=1 and MAX_BS trtexec runs
using the benchmark procedure in `engine-build.md` without rebuilding the
approved engine. A static engine may run only supported shapes/batches; do
not advertise dynamic or multistream capacity it cannot execute. Missing
profile verification is `BLOCKED` for that run, not an inferred `PASS`.

Approval skips the engine-build/scaling steps, not these profile checks or
the measurements required by the requested report. Every subsequent load
remains subject to the same recorded approval and trusted-root contract.
