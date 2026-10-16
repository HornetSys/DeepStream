<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adapter Readiness Gate

Generated adapters are executable task-type implementations. Keep them
`ADAPTER_REQUIRED` until the validator promotes them using real-input
DeepStream, semantic, and visual evidence. Classification uses the shipped
`run_type_adapter_pipeline.py` reference. Segmentation, pose, and depth cannot
be promoted until an equivalent concrete runner is added.
Promotion proves functional readiness on the exact fixture; it does not prove
a multi-camera 30-FPS ceiling. Keep capacity `UNTESTED` until a task-compatible
benchmark measures it.

## Transition

1. Generate the adapter from a complete supported task contract. Verify every
   `required_hooks` entry and the generated
   `adapter_contract.implementation_status=implemented`.
2. Generate candidate configs with `--allow-implemented-adapter`.
3. Run the same supplied fixture through reference/ONNX/TensorRT and the
   concrete DeepStream runner. Use one `campaign_id` for all evidence.
4. Populate these `adapter_contract.validation_evidence` paths:
   - `input_fixture`: exact real media input;
   - `backend_parity`: schema-v1 JSON naming reference/candidate backends and
     a finite passing metric;
   - `deepstream_runtime`: runner-authored schema-v1 JSON proving startup,
     positive buffer and tensor-metadata counts, tensor names, finite numeric
     assertion, EOS, exit code 0, and structured/rendered hashes;
   - `structured_output`: frame/ROI-keyed schema-v1 task output JSON;
   - `semantic_validation`: oracle result with metric, oracle hash, and
     structured-output hash;
   - `overlay_evidence`: renderer JSON binding frame/ROI, semantic fields,
     structured-output hash, and rendered-frame hash;
   - `rendered_output`: the exact decoded image hashed by overlay evidence;
   - `visual_validation`: `RENDERED_PASS` report whose required image,
     structured-output, overlay, difference, and video checks all pass.
5. Every JSON artifact must contain `schema_version=1`, the same non-empty
   `campaign_id`, `task`, routed-model `source_hash`, exact
   `input_fixture_sha256`, and `producer`. Producers are fixed by artifact:
   `compare_outputs.py`, `deepstream-type-adapter-runner` for runtime and
   structured output, `adapter-semantic-validator`, `adapter-renderer`, and
   `validate_rendered_output.py`. The runtime record must also match the
   installed runner SHA-256. Producers derive the model hash when the caller
   did not supply one; this does not require an expected checksum.
6. Run the only promotion command:

```bash
python3 .claude/skills/deepstream-import-vision-model/scripts/multitask/validation/validate_adapter_readiness.py \
  models/$MODEL_NAME/adapters/$CATEGORY/manifest-with-adapter.json \
  --status-output models/$MODEL_NAME/adapters/$CATEGORY/adapter-status.json \
  --manifest-output models/$MODEL_NAME/adapters/$CATEGORY/manifest-tested.json
```

The validator enforces the per-evidence schemas above, campaign/model/input
identity, task semantics, runner provenance, gate details, cross-artifact
SHA-256 bindings, and immutable evidence hashes. A bare `{"status":"PASS"}`
cannot promote an adapter. It writes `manifest-tested.json` only when every
check passes. The router revalidates `adapter-status.json` and its hashes
before returning `READY`.

These checks establish source-controlled producer provenance and artifact
integrity; they are not a substitute for a separately signed CI attestation
against a malicious workspace owner. Publication CI should sign the promoted
bundle when cryptographic origin is required.

The promoted manifest records the status path relative to itself, and the
status records evidence paths relative to itself. Keep the manifest, status,
and evidence in the same bundle when moving it. Never hand-edit
`implementation_status=tested`, `readiness_status`, or status JSON. Missing,
weak, mismatched, or changed evidence remains `ADAPTER_REQUIRED` with exit 4.
