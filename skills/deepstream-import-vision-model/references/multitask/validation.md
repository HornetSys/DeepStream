<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mandatory validation and parity gates

The pipeline is successful only when every applicable gate passes. A missing
optional dependency is an `UNTESTED` result, not a pass.

## Required execution order

For ONNX and converted inputs, execute and record:
`environment/manifest → onnx.checker + shape inference → source/reference versus
ONNX Runtime parity → trtexec build/smoke → TensorRT parity → executable
task-type adapter generation when required → candidate config generation → one-stream
DeepStream smoke → DeepStream parity → task-compatible multi-stream profile when
one exists`.
Detection retains its original two-run benchmark. Other categories may use
`scripts/multitask/deepstream/run_task_capacity.py` only with the complete
caller-owned inference graph and an executable task decoder. Record capacity
`UNTESTED` until that compatible runner actually measures every logical
stream; a supplied tensor hook is not automatic adapter readiness.
Stop when continuing would be unsafe or would invalidate downstream evidence.
Otherwise preserve the failed gate and allow an explicitly separate
runtime-visual diagnostic lane when a compatible TensorRT engine can still be
executed safely. Never rewrite the failed backend gate as a pass. For a supplied
compatible TensorRT engine, skip the ONNX/build legs and start with
trusted-root/version/profile/plugin/binding validation before config
generation.

## Gate 0: environment

Record:

```bash
nvidia-smi
trtexec --version
gst-inspect-1.0 nvinfer
gst-inspect-1.0 nvdspreprocess
gst-inspect-1.0 nvdspostprocess
deepstream-app --version
```

Also record CUDA/TensorRT/DeepStream package versions, GPU compute capability,
OS/architecture, and whether the run is dGPU or Jetson. Write these values to
`environment.json` beside the manifest, set `environment_snapshot` to that
path, and copy the version keys into the manifest's `environment` object. If
the target release differs from the installed runtime, label results with the
installed version.

## Gate 1: manifest and source

- caller-supplied local source exists; optional source hash is provenance only;
- format and task are explicit;
- all inputs/outputs, shapes, dtypes, layout, and preprocessing are recorded;
- labels, units, coordinate conventions, task contract, and postprocess library are present;
- unsupported/experimental routes are marked before conversion.

The execution preflight rejects missing `output_names`, `environment_snapshot`,
version fields in `environment`, or required task-contract fields. Artifact and
fixture checksum matching, when required, is completed by the caller or Devtest
harness before this skill is invoked. A route manifest with warnings is
not an execution pass.

## Gate 2: framework/reference

Run the source framework on a fixed fixture set and save raw outputs. Use eval
mode, deterministic seeds, disabled augmentation, fixed decode, and the exact
production preprocessing. Save stable fixture identities and output metadata; hashes are optional provenance.

## Gate 3: ONNX

Run the graph/manifest contract check and raw-output parity explicitly:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/multitask/validation/validate_artifact.py \
  manifest.json --onnx model.onnx
PYTHONDONTWRITEBYTECODE=1 python scripts/multitask/validation/compare_outputs.py \
  reference.npz onnx-runtime.npz --atol 1e-5 --rtol 1e-4
```

The artifact validator runs checker and shape inference, derives graph-owned
input/output names, rank, dimensions, and dtype, and rejects contradictory
manifest metadata. Compare raw ONNX Runtime outputs to the framework with
task-appropriate recorded tolerances; do not compare post-NMS detections only
when raw logits can be compared.

## Gate 4: TensorRT

 Build dynamic models with explicit min/opt/max profiles, precision, workspace,
 and plugins. Static models must not be given shape-profile flags; their input
 batch must match the requested batch.
Run `trtexec` with the same fixtures or deterministic synthetic inputs. Compare
bindings, shapes, finite values, and task metrics. Record engine hash and build
log. A successful engine build without parity is a failure. Deserialize a
supplied engine only after recording `--engine-provenance` and resolving it as
a regular, non-symbolic-link file inside the caller-approved `--trusted-root`
model directory. The manifest must also contain `execution_approval` with
`approved=true`, caller identity in `approver`, a timezone-bearing ISO-8601
`approved_at`, the exact `trusted_root`, the provenance record path in
`engine_provenance`, and `scope=deserialize-supplied-tensorrt-engine`. The calling agent records it only
after explicit caller approval; absent approval is `BLOCKED`. Treat serialized
engines as native-code artifacts and run untrusted third-party plans only in an
external isolated container/sandbox. An engine existence check without
deserialization is `UNTESTED`. Verify a caller-approved engine with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/multitask/validation/validate_artifact.py \
  manifest.json --engine model.engine --run-trtexec \
  --trusted-root /data/models --engine-provenance /data/models/build.json
```

## Gate 4.5: adapter implementation and candidate config

For `ADAPTER_REQUIRED`, generate the executable type adapter and merged
manifest from a complete supplied tensor and semantic contract. The generator
sets `implementation_status=implemented` only for classification,
segmentation, 2D pose, or monocular depth and rejects incomplete contracts.
Before candidate execution, require a valid manifest and graph, all required
hook fields, parseable referenced Python files, present named callables, no
empty or `NotImplemented` callable bodies, and an existing absolute native
preprocessing-library path when required.
`generate_configs.py --allow-implemented-adapter` enforces these static checks
and emits test-only configs. An implemented adapter is not production-ready.
Preserve `adapter-validation-required.json` until all eight
`validation_evidence` fields,
including `overlay_evidence`, are populated from real supplied-input results.

## Gate 5: DeepStream

Run a one-stream `nvinfer` or adapter-owned pipeline with `GST_DEBUG=2`. Record
startup, nonzero buffers/frames, EOS, required tensor/object/task metadata, and
the tensor names received by the postprocessor. For video tasks, save at least
one decoded output and one machine-readable output. If NVENC is unavailable,
make exactly one sink-only repair from `nvv4l2h264enc` to
`theoraenc + oggmux` with I420 input and a 300-second timeout; change the output
extension to `.ogv` and do not change model, preprocessing, metadata, renderer
behavior, or any other sink property. `x264enc`, `openh264enc`, and
`avenc_h264` are prohibited. If Theora is unavailable, record capture as
`skipped`; set `visual_status=UNTESTED`,
`pipeline_visual_result=UNTESTED`, and strict `overall_result=UNTESTED` (or
`BLOCKED` when rendered evidence is mandatory). Do not claim visual or overall
PASS without another valid rendered artifact.
Only after the functional gate passes may a production multi-stream benchmark
run. A separately authorized diagnostic capacity attempt must retain failed
backend/semantic gates and cannot promote the adapter. Detection's original
benchmark uses batch size equal to logical streams. The task runner records
mux batch, each inference batch, and its frame/ROI unit separately; do not
force a static B1 engine or secondary ROI batch to equal camera count. Do not infer a
non-detection capacity ceiling from a one-stream run or TensorRT throughput.

## Gate 5.5: visual input/output inspection

For every materialized artifact, run the visual probe from the caller-owned
validation environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  skills/deepstream-import-vision-model/scripts/multitask/visual/visual_io_probe.py \
  /path/to/inventory.json --output-dir /path/to/visual-io
```

For the final rendered-output gate, pass the independently established pipeline
status to `validate_rendered_output.py --deepstream-status PASS`. Also pass
`--overlay-evidence overlay-evidence.json`. The renderer writes that sidecar with
`frame_id`, optional `roi_id`, renderer identity, semantic `overlay_fields`, and
SHA-256 values for the exact structured-output file and rendered frame. The
validator matches both hashes, frame/ROI identity, and semantic fields and
requires measurable changed pixels against the undecorated source frame.
The resulting `pipeline_visual_result` is PASS only when both DeepStream and
model-aware visual checks pass.

```json
{
  "frame_id": 42,
  "roi_id": "full-frame",
  "renderer": "adapter.py:render",
  "overlay_fields": ["label", "score"],
  "structured_output_sha256": "64 lowercase hexadecimal characters",
  "rendered_frame_sha256": "64 lowercase hexadecimal characters"
}
```

The probe creates one case image per catalog case and a montage. Its fixture is
non-zero, deterministic, and shape-preserving: image-like inputs are rendered
as RGB/gray panels, audio and feature inputs as heatmaps, and scalar/vector
inputs as bars. Executed outputs are rendered with the same tensor names and
include shape, dtype, finite ratio, and range. Depth/disparity, optical flow,
and high-channel segmentation/feature tensors receive task-aware diagnostic
reductions. Every output must be an actual runtime result; placeholders for
missing artifacts or unsupported operators are `UNTESTED`. When ONNX Runtime
cannot load a graph but TensorRT builds and runs it, export deterministic raw
bindings with `trtexec --dumpRawBindingsToFile` and pass that directory through
`scripts/multitask/visual/render_trt_visual.py`; record TensorRT as the output backend.

This gate checks visibility of data flow and catches empty, constant, malformed,
non-finite, or channel/layout-confused tensors. It does not establish semantic
correctness: use the task metrics below and the model's decoder/reference
fixtures for that. A live display claim additionally needs a captured window
and display probe from a DeepStream pipeline.

## Task metrics and default acceptance thresholds

Before executing a semantic gate, read
[../semantic-oracles.md](../semantic-oracles.md) and run:

```bash
python3 scripts/multitask/validation/validate_semantic_oracle.py \
  oracle.json --task <detection|classification|segmentation|pose|depth>
```

Exit `2` is `BLOCKED`; do not run promotion. For each validated sample, resolve
its expected payload, run the category validator, and bind the result to the
fixture/frame/ROI identity plus oracle and structured-output hashes.

Record the metric, precision, threshold, observed value, and result. Use raw
allclose `atol=1e-5, rtol=1e-4` for FP32 and `atol=1e-3, rtol=1e-2` for FP16.
INT8 requires a task metric. Apply a stricter model-owned threshold when
documented; never relax a default merely to turn a failure into a pass.

- classification: exact top-1 and top-k set agreement, plus score/logit error;
- segmentation: mean IoU >= 0.99 (>= 0.98 for INT8), shape, and class range;
- pose: PCK/OKS agreement >= 0.99, keypoint count/order, and coordinate range;
- depth: direct RMSE <= 1% of reference range, finite ratio, units/range;
- detection: delegate parser/runtime checks to the vision skill; score IoU,
  precision/recall, and mAP on reviewed annotations. Do not use all-frame
  nonempty-detection percentage as a correctness gate unless the oracle says
  every scored frame must contain a target.

## Result labels

Gate outcomes use `PASS`, `FAIL`, `BLOCKED`, or `UNTESTED`. Routing additionally
uses `DELEGATED` and `ADAPTER_REQUIRED`. Every non-PASS value must name the
missing contract or failed assertion and link its evidence.


## Orthogonal result record

Do not compress routing and execution into one Pass/Fail cell. Record:

| Field | Allowed values | Meaning |
|---|---|---|
| `route_status` | `READY`, `DELEGATED`, `ADAPTER_REQUIRED`, `BLOCKED` | Whether this skill owns an executable route. |
| `artifact_status` | `PASS`, `BLOCKED` | The caller-supplied local artifact exists; a Devtest harness may separately verify its expected SHA-256. |
| `backend_status` | `PASS`, `FAIL`, `BLOCKED`, `UNTESTED` | Reference, ONNX, and TensorRT result. |
| `deepstream_status` | `PASS`, `FAIL`, `BLOCKED`, `UNTESTED` | Pipeline startup, buffers, EOS, and metadata result. |
| `visual_status` | `PASS`, `FAIL`, `BLOCKED`, `UNTESTED` | Valid video plus visible model-aware task rendering where required. |
| `pipeline_visual_result` | `PASS`, `FAIL`, `BLOCKED`, `UNTESTED` | Devtest runtime verdict; PASS iff `deepstream_status=PASS` and `visual_status=PASS`. |
| `semantic_status` | `PASS`, `FAIL`, `BLOCKED`, `UNTESTED` | Ground-truth or reviewed-oracle result. |
| `overall_result` | `PASS`, `FAIL`, `BLOCKED`, `UNTESTED` | Optional strict aggregate across all required gates; never substitute it for the runtime-visual verdict. |

Use `FAIL` only after an executable gate ran and violated its contract. Use
`BLOCKED` when an artifact, platform capability, memory requirement, runtime,
or required external component is unavailable. Use `UNTESTED` when execution
did not establish the stated claim, especially when the semantic oracle or
runtime metadata capture is absent.

A route may be `DELEGATED` while its handoff gate is `PASS`. A route may be
`ADAPTER_REQUIRED` while ONNX and TensorRT backend gates are `PASS`. Neither
case is a route-readiness claim. A case may still have
`pipeline_visual_result=PASS` after an implemented adapter runs successfully;
record the unresolved route, backend, or semantic status beside it.

## Root-cause records

For every non-PASS gate, emit one or more deduplicated records with `gate`,
`status`, `category`, `detail`, `evidence`, and `next_action`. Keep platform
compatibility, input readiness, backend parity, pipeline runtime, structured
output, visual rendering, and semantic-oracle causes separate. Do not repeat a
missing semantic oracle as a pipeline failure, and do not call an absent input
bundle a model failure.

## Precision fallback

Run the manifest `precision_policy` in order. The default is FP16, FP16 with
TF32 disabled, then FP32. Stop at the first passing precision, but retain the
earlier failure logs and metrics. Report, for example, `FP16 FAIL; FP32 PASS`;
do not rewrite this as an FP16 pass. INT8 is opt-in and requires calibration
provenance and task metrics.

## ARM media preflight

On ARM SBSA and Jetson, record codec, profile, chroma format, bit depth,
resolution, and frame count before inference. If the target decoder rejects the
source, retain the original and create one deterministic H.264/YUV420
derivative. Hash both, record the exact transcode command, and run the semantic
oracle against reviewed frames from the derivative. Transcoding repairs media
compatibility only; it cannot repair a wrong task fixture.
