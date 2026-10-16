<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Evidence-backed report — Step 8

For shared capacity and comparison tables, read
[benchmark-evidence.md](benchmark-evidence.md). Keep batched stream capacity
separate from rendered/encoded diagnostics; neither substitutes for the other.

For a successful case, this phase runs after Step 7 validation and any adapter
promotion. For an early stop, it documents the partial evidence and first
causal blocker without running skipped phases. It is not a substitute for a
DeepStream run. Keep all reports in `models/$MODEL_NAME/reports/` in the
caller's mounted work root, as in the original detection skill.

## Thirteen-application campaign

Use [`../assets/vision13-cases.json`](../assets/vision13-cases.json) as the exact
case/task registry: TrafficCamNet, PeopleNet, YOLOv8n, RT-DETR, LPD-US,
VehicleMakeNet, VehicleTypeNet, CitySemSegFormer, SegFormer, BodyPose2D,
Depth Anything V2, FoundationStereo, and **one complete Bi3D application**.
Bi3D FeatNet and SegNet are components of one case, not separate passes.
`vision13` is a test suite, not a new public task category: stereo cases stay
experimental and the five-category router remains unchanged. The source and
media map is in [model-zoo-media.md](model-zoo-media.md).
`vision14` adds one pinned primary ImageNet classifier,
`mobilenetv2-imagenet`, to those 13 cases; it does not replace either
vehicle-ROI classifier. Its source is
[`onnx-community/mobilenet_v2_0.35_96` at commit `f37d9b91354e5a43dae384840f46119c67a76d58`](https://huggingface.co/onnx-community/mobilenet_v2_0.35_96/tree/f37d9b91354e5a43dae384840f46119c67a76d58),
with `onnx/model.onnx`, `config.json`, and `preprocessor_config.json`. Validate
its 1,001-way output and center-crop/normalization contract from those pinned
files before runtime. The generated dog image may be used as a synthetic
diagnostic fixture; it does not automatically supply a reviewed ImageNet oracle.
For 96×96 media, pass `--mux-width 96 --mux-height 96` to the classifier
reference runner. Its default remains 1280×720. If hardware H.264 encoder
initialization fails, retain the first failure and make at most one fresh-output
retry with `--encoder theora`; this changes only the output sink. A larger
square presentation fixture may preserve the same central crop while allowing
the class label overlay to remain legible. A reviewed label on one generated
dog image proves only that fixture, not ImageNet or real-world accuracy. Keep
an FP16 parity failure visible if FP32/no-TF32 repair is used.

1. Prepare an input JSON with `schema_version=1`, `suite="vision13"` (or
   `"vision14"` when the extra ImageNet classifier is requested),
   `campaign_id`, and exactly the registered `cases`. Each case supplies
   `case_id`, `task`, exact local `source`, required `components`, media
   `inputs` with roles, `oracle`, `upstream_stages`, and stereo `calibration`
   where applicable. Run `prepare-campaign.py INPUT_JSON models/` only into a
   fresh empty `models/` directory. It validates suite membership and hashes
   available files; it does **not** run tests or mark a gate PASS.
2. Route each case through Step 0 first. The suite registry is not permission
   to execute unsupported tasks. FoundationStereo and Bi3D are stereo and
   therefore `BLOCKED`/unsupported by the public five-category router: stop
   without conversion, engine build, DeepStream, or an adapter promotion
   attempt, then produce only a status report. For a supported case whose
   prerequisites pass, continue the same skill's applicable Steps 1–7 on a
   single selected GPU: inspect the tensor/preprocess/postprocess contract,
   build or validate its engine, run its task-specific DeepStream graph, and
   gate backend parity, structured output, semantic oracle, and the **rendered
   frame**. Required upstream stages must produce evidence bound to the case
   input: vehicle ROIs for LPD-US and secondary classifiers and person ROIs
   for pose where required. Stereo would additionally need real synchronized
   calibrated pairs and an implemented paired-input runner before any future
   scope expansion. A generated adapter must pass the bounded development and
   readiness gates before functional PASS. Preserve the legacy detector
   parser, KITTI, one-stream, and two-run benchmark checks.
3. Record executed commands, input/output hashes, first causal failure, all
   four gates, and capacity separately in each `case-evidence.json`. If the
   model, stage, media, adapter, or oracle is missing, record the correct
   `BLOCKED`, `ADAPTER_REQUIRED`, or `UNTESTED` gate; do not substitute another
   model, shifted mono video for stereo, or an unrelated SDK clip. A generated
   plate scene may demonstrate LPD-US pipeline/visual behavior, but cannot
   certify accuracy on real traffic.
4. Run `generate-model-report.py` for **every registered case**, including
   blocked cases. Then run `generate-campaign-index.py models/campaign.json`.
   With a named suite, the index rejects missing/extra case IDs, wrong
   task mappings, missing PDFs, or digest mismatches and shows separate
   pipeline+visual, functional, and capacity statuses. A report or index does
   not run inference, fill missing evidence, or convert a diagnostic result
   into a functional PASS.

Keep the original detection benchmark/report path below. Do not replace its
12-section/five-chart contract with a generic 13-row summary. The campaign
index is an additional summary of the thirteen independent application runs.

For every case, write `models/$MODEL_NAME/case-evidence.json` with
`schema_version=1`, `case_id` matching the model directory, `task`, pinned
`revision`, `source` (`path` and SHA-256), `inputs` (`role`, `path`, and
SHA-256), optional application `components`, `oracle`, the validator-produced
`oracle_validation` record and `oracle_status`,
`upstream_stages`, `commands` (`phase`, `exit_code`, `log` with path and
SHA-256), `gates` (`backend`,
`pipeline`, `semantic`, `visual`, each with a recorded status), `evidence`,
`capacity`, and `first_failure` when applicable. The reporter computes actual
file digests and validates declared digests; caller/Devtest matching against
an expected model or fixture checksum stays in the test case, not in this skill.
Stereo cases require exactly one distinct `left` and `right` input plus a
nonempty calibration file recorded by path and observed SHA-256; naming a
missing calibration path or reusing one image twice does not pass intake.
Every declared upstream stage must have `evidence.upstream_{stage}` pointing to
a SHA-256-pinned JSON record with `schema_version=1`, the matching `stage`,
`status=PASS`, `kind=inference|transform`, a nonempty `producer`, ordered
`input_fixture_sha256s` matching the case inputs, and a nonempty `output`
artifact with its SHA-256. Inference records must bind `model_sha256` to a
declared source/component artifact; transforms must not claim a model. A stage
name or an unbound file alone cannot satisfy a cascade prerequisite.

Use `scripts/report/prepare-campaign.py` to create fresh manifests from the
caller's input list. It only records file identity and may decode source frames;
it never invokes a model or sets a gate to `PASS`. Do not run it over a nonempty
campaign directory. It separates `input_blockers` (missing model, component,
media, or required stereo calibration) from `certification_limitations`
(such as an unreviewed oracle); neither list asserts a pipeline ran.

Select model artifacts and task-appropriate SDK/caller media using
[model-zoo-media.md](model-zoo-media.md). NGC model files and DeepStream sample
streams are separate sources. The case manifest must name actual local files,
not catalog pages or repository directories. A missing reviewed oracle limits
semantic certification; if source/media and required stages are available,
continue a **diagnostic** backend/DeepStream/visual attempt and retain semantic
`UNTESTED`. Do not convert that diagnostic result into a functional `PASS`.

Only a complete functional, semantic, and model-derived visual pass may create
a successful benchmark/validation report. Non-detection PASS also requires a
tested `adapter-status.json` from `validate_adapter_readiness.py`, including
all eight verified producer records. Detection PASS additionally requires the
legacy benchmark Markdown, data JSON, five charts, KITTI output, and embedded
runtime input/output frames. Its `backend_parity`, `deepstream_runtime`,
`semantic_validation`, and `visual_validation` JSON records must bind the same
campaign, model digest, and fixture digest. Runtime must show startup, buffers,
EOS, and exit 0; the oracle must be reviewed PASS; visual evidence must show a
model-derived overlay PASS, and the visual record's source/rendered-frame and
structured-output digests must match the files in the report. Every successful
case must provide valid PNG/JPEG source and rendered frames with declared
SHA-256 digests; the PDF embeds those exact bytes. Non-detection PASS also binds
the overlay's input-frame digest and the rendered frame to the adapter's
validated output. Missing
inputs, an incomplete adapter, a failed gate, or a PDF conversion failure
never become PASS.

Run the common reporter after recording the first causal failure even when a
case cannot proceed. A status report may use partial evidence without complete
model/media/oracle intake; record missing prerequisites and leave unexecuted
gates `UNTESTED`. It cannot claim functional or semantic PASS. A visual PASS
is recorded only when an actual model-derived overlay passed its visual gate
on available media; a report cannot manufacture one. If the reporter or PDF
dependencies are unavailable, record that as an additional
blocker rather than claiming that a PDF exists. Exit `4` means it wrote a
truthful non-PASS status report;
use `report-result.json` to distinguish that result from a conversion failure:

```bash
build/.venv_optimum/bin/python \
  .claude/skills/deepstream-import-vision-model/scripts/report/generate-model-report.py \
  "models/$MODEL_NAME/case-evidence.json"
```

It writes `report-result.json` and one of these families under `reports/`:

| Outcome | Markdown / HTML / PDF |
|---|---|
| Detection functional PASS | `benchmark_report.md`, `benchmark_report.html`, `benchmark_report_${MODEL_NAME}.pdf` |
| Other supported category functional PASS | `validation_report.md`, `validation_report.html`, `validation_report_${MODEL_NAME}.pdf` |
| FAIL, BLOCKED, ADAPTER_REQUIRED, or experimental stereo | `status_report.md`, `status_report.html`, `status_report_${MODEL_NAME}.pdf` |

Store a campaign's case IDs in `models/campaign.json` as a unique `cases`
array with `schema_version=1`; after all cases have a report, run
`scripts/report/generate-campaign-index.py models/campaign.json`. The index
refuses missing or digest-mismatched PDFs. A status PDF documents what was
actually attempted and the next repair; it is never a successful model report.
If a report was generated in a container, its recorded PDF path may start at a
container mount such as `/work/models/`. The index may resolve that path only
to the exact registered case's report filename under the local campaign root,
then rechecks the recorded SHA-256; it never trusts a same-named PDF elsewhere.

Capacity is separate from functional readiness. Set `capacity.status=PASS`
only with a task-compatible runner, at least two measured logical streams,
`fps_per_stream >= target_fps`, and a real log with a matching SHA-256. For
stereo, one synchronized left/right pair is one logical stream. Otherwise use
`FAIL` or `UNTESTED` as observed; never extrapolate the old detector ceiling to
another task.

## Detection benchmark contract

Generate a fail-closed benchmark report only after engine, KITTI, visual, and
two-run DeepStream gates pass. The model directory is `$ARGUMENTS`.

## Required evidence

Stop `BLOCKED` if any required item is absent, empty, or unparsable:

- final engine selected by `scripts/model/resolve-engine.sh`;
- fixed trtexec logs `trtexec_b1.log` and `trtexec_b{MAX_BS}.log`;
- newest converged `ds_s{N}_run2.log` plus calibration `ds_s{N}_run1.log`;
- KITTI frame count, frames with detections, detection rate, and top classes;
- model-derived rendered output and encoder mode;
- model, preprocessing, environment, source revision/version, and timing data.

Do not use placeholders, guessed values, or a failed run as the final run.
Select Run 2 by modification time because convergence steps downward. Pair it
with the newest Run 1 that is not newer than that Run 2; Run 1 and Run 2 may
intentionally use different stream counts.

## Collect benchmark data

Use the helpers extracted on main, now hardened to propagate errors and read
all sources rather than extrapolating source 0:

```bash
source .claude/skills/deepstream-import-vision-model/scripts/report/collect-report-vars.sh "$ARGUMENTS" || exit 4
build/.venv_optimum/bin/python .claude/skills/deepstream-import-vision-model/scripts/report/write-benchmark-data.py
```

Require at least five complete `**PERF:` windows. Sum all sources' measured
average FPS; test every source's minimum over the final five windows. A high
average cannot conceal a slow source. Preserve failed calibration/retry logs
and pair the newest converged Run 2 with the newest Run 1 not newer than it.

Compute GPU images/s as QPS × batch, GPU upper-bound streams as
`floor(images/s / 30)`, and comparable-scope DS efficiency as measured DS
aggregate FPS ÷ GPU images/s × 100. Reject non-finite or missing measurements.
Bind the engine and logs with their real paths and hashes in the case evidence
manifest. The JSON helper refuses to replace an existing output; start a fresh
campaign or preserve the earlier report before writing a new one.

## Generate exactly five charts

Run the canonical chart generator in the shared environment:

```bash
build/.venv_optimum/bin/python \
  .claude/skills/deepstream-import-vision-model/scripts/report/generate-benchmark-charts.py \
  "models/$MODEL_NAME/reports/charts" \
  "models/$MODEL_NAME/reports/benchmark_data.json"
```

Require exactly these nonempty files:

1. `chart_trtexec_bs1_vs_bsmax.png`
2. `chart_trtexec_throughput.png`
3. `chart_ds_streams_vs_fps.png`
4. `chart_trt_vs_ds.png`
5. `chart_efficiency.png`

Do not generate host-transfer charts. Use 30 FPS as the real-time threshold.

## Write the Markdown report

Create `benchmark_report.md` with exactly these numbered sections:

1. **Model Configuration** — source and immutable revision/version, architecture,
   ONNX path/hash, tensors, labels, parser, cluster mode, precision, profile.
2. **System Configuration** — GPU/VRAM, driver, CUDA, TensorRT, DeepStream, OS,
   Python, PyTorch, ONNX, and pinned container image.
3. **Preprocessing** — scale, offsets, color, layout, normalization, resolution.
4. **Engine Build Summary** — conversion path, engine path/hash/size, precision,
   MAX_BS, workspace, timing cache, and build log.
5. **trtexec Results** — BS=1 and MAX_BS QPS, images/s, GPU mean/P99.
6. **PEAK_GPU_STREAMS Derivation** — formula, inputs, result, and rebuild history.
7. **Single-Stream Validation** — media, KITTI totals/rate/classes, rendered
   artifact, encoder mode, and PASS/FAIL. Copy verified source/rendered frames
   to `reports/assets/source_frame.*` and `rendered_frame.*`; embed both images
   with relative Markdown image links in this section. Their bytes must match
   the `case-evidence.json` frame paths.
8. **DeepStream Benchmark Results** — Run 1 calibration and converged Run 2,
   streams, batch, FPS/stream, total images/s, and real-time result.
9. **trtexec vs DeepStream Comparison** — engine, batch/streams, total images/s,
   FPS/stream, real-time result, and efficiency.
10. **Efficiency Analysis** — both efficiency calculations and an evidence-based
    GPU/pipeline bottleneck verdict.
11. **Pipeline Timing** — Steps 1–8 and total wall time.
12. **Reference Commands** — exact executed acquisition, export/build, trtexec,
    parser, single-stream, KITTI, and benchmark commands; no placeholders.

Embed all five charts with relative `charts/...png` paths. Report
`nvv4l2h264enc MP4`, `theoraenc OGV (NVENC unavailable)`, or
`skipped (no encoder available)` exactly. A skipped capture cannot pass the
visual gate: report `visual_status=UNTESTED`,
`pipeline_visual_result=UNTESTED`, and strict `overall_result=UNTESTED` or
`BLOCKED` when rendered evidence is mandatory.

## Generate HTML and PDF

Use only the canonical converter and stylesheet:

```bash
build/.venv_optimum/bin/python \
  .claude/skills/deepstream-import-vision-model/scripts/report/md-to-html-pdf.py \
  "models/$MODEL_NAME/reports/benchmark_report.md" \
  .claude/skills/deepstream-import-vision-model/scripts/report/report-style.css \
  "models/$MODEL_NAME/reports/" "$MODEL_NAME"
```

It must produce nonempty `benchmark_report.html` and
`benchmark_report_${MODEL_NAME}.pdf`. Do not call pandoc, pdflatex,
`wkhtmltopdf`, or a custom HTML generator directly.

## Final gate

Before PASS, verify all 12 headings, five chart files, HTML, PDF, evidence JSON,
source/log hashes, KITTI rate >=90%, and rendered model output. A measured Run 2
at or above 30 FPS per stream is required for **capacity** PASS, not for a
separate functional PASS. Preserve failed logs and report the first causal
failure as `FAIL`, `BLOCKED`, or `UNTESTED`; never downgrade it to a warning.

## Troubleshooting

- Missing fixed log: rerun the producing phase; do not search arbitrary logs.
- Wrong Run 2: select newest by modification time, not highest stream count.
- Missing chart: repair `benchmark_data.json`, then rerun the chart generator.
- HTML/PDF failure: run the canonical converter and inspect its stderr.
- Missing encoder: install the documented in-container GStreamer fallback or
  run on an NVENC-capable system; visual status remains blocked until rerun.

## Final visual inspection

The detection sections and chart names above retain the original benchmark
contract. Inspect a rendered PDF page: file size alone does not prove the five
charts and actual input/output frames are visible. Keep failed conversion logs
and a non-PASS status. For non-detection cases, use task-specific structured
metrics and rendered evidence only after their gates pass; do not copy detector
capacity data into them. One-stream EOS leaves multi-stream capacity
`UNTESTED`. Bi3D remains one application with two components, and stereo
needs real calibrated paired inputs before any future readiness claim.
