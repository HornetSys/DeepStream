<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared Step 8 capacity tables

Keep Sections 8–10 named **DeepStream Benchmark Results**, **trtexec vs
DeepStream Comparison**, and **Efficiency Analysis** for every task. Section 8
answers the tested realtime stream count, not merely whether two independent
rendered pipelines ran. Show calibration and validation with batch, streams,
average/worst sustained FPS per source, aggregate frames/s, and ≥30 FPS for
every source in the final five measurement windows. Preserve failed trials.
Do not label the highest passing tested count an absolute maximum without a
boundary search. Keep rendered/encoded diagnostics in a separate subsection.

For status/validation reports, optionally add `benchmark` to `case-evidence.json`:

```json
{
  "benchmark": {
    "engine": {"path": "engines/final.engine", "sha256": "observed digest"},
    "input_unit": "frame",
    "scope": "decode-mux-infer-parser-fakesink",
    "trtexec": {
      "bs1": {"batch": 1, "log": {"path": "benchmarks/b1/trtexec_b1.log", "sha256": "observed digest"}},
      "bsmax": {"batch": 64, "log": {"path": "benchmarks/b64/trtexec_b64.log", "sha256": "observed digest"}}
    },
    "deepstream": {
      "run1": {"streams": 36, "batch": 36, "exit_code": 0, "config": {"path": "config/nvinfer-b36.txt", "sha256": "observed digest"}, "log": {"path": "benchmarks/ds/ds_s36_run1.log", "sha256": "observed digest"}},
      "run2": {"streams": 33, "batch": 33, "exit_code": 0, "config": {"path": "config/nvinfer-b33.txt", "sha256": "observed digest"}, "log": {"path": "benchmarks/ds/ds_s33_run2.log", "sha256": "observed digest"}}
    }
  }
}
```

Replace example paths/counts/digests with executed evidence. Paths and the
runtime paths recorded inside logs/configs must resolve in the reporting
environment. The reporter reparses logs, checks hashes, successful GPU-only
trtexec execution, input batches, and matching DeepStream engine/config batches.
Invalid supplied evidence stops PDF conversion. No functional/semantic/visual
gate or adapter readiness is changed by these tables.

`input_unit=roi` preserves ROI/s versus application frames/s and reports
efficiency and engine-derived camera capacity as N/A. For a cascade, benchmark
the complete upstream graph with `scope=task-application-fakesink`; do not
copy a detector ceiling into a classifier, pose, or OCR application. A scope
label is not evidence that the complete graph ran: retain its executed graph
and upstream component records in the case manifest. Static B1 models need a
compatible capacity runner/profile before a batched benchmark can be claimed.

With comparable frame measurements, the existing five-chart generator and
canonical HTML/PDF converter are reused. Without verified capacity runs, retain
UNTESTED capacity/comparison instead of extrapolating from trtexec, EOS, or a
two-pipeline diagnostic. Legacy successful detection benchmark reports are
preserved unchanged.

## Task-aware capacity execution

Use `scripts/multitask/deepstream/run_task_capacity.py PLAN.json --output-dir
FRESH_DIRECTORY` inside the approved DeepStream container, after caller-owned
engine/profile/plugin validation. The JSON plan supplies `task`, `video`,
`streams`, `mux_batch`, `width`, `height`, ordered `stages` with local `config`
paths, `decoder_kind`, `campaign_id`, and optionally `duration_seconds` (60–300),
`warmup_seconds` (0–30), and `target_fps` (default 30). Decoder kinds are
`native-detection`, `native-classification`, and `tensor-hook`; tensor routes
require a caller-reviewed `decoder` file.py:decode receiving named arrays and
frame context and returning a finite, JSON-serializable task record. Execute
only reviewed local hooks; never load code obtained from model output.

The runner repeats the pinned video as logical sources, executes every declared
upstream inference stage and the task decoder, and counts per-source frames
before a fakesink. It excludes display, rendering, and encoding. Every source
needs task evidence and five complete sustained two-second windows. Its result
records actual inference batches and their units, engine/config/decoder hashes,
all windows, and errors. Require task records for every source in each final
window; a target-free fixture that cannot establish that evidence does not
certify capacity. ROI/s is not application frames/s. Static B1 inference
may process muxed frames sequentially; the actual measured graph, not an
invented dynamic engine profile, determines capacity.

Run calibration and validation separately, retain all failed trials, and bound
convergence to five decreasing counts. A highest passing tested count is not an
absolute maximum without a tested failing boundary. Preserve the 30-FPS column
even if the caller additionally accepts 5 FPS; show that acceptance separately.
These capacity measurements cannot change backend, semantic, visual, or adapter
readiness gates. Native detection's benchmark table/schema remains unchanged.

Bind task results to Step 8 using `benchmark.runner="task-capacity"`, the same
`engine`, `input_unit`, and `trtexec` records shown above, plus `deepstream.run1`
and `run2` containing `streams`, hash-pinned `result` JSON, and hash-pinned
executed `log`. Include matching `task` and `campaign_id` in the case manifest.
The collector reparses the logged windows, requires per-window task producers,
and checks every stage config, engine, batch, and decoder identity. A cascade
uses `input_unit="roi"`; comparison plots retain separate ROI/s and frames/s
axes and efficiency stays N/A. Never reinterpret a static B1 engine as a
dynamic batched engine. Invalid or older producer-incomplete records require a
fresh run; they cannot fill these report tables.
