<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->


# NV Engine Build -- Steps 4-5

Build a TensorRT engine from ONNX and derive PEAK_GPU_STREAMS for DeepStream sizing.

Pass the ONNX model path explicitly through `ONNX_PATH`. Example:

```bash
ONNX_PATH="models/<model_name>/model/<model>.onnx"
```

## Pre-flight: Validate Inputs and Extract Variables

Before anything else, validate `ONNX_PATH` and derive the remaining variables:
```bash
: "${ONNX_PATH:?set ONNX_PATH to the exact local .onnx artifact}"

# Derive MODEL_NAME from directory structure: models/{MODEL_NAME}/model/...
MODEL_NAME=$(echo "$ONNX_PATH" | sed 's|models/\([^/]*\)/.*|\1|')

# Derive MODEL_FILENAME as the ONNX basename without extension
MODEL_FILENAME=$(basename "$ONNX_PATH" .onnx)

# MAX_BS drives --optShapes, --maxShapes, and the engine filename postfix
# Starting value is 64 — will double iteratively in Step 5 if PEAK_GPU_STREAMS > 64
MAX_BS=64

echo "Model:    $MODEL_NAME"
echo "File:     $MODEL_FILENAME"
echo "ONNX:     $ONNX_PATH"
echo "Engine:   models/$MODEL_NAME/benchmarks/engines/${MODEL_FILENAME}_dynamic_b${MAX_BS}.engine"

# Verify ONNX file exists
ls -lh "$ONNX_PATH" || { echo "ERROR: ONNX file not found at $ONNX_PATH"; exit 1; }

# trtexec lives at a fixed path IN the container (not on PATH); read the TRT version from it.
TRTEXEC=/usr/src/tensorrt/bin/trtexec
[ -x "$TRTEXEC" ] || { echo "ERROR: trtexec not found at $TRTEXEC — is this the DeepStream/TensorRT image?"; exit 1; }
# `head -2` closes the pipe early, so trtexec takes SIGPIPE; under `set -o pipefail` that
# surfaces as exit 141 and aborts the calling script. Engine/log existence is checked
# explicitly below, so the version banner is advisory only.
"$TRTEXEC" --version 2>&1 | head -2 || true

# Verify GPU is available (in-container)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

If the ONNX file doesn't exist, inform the user to run Steps 1-3 first (see references/model-acquire.md).

> All subsequent commands use `$MODEL_NAME`, `$MODEL_FILENAME`, `$MAX_BS`, and `$TRTEXEC` — never hardcoded paths or template placeholders.

Inspect the ONNX model and auto-parse input name and spatial dimensions:
```bash
INSPECT_OUT=$(python3 .claude/skills/deepstream-import-vision-model/scripts/model/inspect-onnx.py "$ONNX_PATH")
echo "$INSPECT_OUT"

INPUT_NAME=$(echo "$INSPECT_OUT" | grep -oP 'input_name:\s*\K\S+')
H=$(echo "$INSPECT_OUT"          | grep -oP 'height:\s*\K[0-9]+')
W=$(echo "$INSPECT_OUT"          | grep -oP 'width:\s*\K[0-9]+')

echo "INPUT_NAME=$INPUT_NAME  H=$H  W=$W"
[ -z "$INPUT_NAME" ] && { echo "ERROR: could not parse INPUT_NAME from inspect output"; exit 1; }
# Dynamic spatial dimensions must resolve from the supplied task contract,
# pinned config.json/model card, or caller-declared MODEL_INPUT_H/W. Try only
# the explicitly declared candidate(s); never guess a common resolution.
H="${H:-${MODEL_INPUT_H:-}}"
W="${W:-${MODEL_INPUT_W:-}}"
[ -z "$H" ] && { echo "BLOCKED: dynamic H is unresolved; supply MODEL_INPUT_H from pinned metadata" >&2; exit 5; }
[ -z "$W" ] && { echo "BLOCKED: dynamic W is unresolved; supply MODEL_INPUT_W from pinned metadata" >&2; exit 5; }
```

## Step 4: Build TensorRT Engine

Build one dynamic engine optimized for BS=64. `opt=max=64` ensures TRT optimizes kernels for
the exact batch size used for benchmarking and DeepStream. `min=1` handles single-stream validation.

```bash
STEP4_START=$(date +%s.%N)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Validate every component before deriving artifact paths. These values must
# never contain separators, traversal components, or shell metacharacters.
case "$MODEL_NAME" in ""|.|..|*[!A-Za-z0-9._-]*) echo "ERROR: unsafe MODEL_NAME" >&2; exit 2;; esac
case "$MODEL_FILENAME" in ""|.|..|*[!A-Za-z0-9._-]*) echo "ERROR: unsafe MODEL_FILENAME" >&2; exit 2;; esac
case "$MAX_BS" in ""|*[!0-9]*) echo "ERROR: MAX_BS must be a positive integer" >&2; exit 2;; esac
[ "$MAX_BS" -gt 0 ] || { echo "ERROR: MAX_BS must be positive" >&2; exit 2; }
# benchmarks/engines/ already exists from model-acquire;
# mkdir -p kept here as a safety net for standalone use
mkdir -p models/$MODEL_NAME/benchmarks/engines models/$MODEL_NAME/benchmarks/b1 models/$MODEL_NAME/benchmarks/b${MAX_BS}
ENGINE_TARGET="models/$MODEL_NAME/benchmarks/engines/${MODEL_FILENAME}_dynamic_b${MAX_BS}.engine"
ENGINE_CANDIDATE="${ENGINE_TARGET}.partial.${TIMESTAMP}.$$"
BUILD_LOG="models/$MODEL_NAME/benchmarks/engines/${MODEL_FILENAME}_dynamic_build_${TIMESTAMP}.log"
[ ! -e "$ENGINE_CANDIDATE" ] || { echo "ERROR: candidate engine path already exists: $ENGINE_CANDIDATE" >&2; exit 2; }
set -o pipefail
if $TRTEXEC \
  --onnx="$ONNX_PATH" \
  --minShapes=$INPUT_NAME:1x3x${H}x${W} \
  --optShapes=$INPUT_NAME:${MAX_BS}x3x${H}x${W} \
  --maxShapes=$INPUT_NAME:${MAX_BS}x3x${H}x${W} \
  --fp16 --skipInference \
  --memPoolSize=workspace:32768M \
  --timingCacheFile=models/$MODEL_NAME/benchmarks/engines/timing.cache \
  --saveEngine="$ENGINE_CANDIDATE" 2>&1 | tee "$BUILD_LOG"; then
  BUILD_STATUS=("${PIPESTATUS[@]}")
else
  BUILD_STATUS=("${PIPESTATUS[@]}")
fi
if [ "${BUILD_STATUS[0]:-1}" -ne 0 ] || [ "${BUILD_STATUS[1]:-1}" -ne 0 ] || [ ! -s "$ENGINE_CANDIDATE" ]; then
  echo "ERROR: TensorRT engine build failed — preserved $BUILD_LOG and candidate $ENGINE_CANDIDATE" >&2
  exit 1
fi
if [ -e "$ENGINE_TARGET" ]; then
  ENGINE_BACKUP="${ENGINE_TARGET}.previous.${TIMESTAMP}"
  [ ! -e "$ENGINE_BACKUP" ] || { echo "ERROR: backup path already exists: $ENGINE_BACKUP" >&2; exit 2; }
  mv -- "$ENGINE_TARGET" "$ENGINE_BACKUP"
  echo "Preserved previous engine: $ENGINE_BACKUP"
fi
mv -- "$ENGINE_CANDIDATE" "$ENGINE_TARGET"

STEP4_END=$(date +%s.%N)
STEP4_DURATION=$(python3 -c "print(round($STEP4_END - $STEP4_START, 2))")   # bc is not in the container; python3 always is
echo "[Step 4] Engine build completed in ${STEP4_DURATION}s"
```

Set the ENGINE variable — used by all subsequent trtexec and DeepStream runs:
```bash
ENGINE="models/$MODEL_NAME/benchmarks/engines/${MODEL_FILENAME}_dynamic_b${MAX_BS}.engine"
```

## Step 5: Benchmark — 2 Runs Only

Run exactly **2 trtexec benchmarks** using the Step 4 engine. No sweep needed.
- BS=1 → latency baseline (single-stream worst case)
- BS=64 → peak throughput → `PEAK_GPU_STREAMS`

```bash
STEP5_START=$(date +%s.%N)
```

### Run 5a — Latency baseline (BS=1)

> Log filename is **fixed** — no timestamp, no variation. Always `trtexec_b1.log`. This ensures the report-generation skill can find it with an exact path, not a wildcard.

```bash
B1_LOG="models/$MODEL_NAME/benchmarks/b1/trtexec_b1.log"
B1_CANDIDATE="${B1_LOG}.partial.$(date +%Y%m%d_%H%M%S).$$"
$TRTEXEC \
  --loadEngine="$ENGINE" \
  --shapes=$INPUT_NAME:1x3x${H}x${W} \
  --noDataTransfers --duration=10 --warmUp=1000 \
  2>&1 | tee "$B1_CANDIDATE"
BENCH_STATUS=("${PIPESTATUS[@]}")
if [ "${BENCH_STATUS[0]:-1}" -ne 0 ] || [ "${BENCH_STATUS[1]:-1}" -ne 0 ]; then
  B1_FAILED="${B1_LOG}.failed.$(date +%Y%m%d_%H%M%S)"
  mv -- "$B1_CANDIDATE" "$B1_FAILED"
  echo "ERROR: BS=1 trtexec benchmark failed; preserved $B1_FAILED" >&2
  exit 1
fi
if [ -e "$B1_LOG" ]; then
  B1_BACKUP="${B1_LOG}.previous.$(date +%Y%m%d_%H%M%S)"
  mv -- "$B1_LOG" "$B1_BACKUP"
fi
mv -- "$B1_CANDIDATE" "$B1_LOG"
```

### Run 5b — Peak throughput (BS=MAX_BS)

> Log filename is **fixed** — always `trtexec_b${MAX_BS}.log`. Updated by the while loop if MAX_BS changes.

```bash
BMAX_LOG="models/$MODEL_NAME/benchmarks/b${MAX_BS}/trtexec_b${MAX_BS}.log"
BMAX_CANDIDATE="${BMAX_LOG}.partial.$(date +%Y%m%d_%H%M%S).$$"
$TRTEXEC \
  --loadEngine="$ENGINE" \
  --shapes=$INPUT_NAME:${MAX_BS}x3x${H}x${W} \
  --noDataTransfers --duration=10 --warmUp=1000 \
  2>&1 | tee "$BMAX_CANDIDATE"
BENCH_STATUS=("${PIPESTATUS[@]}")
if [ "${BENCH_STATUS[0]:-1}" -ne 0 ] || [ "${BENCH_STATUS[1]:-1}" -ne 0 ]; then
  BMAX_FAILED="${BMAX_LOG}.failed.$(date +%Y%m%d_%H%M%S)"
  mv -- "$BMAX_CANDIDATE" "$BMAX_FAILED"
  echo "ERROR: BS=$MAX_BS trtexec benchmark failed; preserved $BMAX_FAILED" >&2
  exit 1
fi
if [ -e "$BMAX_LOG" ]; then
  BMAX_BACKUP="${BMAX_LOG}.previous.$(date +%Y%m%d_%H%M%S)"
  mv -- "$BMAX_LOG" "$BMAX_BACKUP"
fi
mv -- "$BMAX_CANDIDATE" "$BMAX_LOG"
```

### Parse results and compute PEAK_GPU_STREAMS
```bash
QPS_BS1=$(grep -oP 'Throughput:\s*\K[0-9.]+' \
  models/$MODEL_NAME/benchmarks/b1/trtexec_b1.log | tail -1)
GPU_MEAN_BS1=$(grep -oP 'GPU Compute Time:.*mean = \K[0-9.]+' \
  models/$MODEL_NAME/benchmarks/b1/trtexec_b1.log | tail -1)

QPS_BS_MAX=$(grep -oP 'Throughput:\s*\K[0-9.]+' \
  models/$MODEL_NAME/benchmarks/b${MAX_BS}/trtexec_b${MAX_BS}.log | tail -1)
GPU_MEAN_BS_MAX=$(grep -oP 'GPU Compute Time:.*mean = \K[0-9.]+' \
  models/$MODEL_NAME/benchmarks/b${MAX_BS}/trtexec_b${MAX_BS}.log | tail -1)
GPU_P99_BS_MAX=$(grep -oP 'GPU Compute Time:.*percentile\(99%\) = \K[0-9.]+' \
  models/$MODEL_NAME/benchmarks/b${MAX_BS}/trtexec_b${MAX_BS}.log | tail -1)

read IMGS_PER_SEC PEAK_GPU_STREAMS < <(python3 -c "
import math
imgs = float('$QPS_BS_MAX') * $MAX_BS
streams = int(math.floor(imgs / 30))
print(round(imgs, 2), streams)
")

echo "BS=1:       QPS=$QPS_BS1  GPU mean=${GPU_MEAN_BS1}ms"
echo "BS=$MAX_BS: QPS=$QPS_BS_MAX  imgs/s=$IMGS_PER_SEC  GPU mean=${GPU_MEAN_BS_MAX}ms  P99=${GPU_P99_BS_MAX}ms"
echo "PEAK_GPU_STREAMS=$PEAK_GPU_STREAMS  (floor($IMGS_PER_SEC / 30))"

STEP5_END=$(date +%s.%N)
STEP5_DURATION=$(python3 -c "print(round($STEP5_END - $STEP5_START, 2))")   # bc is not in the container; python3 always is
echo "[Step 5] Benchmarks completed in ${STEP5_DURATION}s"
```

`PEAK_GPU_STREAMS` is the GPU-only upper bound on real-time 30fps stream count. DeepStream will always achieve fewer streams due to NVDEC, mux, and GStreamer overhead (typically 10–40%). Use `PEAK_GPU_STREAMS` as the starting stream count for DS Run 1 (calibration).

### Iterative Engine Scaling (PEAK_GPU_STREAMS > MAX_BS)

If `PEAK_GPU_STREAMS > MAX_BS`, the engine's max batch size is the bottleneck.
Double and remeasure, but cap the campaign at three rebuilds and batch 512. A
successful validated build at a cap remains usable. If the next required build
would exceed either cap, record `BLOCKED: engine scaling limit reached` and
stop; do not build an unbounded sequence of engines.

**Why doubling, not jumping to PEAK directly**: Jumping from 64→512 based on an extrapolated projection wastes GPU memory if the projection was off. Doubling (64→128→256→512) makes incremental, verifiable steps — each trtexec run gives real throughput data before committing to a larger rebuild.

```bash
MAX_ENGINE_REBUILDS="${MAX_ENGINE_REBUILDS:-3}"
MAX_ENGINE_BATCH="${MAX_ENGINE_BATCH:-512}"
case "$MAX_ENGINE_REBUILDS" in
  ""|*[!0-9]*) echo "ERROR: MAX_ENGINE_REBUILDS must be a positive integer" >&2; exit 2 ;;
esac
case "$MAX_ENGINE_BATCH" in
  ""|*[!0-9]*) echo "ERROR: MAX_ENGINE_BATCH must be a positive integer" >&2; exit 2 ;;
esac
[ "$MAX_ENGINE_REBUILDS" -gt 0 ] || {
  echo "ERROR: MAX_ENGINE_REBUILDS must be positive" >&2
  exit 2
}
[ "$MAX_ENGINE_BATCH" -gt 0 ] || {
  echo "ERROR: MAX_ENGINE_BATCH must be positive" >&2
  exit 2
}
ENGINE_REBUILDS=0
while [ "$PEAK_GPU_STREAMS" -gt "$MAX_BS" ]; do
  NEW_MAX_BS=$(python3 -c "print($MAX_BS * 2)")  # STRICT DOUBLING — do not change to ceil(log2(PEAK))
  echo "Rebuilding engine: PEAK_GPU_STREAMS=$PEAK_GPU_STREAMS > MAX_BS=$MAX_BS — doubling to: $NEW_MAX_BS"
  if [ "$ENGINE_REBUILDS" -ge "$MAX_ENGINE_REBUILDS" ] || \
     [ "$NEW_MAX_BS" -gt "$MAX_ENGINE_BATCH" ]; then
    echo "BLOCKED: engine scaling limit reached (rebuilds=$ENGINE_REBUILDS, next_batch=$NEW_MAX_BS)" >&2
    echo "Successful engines and diagnostic logs are retained; no over-cap target is created." >&2
    exit 4
  fi
  ENGINE_REBUILDS=$((ENGINE_REBUILDS + 1))


  mkdir -p models/$MODEL_NAME/benchmarks/b${NEW_MAX_BS}
  NEW_ENGINE="models/$MODEL_NAME/benchmarks/engines/${MODEL_FILENAME}_dynamic_b${NEW_MAX_BS}.engine"
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  NEW_ENGINE_CANDIDATE="${NEW_ENGINE}.partial.${TIMESTAMP}.$$"
  NEW_BUILD_LOG="models/$MODEL_NAME/benchmarks/engines/${MODEL_FILENAME}_dynamic_build_b${NEW_MAX_BS}_${TIMESTAMP}.log"
  [ ! -e "$NEW_ENGINE_CANDIDATE" ] || { echo "ERROR: candidate engine path already exists: $NEW_ENGINE_CANDIDATE" >&2; exit 2; }
  if $TRTEXEC \
    --onnx="$ONNX_PATH" \
    --minShapes=$INPUT_NAME:1x3x${H}x${W} \
    --optShapes=$INPUT_NAME:${NEW_MAX_BS}x3x${H}x${W} \
    --maxShapes=$INPUT_NAME:${NEW_MAX_BS}x3x${H}x${W} \
    --fp16 --skipInference \
    --memPoolSize=workspace:32768M \
    --timingCacheFile=models/$MODEL_NAME/benchmarks/engines/timing.cache \
    --saveEngine="$NEW_ENGINE_CANDIDATE" 2>&1 | tee "$NEW_BUILD_LOG"; then
    BUILD_STATUS=("${PIPESTATUS[@]}")
  else
    BUILD_STATUS=("${PIPESTATUS[@]}")
  fi
  if [ "${BUILD_STATUS[0]:-1}" -ne 0 ] || [ "${BUILD_STATUS[1]:-1}" -ne 0 ] || [ ! -s "$NEW_ENGINE_CANDIDATE" ]; then
    echo "ERROR: TensorRT engine b${NEW_MAX_BS} failed — preserved $NEW_BUILD_LOG and $NEW_ENGINE_CANDIDATE" >&2
    exit 1
  fi
  if [ -e "$NEW_ENGINE" ]; then
    NEW_ENGINE_BACKUP="${NEW_ENGINE}.previous.${TIMESTAMP}"
    [ ! -e "$NEW_ENGINE_BACKUP" ] || { echo "ERROR: backup path already exists: $NEW_ENGINE_BACKUP" >&2; exit 2; }
    mv -- "$NEW_ENGINE" "$NEW_ENGINE_BACKUP"
  fi
  mv -- "$NEW_ENGINE_CANDIDATE" "$NEW_ENGINE"

  # Update ENGINE and MAX_BS — re-run trtexec at new BS and recompute PEAK_GPU_STREAMS
  ENGINE="models/$MODEL_NAME/benchmarks/engines/${MODEL_FILENAME}_dynamic_b${NEW_MAX_BS}.engine"
  MAX_BS=$NEW_MAX_BS

  BMAX_LOG="models/$MODEL_NAME/benchmarks/b${MAX_BS}/trtexec_b${MAX_BS}.log"
  BMAX_CANDIDATE="${BMAX_LOG}.partial.$(date +%Y%m%d_%H%M%S).$$"
  $TRTEXEC \
    --loadEngine="$ENGINE" \
    --shapes=$INPUT_NAME:${MAX_BS}x3x${H}x${W} \
    --noDataTransfers --duration=10 --warmUp=1000 \
    2>&1 | tee "$BMAX_CANDIDATE"
  BENCH_STATUS=("${PIPESTATUS[@]}")
  if [ "${BENCH_STATUS[0]:-1}" -ne 0 ] || [ "${BENCH_STATUS[1]:-1}" -ne 0 ]; then
    BMAX_FAILED="${BMAX_LOG}.failed.$(date +%Y%m%d_%H%M%S)"
    mv -- "$BMAX_CANDIDATE" "$BMAX_FAILED"
    echo "ERROR: iterative trtexec benchmark failed at BS=$MAX_BS; preserved $BMAX_FAILED" >&2
    exit 1
  fi
  if [ -e "$BMAX_LOG" ]; then
    BMAX_BACKUP="${BMAX_LOG}.previous.$(date +%Y%m%d_%H%M%S)"
    mv -- "$BMAX_LOG" "$BMAX_BACKUP"
  fi
  mv -- "$BMAX_CANDIDATE" "$BMAX_LOG"

  QPS_BS_MAX=$(grep -oP 'Throughput:\s*\K[0-9.]+' \
    models/$MODEL_NAME/benchmarks/b${MAX_BS}/trtexec_b${MAX_BS}.log | tail -1)
  GPU_MEAN_BS_MAX=$(grep -oP 'GPU Compute Time:.*mean = \K[0-9.]+' \
    models/$MODEL_NAME/benchmarks/b${MAX_BS}/trtexec_b${MAX_BS}.log | tail -1)
  GPU_P99_BS_MAX=$(grep -oP 'GPU Compute Time:.*percentile\(99%\) = \K[0-9.]+' \
    models/$MODEL_NAME/benchmarks/b${MAX_BS}/trtexec_b${MAX_BS}.log | tail -1)
  read IMGS_PER_SEC PEAK_GPU_STREAMS < <(python3 -c "
import math
imgs = float('$QPS_BS_MAX') * $MAX_BS
print(round(imgs, 2), int(math.floor(imgs / 30)))
")
  echo "Recomputed: BS=$MAX_BS  imgs/s=$IMGS_PER_SEC  PEAK_GPU_STREAMS=$PEAK_GPU_STREAMS"
done

echo "PEAK_GPU_STREAMS ($PEAK_GPU_STREAMS) <= MAX_BS ($MAX_BS) — engine scaling complete."
```

**Engine count summary:**

| Scenario | Example | Engines | trtexec runs |
|----------|---------|---------|-------------|
| PEAK_GPU_STREAMS ≤ 64 (transformer/large models) | RT-DETR, OWL-ViT | **1** (`b64`) | **2** |
| PEAK_GPU_STREAMS > 64, ≤ 128 (mid models) | TrafficCamNet | **2** (`b64` + `b128`) | **3** |
| PEAK_GPU_STREAMS > 128, ≤ 256 (fast models) | YOLO26n | **3** (`b64`+`b128`+`b256`) | **4** |
| PEAK_GPU_STREAMS > 256 (very fast nano models) | — | **≤4** (through `b512`; then `BLOCKED`) | **≤5** |

## trtexec Flags Reference

### Recommended Flags
| Flag | Purpose | When to use |
|------|---------|-------------|
| `--duration=10` | Longer run for stable numbers | All benchmark runs (5a, 5b) |
| `--warmUp=1000` | 1s warmup before measurement | All benchmark runs (5a, 5b) |
| `--noDataTransfers` | GPU-only compute (matches DS reality) | Always |

### Why GPU-only (`--noDataTransfers`) Only
In DeepStream, frames are decoded on GPU (`nvv4l2decoder`) and stay on GPU through `nvinfer` — no H2D transfer. Standard trtexec transfers synthetic data from host, which is not representative. Do NOT report H2D/D2H latency.

### Flags That Do NOT Help (tested)
| Flag | Result | Why |
|------|--------|-----|
| `--best` | No improvement | Engine already built with --fp16, runtime flag doesn't change precision |
| `--exposeDMA` | **45% WORSE** throughput | Serializes DMA transfers — kills pipelining |
| `--infStreams=4` | +2% QPS max | GPU already saturated |

### Key Metrics to Report from trtexec
- **Throughput (QPS)** and **Images/s** (QPS × batch_size)
- **GPU Compute mean (ms)** and **GPU Compute P99 (ms)**
- **GPU Compute per image (ms)** (GPU Compute mean / batch_size)
- Do **NOT** report: H2D latency, D2H latency, Host Latency, transfer overhead

## Engine Version Compatibility -- CRITICAL

TensorRT engine files are **not portable** across TensorRT versions.

### Pre-flight Version Check
`trtexec` lives in the container at `/usr/src/tensorrt/bin/trtexec` (set `TRTEXEC=/usr/src/tensorrt/bin/trtexec`);
get its version with `"$TRTEXEC" --version`. No host `dpkg`/`libnvinfer-bin` check is needed.

### Build and run in the SAME container image
The engine is **built and run inside the same image** (`nvcr.io/nvidia/deepstream:9.1-triton-multiarch`),
so the `libnvinfer` that `trtexec` builds against is byte-for-byte the one DeepStream loads at runtime —
**there is no build-vs-runtime version skew.** This is the correct fix for TensorRT's engine
non-portability: it *eliminates* the "Docker-built engine silently fails on a differently-versioned
host runtime" failure mode (0% GPU / stuck pipeline) that the old "always build on the host" rule tried
to avoid. **Never build an engine against a different TensorRT than the one that will run it** — using
one image for both guarantees this by construction. (Do not build engines on the host; the host has no
required toolchain in the container-run model.)

## Known Issues and Workarounds

### `--memPoolSize` Flag Format — `M` vs `MiB` (CRITICAL silent failure)

- **Correct**: `--memPoolSize=workspace:32768M` (suffix `M` = Mebibytes)
- **WRONG**: `--memPoolSize=workspace:32768MiB` — trtexec interprets `MiB` as bytes, so `32768MiB` becomes 32 KB. All tactics fail with "insufficient workspace". There is no parse warning; the only symptom is `Memory Pools: workspace: 0.03125 MiB` in the build log.
- Valid suffixes: `B`, `K`, `M`, `G`, or no suffix (default MiB).

### Deformable Attention Models (RT-DETR, DDETR, Deformable DETR)

Models using `MultiscaleDeformableAttnPlugin_TRT` build correctly on TRT 10.16 **provided workspace is sufficient**.
- **Required**: `--memPoolSize=workspace:32768M` (not the default 8GB) — deformable attention at BS=64 needs substantial workspace for ForeignNode fusion tactics.
- `--builderOptimizationLevel=4` (default) works; do not lower it unless necessary.
- Typical footprint at BS=64 on H100: activation ~4266 MiB, peak memory ~7809 MiB, build time ~825s. The compiler backend phase after engine generation can take 5-10 minutes with no log output — this is normal, not a hang.
- Error "Could not find any implementation for node {ForeignNode[...]} due to insufficient workspace" is a genuine signal to raise the workspace.

### DETR / DETR-family Backbone Mask ForeignNode Failure (TRT 10.16)

HF-exported DETR/DDETR models contain a dynamic backbone mask path (`Cast → Resize → Sigmoid`) that TRT 10.16 fuses into a ForeignNode with no valid tactic: `"Could not find any implementation for node {ForeignNode[.../Cast_2.../Sigmoid]}"`.

- **Preferred fix** (TRT 10.16.01+, PyTorch 2.11+, transformers 5.5+): use the dynamo export path with `torch.export.Dim("batch", min=1, max=N)` in `dynamic_shapes`. The dynamo exporter produces a different graph that does NOT trigger the ForeignNode failure. TRT converts it directly as a dynamic-batch engine.
- **Fallback for older toolchains**: run `onnxsim.simplify(model, input_shapes={'pixel_values': [BS, 3, H, W]})` first. This folds the mask into constants but bakes batch size, requiring per-batch ONNX + engine files.
- **Secondary workaround**: lower `builder_optimization_level` to 2 via the Python TRT API (`config.builder_optimization_level = 2`). Prevents over-aggressive fusion; engines built this way are still compatible with `trtexec --loadEngine`.

### Dynamic Engine Batch-Size Anomalies (transformer models)

Dynamic-shape engines for transformer models (DETR, RT-DETR) can show **non-monotonic throughput** — specific non-power-of-2 batch sizes (e.g., BS=17-19) perform dramatically worse than neighboring values. Cause: TRT tactic selection for attention layers at non-optimal shapes. When DS at `N` streams shows surprisingly low FPS, test `N±8` before concluding the GPU is saturated. Prefer power-of-2 batch sizes for production.

## Output Summary

```bash
TOTAL_DURATION=$(python3 -c "print(round($STEP4_DURATION + $STEP5_DURATION, 2))")   # bc is not in the container; python3 always is
```

When complete, print:
```
=== TRT Engine Build Complete ===
Model:   $MODEL_NAME
Engine:  models/$MODEL_NAME/benchmarks/engines/${MODEL_FILENAME}_dynamic_b${MAX_BS}.engine
         (single engine — used for trtexec baseline and all DS runs)

trtexec Results:
  BS=1:       $QPS_BS1 QPS  |  GPU mean: ${GPU_MEAN_BS1}ms
  BS=$MAX_BS: $QPS_BS_MAX QPS  |  $IMGS_PER_SEC img/s  |  GPU mean: ${GPU_MEAN_BS_MAX}ms  P99: ${GPU_P99_BS_MAX}ms

PEAK_GPU_STREAMS (GPU-only upper bound): $PEAK_GPU_STREAMS streams @30fps

Timing:
  Step 4 (engine build): ${STEP4_DURATION}s
  Step 5 (benchmarks):   ${STEP5_DURATION}s
  Total Steps 4-5:       ${TOTAL_DURATION}s

Ready for: Steps 6-7 — read references/pipeline-run.md models/$MODEL_NAME/
```
