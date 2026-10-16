#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Preflight for deepstream-import-vision-model: verify the environment BEFORE running the
# import loop. Everything runs THROUGH the DeepStream container (no host packages). Checks:
#   1. docker daemon reachable            (host-mode only)
#   2. the DeepStream image is pulled     (host-mode only)
#   3. GPU visible inside the container
#   4. the venv, trtexec, visual encoder, and sample video are present
#
# Usage inside the named campaign container:
#   docker exec "$DS_CONTAINER" \
#     .claude/skills/deepstream-import-vision-model/scripts/preflight.sh
# Host-mode diagnostics require DS_IMAGE=<repository@sha256:digest>.
set -u
IMG="${1:-${DS_IMAGE:-}}"
DS_GPU_DEVICE="${DS_GPU_DEVICE:-0}"
VENV="${2:-build/.venv_optimum}"
PREFLIGHT_PROFILE="${DS_PREFLIGHT_PROFILE:-full}"
case "$PREFLIGHT_PROFILE" in
  full|native-pipeline) ;;
  *) echo "[preflight] invalid DS_PREFLIGHT_PROFILE: use full or native-pipeline" >&2; exit 2 ;;
esac
SAMPLE_VIDEO="${DS_VIDEO:-/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4}"
PKGS="torch torchvision transformers onnx onnxruntime onnxscript huggingface_hub matplotlib numpy markdown reportlab PIL gi pyds"
RUNNER_IMPORTS='import gi; gi.require_version("Gst", "1.0"); gi.require_version("GstPbutils", "1.0"); from gi.repository import Gst, GstPbutils; import pyds; from PIL import Image'
CUDA_PROBE='.claude/skills/deepstream-import-vision-model/scripts/check-cuda-context.py'
check_encoder() {
  command -v gst-inspect-1.0 >/dev/null 2>&1 &&
    gst-inspect-1.0 theoraenc >/dev/null 2>&1 &&
    gst-inspect-1.0 oggmux >/dev/null 2>&1
}
fail=0
ok(){  echo "  [OK]      $1"; }
warn(){ echo "  [WARN]    $1"; }
bad(){  echo "  [MISSING] $1"; fail=1; }

# Container-mode: when run INSIDE the container (Windows/macOS invoke it as
#   docker run --gpus device=0 -v <pwd>:/work -w /work <image> bash scripts/preflight.sh
# ), the host docker/image checks are moot and there's no nested docker CLI — verify GPU +
# venv + trtexec DIRECTLY. On a Linux host (no /.dockerenv) fall through to host-orchestration.
if [ -f /.dockerenv ]; then
  echo "[preflight] container-mode (inside the container — verifying GPU + CUDA + venv + trtexec directly)"
  echo "[preflight] 1/7 GPU (nvidia-smi)"
  nvidia-smi -L >/dev/null 2>&1 && ok "GPU visible in container" \
    || bad "no GPU in container — select one device with --gpus device=0 (Windows: Docker Desktop WSL2 backend + NVIDIA driver)"
  echo "[preflight] 2/7 trtexec"
  [ -x /usr/src/tensorrt/bin/trtexec ] && ok "trtexec present (/usr/src/tensorrt/bin/trtexec)" \
    || bad "trtexec not found — is this the DeepStream/TensorRT image?"
  echo "[preflight] 3/7 CUDA device context"
  cuda_result=$(python3 "$CUDA_PROBE" 2>&1)
  if [ $? -eq 0 ]; then ok "$cuda_result"
  else bad "$cuda_result (update the host driver or select a compatible pinned image before inference)"; fi
  echo "[preflight] 4/7 python packages and runner bindings"
  if [ "$PREFLIGHT_PROFILE" = native-pipeline ]; then
    warn "Python/GI/PyDS checks deferred for native C++ pipeline only; full preflight required before promotion/reporting"
  else
    if [ -x "$VENV/bin/python" ]; then
      miss=$("$VENV/bin/python" -c "import importlib.util as u;print(' '.join(p for p in '$PKGS'.split() if u.find_spec(p) is None))" 2>/dev/null)
      if [ -z "${miss// /}" ]; then ok "all packages present"
      else bad "missing in venv: $miss  (run setup.sh to (re)build the venv)"; fi
    else
      bad "venv not found: $VENV/bin/python — run setup.sh first (creates build/.venv_optimum)"
    fi
    if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "$RUNNER_IMPORTS" >/dev/null 2>&1; then
      ok "GI, Gst, GstPbutils, PyDS, and Pillow import from the shared skill venv"
    else
      bad "runner bindings unavailable in $VENV (rerun setup.sh; it installs/bridges PyDS)"
    fi
  fi
  echo "[preflight] 5/7 visual encoder"
  check_encoder && ok "theoraenc+oggmux available for all shipped visual runners" \
    || bad "theoraenc+oggmux required by shipped cascade/map runners (rerun setup.sh)"
  echo "[preflight] 6/7 PDF renderer"
  if [ "$PREFLIGHT_PROFILE" = native-pipeline ]; then
    warn "PDF renderer deferred; a native pipeline PASS is not a completed Step 8 report"
  else
    command -v wkhtmltopdf >/dev/null 2>&1 && ok "wkhtmltopdf available" \
      || bad "wkhtmltopdf missing (run setup.sh in this execution container)"
  fi
  echo "[preflight] 7/7 selected input video"
  [ -f "$SAMPLE_VIDEO" ] && ok "selected input video present: $SAMPLE_VIDEO" \
    || bad "selected input video missing: $SAMPLE_VIDEO (provide an explicit DS_VIDEO)"
  if [ "$fail" -eq 0 ] && [ "$PREFLIGHT_PROFILE" = native-pipeline ]; then
    echo "[preflight] RESULT: PASS — native pipeline prerequisites only; full campaign checks deferred"
  else
    echo "[preflight] RESULT: $([ $fail -eq 0 ] && echo 'PASS — full environment ready' || echo 'FAIL — fix [MISSING] items before running')"
  fi
  exit $fail
fi

if [ -z "$IMG" ]; then
  bad "set DS_IMAGE to the verified DeepStream repository @sha256 digest"
  exit 1
elif ! printf '%s\n' "$IMG" | grep -Eq '^nvcr\.io/nvidia/deepstream@sha256:[0-9a-f]{64}$'; then
  bad "DS_IMAGE must be nvcr.io/nvidia/deepstream@sha256:<64-hex-digest>"
  exit 1
fi
if [[ ! "$DS_GPU_DEVICE" =~ ^([0-9]+|GPU-[0-9a-fA-F-]{36}|MIG-GPU-[0-9a-fA-F-]{36}/[0-9]+/[0-9]+)$ ]]; then
  bad "DS_GPU_DEVICE must select exactly one GPU index, GPU UUID, or MIG device"
  exit 1
fi

  echo "[preflight] 1/8 docker daemon"
docker info >/dev/null 2>&1 && ok "docker reachable" || bad "docker not installed or daemon not running"

echo "[preflight] 2/8 DeepStream image: $IMG"
have_img=0
if docker image inspect "$IMG" >/dev/null 2>&1; then ok "image present"; have_img=1
else bad "image unavailable — authenticate if required, then run: docker pull $IMG"; fi

if [ "$have_img" = 1 ]; then
  echo "[preflight] 3/8 selected GPU via docker --gpus device=$DS_GPU_DEVICE"
  docker run --rm --gpus "device=$DS_GPU_DEVICE" --entrypoint nvidia-smi "$IMG" -L >/dev/null 2>&1 \
    && ok "GPU visible in container" \
    || bad "selected GPU unavailable (check driver + nvidia-container-toolkit)"

  echo "[preflight] 4/8 CUDA device context"
  cuda_result=$(docker run --rm --gpus "device=$DS_GPU_DEVICE" --entrypoint python3 \
    -v "$PWD":/work -w /work "$IMG" "$CUDA_PROBE" 2>&1)
  if [ $? -eq 0 ]; then ok "$cuda_result"
  else bad "$cuda_result (update the host driver or select a compatible pinned image before inference)"; fi
  echo "[preflight] 5/8 python packages and runner bindings"
  if [ "$PREFLIGHT_PROFILE" = native-pipeline ]; then
    warn "Python/GI/PyDS checks deferred for native C++ pipeline only; full preflight required before promotion/reporting"
  else
    if [ -x "$VENV/bin/python" ]; then
      miss=$(docker run --rm --entrypoint /work/"$VENV"/bin/python -v "$PWD":/work "$IMG" \
        -c "import importlib.util as u;print(' '.join(p for p in '$PKGS'.split() if u.find_spec(p) is None))" 2>/dev/null)
      if [ -z "${miss// /}" ]; then ok "all packages present"
      else bad "missing in venv: $miss  (run setup.sh)"; fi
    else
      bad "venv not found: $VENV/bin/python — run setup.sh first"
    fi
    docker run --rm --entrypoint /work/"$VENV"/bin/python -v "$PWD":/work "$IMG" -c "$RUNNER_IMPORTS" >/dev/null 2>&1 \
      && ok "GI, Gst, GstPbutils, PyDS, and Pillow import from the shared skill venv" \
      || bad "runner bindings unavailable in $VENV (run setup.sh in this mounted container)"
  fi
  echo "[preflight] 6/8 visual encoder"
  docker run --rm --entrypoint bash "$IMG" -lc \
    'command -v gst-inspect-1.0 >/dev/null 2>&1 && gst-inspect-1.0 theoraenc >/dev/null 2>&1 && gst-inspect-1.0 oggmux >/dev/null 2>&1' \
    && ok "theoraenc+oggmux available for all shipped visual runners" \
    || bad "theoraenc+oggmux required by shipped cascade/map runners (run setup.sh)"
  echo "[preflight] 7/8 PDF renderer"
  if [ "$PREFLIGHT_PROFILE" = native-pipeline ]; then
    warn "PDF renderer deferred; a native pipeline PASS is not a completed Step 8 report"
  else
    docker run --rm --entrypoint sh "$IMG" -c 'command -v wkhtmltopdf >/dev/null 2>&1' \
      && ok "wkhtmltopdf available in image" \
      || bad "wkhtmltopdf missing in image (run setup and validation in one persistent container)"
  fi
  echo "[preflight] 8/8 selected input video"
  docker run --rm -v "$PWD":/work -w /work --entrypoint test "$IMG" -f "$SAMPLE_VIDEO" \
    && ok "selected input video present: $SAMPLE_VIDEO" \
    || bad "selected input video missing: $SAMPLE_VIDEO (provide an explicit DS_VIDEO)"
else
  echo "[preflight] 3/8 GPU            — SKIPPED (image unavailable)"
  echo "[preflight] 4/8 CUDA context   — SKIPPED (image unavailable)"
  echo "[preflight] 5/8 python packages — SKIPPED (image unavailable)"
  echo "[preflight] 6/8 visual encoder  — SKIPPED (image unavailable)"
  echo "[preflight] 7/8 PDF renderer    — SKIPPED (image unavailable)"
  echo "[preflight] 8/8 sample video    — SKIPPED (image unavailable)"
fi

if [ "$fail" -eq 0 ] && [ "$PREFLIGHT_PROFILE" = native-pipeline ]; then
  echo "[preflight] RESULT: PASS — native pipeline prerequisites only; full campaign checks deferred"
else
  echo "[preflight] RESULT: $([ $fail -eq 0 ] && echo 'PASS — full environment ready' || echo 'FAIL — fix [MISSING] items before running')"
fi
exit $fail
