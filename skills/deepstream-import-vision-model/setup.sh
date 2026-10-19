#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved. Apache-2.0.
#
# One-command environment bootstrap for deepstream-import-vision-model on a FRESH machine.
# Creates the shared venv (build/.venv_optimum), installs the ONNX-export + report Python deps,
# and installs report/visual dependencies — ALL INSIDE the container. Idempotent.
#
# Nothing runs on the host: the host only needs Docker + the NVIDIA driver. Run this INSIDE the
# DeepStream container, from the working root (where models/, reports/, build/ will live) — so
# torch/CUDA/TensorRT and the compiled parser all match the runtime:
#
# Start the digest-pinned persistent campaign container exactly as documented
# in SKILL.md, then run setup.sh, preflight.sh, and every phase in that container.
set -euo pipefail

SK=".claude/skills/deepstream-import-vision-model"
VENV="build/.venv_optimum"
PY="$VENV/bin/python"
DS_ROOT="/opt/nvidia/deepstream/deepstream"
PYDS_INSTALLER="$DS_ROOT/user_deepstream_python_apps_install.sh"
PYDS_SOURCE="$DS_ROOT/sources/deepstream_python_apps"
PYDS_VENV="$DS_ROOT/sources/deepstream_python_apps/pyds"

if [ ! -d "$SK" ]; then
  echo "[setup] ERROR: run from the working root (the dir that contains $SK)." >&2
  exit 1
fi

# 1) virtualenv (the container python lacks ensurepip, so bootstrap virtualenv via pip)
if [ ! -x "$PY" ]; then
  echo "[setup] creating venv at $VENV"
  python3 -m pip install --quiet --user virtualenv 2>/dev/null || python3 -m pip install --quiet virtualenv
  python3 -m virtualenv --system-site-packages "$VENV"
else
  echo "[setup] venv exists: $VENV"
  if ! grep -qx "include-system-site-packages = true" "$VENV/pyvenv.cfg"; then
    echo "[setup] enabling system site packages for DeepStream GI bindings"
    sed -i 's/^include-system-site-packages = false$/include-system-site-packages = true/' "$VENV/pyvenv.cfg"
  fi
fi

# 2) Python dependencies (ONNX export + report)
echo "[setup] installing Python deps from $SK/scripts/requirements.txt (this can take several minutes)"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install -r "$SK/scripts/requirements.txt"

# 3) Report and visual dependencies — installed IN the container.
APT_PACKAGES=()
command -v wkhtmltopdf >/dev/null 2>&1 || APT_PACKAGES+=(wkhtmltopdf)
if ! command -v gst-inspect-1.0 >/dev/null 2>&1 || \
   ! gst-inspect-1.0 theoraenc >/dev/null 2>&1 || \
   ! gst-inspect-1.0 oggmux >/dev/null 2>&1; then
  APT_PACKAGES+=(gstreamer1.0-plugins-base)
fi
if ! python3 -c 'import gi; gi.require_version("Gst", "1.0"); gi.require_version("GstPbutils", "1.0"); from gi.repository import Gst, GstPbutils' >/dev/null 2>&1; then
  APT_PACKAGES+=(python3-gi python3-dev python3-gst-1.0 gir1.2-gst-plugins-base-1.0)
fi
if [ ! -f /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf ]; then
  APT_PACKAGES+=(fontconfig fonts-dejavu-core)
fi
if [ "${#APT_PACKAGES[@]}" -gt 0 ]; then
  echo "[setup] installing in-container packages: ${APT_PACKAGES[*]}"
  if [ "$(id -u)" = "0" ]; then APT=""; else APT="sudo"; fi
  $APT apt-get update -qq && $APT apt-get install -y -qq "${APT_PACKAGES[@]}" \
    || echo "[setup] WARN: dependency install failed — preflight will report unavailable gates"
else
  echo "[setup] report and visual dependencies already present"
fi

# 4) Bind the reference runner to this shared venv. The DeepStream Triton
# image's optional installer creates its own pyds venv; the .pth bridge below
# deliberately exposes only its installed site-packages to the skill venv.
if ! "$PY" -c 'import gi; gi.require_version("Gst", "1.0"); from gi.repository import Gst; import pyds; from PIL import Image' >/dev/null 2>&1; then
  if [ -x "$PYDS_VENV/bin/python" ] &&
    "$PYDS_VENV/bin/python" -c 'import pyds' >/dev/null 2>&1; then
    echo "[setup] reusing PyDS from the image-provided venv"
  elif [ -e "$PYDS_SOURCE" ]; then
    echo "[setup] reusing the existing DeepStream Python apps checkout"
    bash "$SK/scripts/model/build-existing-pyds.sh"
  else
    if [ ! -x "$PYDS_INSTALLER" ]; then
      echo "[setup] ERROR: PyDS is unavailable and DeepStream's binding installer is absent: $PYDS_INSTALLER" >&2
      exit 1
    fi
    echo "[setup] installing DeepStream Python bindings with the image-provided installer"
    bash "$PYDS_INSTALLER" --build-bindings
  fi
  if [ ! -x "$PYDS_VENV/bin/python" ]; then
    echo "[setup] ERROR: PyDS installation did not create $PYDS_VENV/bin/python" >&2
    exit 1
  fi
  "$PYDS_VENV/bin/python" -c 'import pyds' || {
    echo "[setup] ERROR: PyDS remains unavailable in $PYDS_VENV" >&2
    exit 1
  }
  PYDS_SITE=$("$PYDS_VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  VENV_SITE=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
  case "$PYDS_SITE" in
    "$DS_ROOT"/*) ;;
    *) echo "[setup] ERROR: refusing an unexpected PyDS site-packages path: $PYDS_SITE" >&2; exit 1 ;;
  esac
  printf '%s\n' "$PYDS_SITE" > "$VENV_SITE/deepstream-pyds.pth"
fi

# 5) verify
echo "[setup] verifying..."
"$PY" - <<'PYV'
import importlib.util, sys
mods = ["torch","torchvision","transformers","onnx","onnxruntime","onnxscript",
        "huggingface_hub","matplotlib","numpy","markdown","reportlab","PIL","gi","pyds"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
import torch
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstPbutils", "1.0")
    from gi.repository import Gst, GstPbutils
    import pyds
    from PIL import Image
except Exception as exc:
    print(f"  MISSING: DeepStream runner bindings: {exc}")
    sys.exit(1)
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if missing: print("  MISSING:", missing); sys.exit(1)
print("  all required packages and DeepStream Python bindings present")
PYV
if command -v wkhtmltopdf >/dev/null 2>&1; then
  echo "  wkhtmltopdf: $(command -v wkhtmltopdf)"
else
  echo "[setup] ERROR: wkhtmltopdf remains unavailable; PDF reporting is BLOCKED" >&2
  exit 1
fi
command -v /usr/src/tensorrt/bin/trtexec >/dev/null 2>&1 && echo "  trtexec: /usr/src/tensorrt/bin/trtexec" || echo "  trtexec: (check TensorRT in image)"

echo
if command -v gst-inspect-1.0 >/dev/null 2>&1 && \
   gst-inspect-1.0 theoraenc >/dev/null 2>&1 && gst-inspect-1.0 oggmux >/dev/null 2>&1; then
  echo "  visual fallback: theoraenc + oggmux"
else echo "  visual fallback: absent — visual validation will remain BLOCKED without NVENC"; fi
echo "[setup] DONE. Keep this container alive; run preflight and every phase here:"
echo "  $SK/scripts/preflight.sh"
echo "  (see $SK/SKILL.md + $SK/references/windows.md for the per-shell mount token)"
