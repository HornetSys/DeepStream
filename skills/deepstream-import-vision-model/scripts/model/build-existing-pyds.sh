#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build PyDS from the source checkout already shipped in some DeepStream images.
# The image's installer always clones into this path, so it cannot be used when
# the checkout already exists. Never remove or replace that checkout.
set -euo pipefail

DS_ROOT="/opt/nvidia/deepstream/deepstream"
PYDS_SOURCE="$DS_ROOT/sources/deepstream_python_apps"
PYDS_VENV="$PYDS_SOURCE/pyds"
BINDINGS="$PYDS_SOURCE/bindings"

if [ ! -d "$PYDS_SOURCE/.git" ] || [ ! -d "$BINDINGS" ]; then
  echo "[pyds] ERROR: existing DeepStream Python apps checkout is incomplete: $PYDS_SOURCE" >&2
  exit 1
fi

echo "[pyds] building bindings from the existing checkout: $PYDS_SOURCE"
if [ "$(id -u)" = "0" ]; then APT=""; else APT="sudo"; fi
$APT apt-get update -qq
$APT apt-get install -y -qq python3-gi python3-dev python3-gst-1.0 python-gi-dev \
  git meson python3-venv cmake g++ build-essential libglib2.0-dev \
  libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake \
  libgirepository-2.0-dev libcairo2-dev apt-transport-https ca-certificates

if [ ! -x "$PYDS_VENV/bin/python" ]; then
  python3 -m venv "$PYDS_VENV"
fi
"$PYDS_VENV/bin/python" -m pip install build PyGObject cuda-python

cd "$BINDINGS"
git submodule update --init
"$PYDS_VENV/bin/python" 3rdparty/git-partial-submodule/git-partial-submodule.py restore-sparse
GST_PYTHON="$BINDINGS/3rdparty/gstreamer/subprojects/gst-python"
if [ -d "$GST_PYTHON/build/meson-private" ]; then
  meson setup --reconfigure "$GST_PYTHON/build"
else
  meson setup "$GST_PYTHON/build" "$GST_PYTHON"
fi
ninja -C "$GST_PYTHON/build"
$APT ninja -C "$GST_PYTHON/build" install
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
"$PYDS_VENV/bin/python" -m pip install "$BINDINGS"
"$PYDS_VENV/bin/python" -c 'import pyds'
