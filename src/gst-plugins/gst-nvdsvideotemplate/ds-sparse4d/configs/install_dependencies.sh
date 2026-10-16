#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sparse4D build dependencies: OpenCV, nlohmann-json, Eigen, and PyTorch 2.9.0
# (cu130) under /usr/local/lib/python3/dist-packages so the Makefile's
# TORCH_PATH default resolves. Invoked from scripts/install_opensource_deps.sh.

set -e
export DEBIAN_FRONTEND=noninteractive

echo "Installing Sparse4D apt dependencies"
apt-get update
apt-get install -y libopencv-dev libeigen3-dev nlohmann-json3-dev python3-pip
dpkg --configure -a

# Makefile requires: /usr/local/lib/python3/dist-packages/torch
TORCH_INSTALL_PATH="/usr/local/lib/python3/dist-packages"
TORCH_VER_FILE="${TORCH_INSTALL_PATH}/torch/version.py"

if [[ -f "$TORCH_VER_FILE" ]] && grep -q '2\.9\.0' "$TORCH_VER_FILE"; then
  echo "PyTorch 2.9.0 already present at ${TORCH_INSTALL_PATH}"
else
  # --target does not overwrite an existing tree; drop only torch, not nvidia-*
  echo "Removing existing PyTorch under ${TORCH_INSTALL_PATH}..."
  rm -rf "${TORCH_INSTALL_PATH}/torch" "${TORCH_INSTALL_PATH}"/torch-*

  echo "Installing PyTorch 2.9.0 to ${TORCH_INSTALL_PATH}..."
  python3 -m pip install torch==2.9.0 \
      --index-url https://download.pytorch.org/whl/cu130 \
      --target="${TORCH_INSTALL_PATH}"
fi

echo "Verifying PyTorch version..."
grep __version__ "${TORCH_VER_FILE}"
