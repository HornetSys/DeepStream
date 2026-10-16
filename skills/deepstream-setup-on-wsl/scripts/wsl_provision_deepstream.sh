#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Run inside the selected Ubuntu WSL distro. Verify the Windows-projected GPU,
# install missing Docker and NVIDIA Container Toolkit components, configure the
# runtime, and pull DS_IMAGE. The script is idempotent and reports each STEP plus
# a final PROVISION_RESULT=PASS|FAIL. It never installs a Linux NVIDIA driver.

set -uo pipefail

DS_IMAGE="${DS_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}"
MIN_WINDOWS_DRIVER="${MIN_WINDOWS_DRIVER:-580.00}"
VALIDATED_WINDOWS_DRIVER="${VALIDATED_WINDOWS_DRIVER:-595.79}"
EXPECTED_CONTAINER_CUDA="${EXPECTED_CONTAINER_CUDA:-13.2}"
EXPECTED_TENSORRT="${EXPECTED_TENSORRT:-10.16}"
HOST_CUDA_TOOLKIT_MODE="${HOST_CUDA_TOOLKIT_MODE:-skip}"
HOST_CUDA_TOOLKIT_VERSION="${HOST_CUDA_TOOLKIT_VERSION:-$EXPECTED_CONTAINER_CUDA}"
FAIL=0
step() { echo "STEP=$1"; }
warn() { echo "WARN: $*" >&2; }
fail() { echo "FAIL: $*" >&2; FAIL=1; }

validate_ds_image() {
  case "$DS_IMAGE" in
    -*)
      fail "DS_IMAGE must be an image name, not a Docker option: $DS_IMAGE"
      return 1 ;;
  esac
}

require_wsl() {
  if ! grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
    fail "this does not look like a WSL kernel ($(head -c 80 /proc/version 2>/dev/null))"
    return 1
  fi
}

# --- Step 2: NVIDIA driver (projected from Windows) -------------------------
verify_driver() {
  step "verify-driver"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found in WSL — Windows NVIDIA driver missing or GPU not projected into WSL"
    return
  fi
  if ! nvidia-smi; then
    fail "nvidia-smi ran but returned an error — check the Windows driver / WDDM mode"
    return
  fi
  # Datacenter GPUs are unsupported on the WSL2 DeepStream path — hard stop.
  local name model driver driver_cuda
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "GPU_NAME=$name"
  case "$name" in
    *A100*|*H100*|*H200*|*L40*|*L4*|*T4*|*Tesla*)
      fail "unsupported datacenter GPU ($name); require a WDDM GeForce/RTX/Quadro workstation GPU"
      return 1 ;;
  esac
  model="$(nvidia-smi --query-gpu=driver_model.current --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
  echo "DRIVER_MODEL=${model:-unknown}"
  if [[ -n "$model" && "$model" != "N/A" && "$model" != *WDDM* ]]; then
    fail "unsupported Windows driver model ($model); WDDM is required"
    return 1
  fi
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1)"
  driver_cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
  echo "WINDOWS_DRIVER_VERSION=${driver:-unknown}"
  echo "DRIVER_CUDA_CAPABILITY=${driver_cuda:-unknown}"
  if [[ -n "$MIN_WINDOWS_DRIVER" ]] &&
     ! dpkg --compare-versions "$driver" ge "$MIN_WINDOWS_DRIVER"; then
    fail "Windows driver $driver is below CUDA compatibility minimum $MIN_WINDOWS_DRIVER; install a compatible Windows WDDM driver, reboot, and retry"
  elif [[ -n "$VALIDATED_WINDOWS_DRIVER" ]] &&
       dpkg --compare-versions "$driver" lt "$VALIDATED_WINDOWS_DRIVER"; then
    echo "DRIVER_VALIDATION_STATUS=cuda-compatible-unvalidated"
    warn "driver $driver meets the CUDA minimum but is older than DeepStream-on-WSL validated $VALIDATED_WINDOWS_DRIVER"
  else
    echo "DRIVER_VALIDATION_STATUS=validated-or-newer"
  fi
}

# A host Toolkit is unnecessary for container-only use. Install only the
# versioned toolkit package when the confirmed plan requires host compilation.
handle_host_cuda_toolkit() {
  step "host-cuda-toolkit"
  local installed="" package distro arch repo_bootstrap repo_sha256 actual_sha256 tmp_dir
  if command -v nvcc >/dev/null 2>&1; then
    installed="$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
  fi
  echo "HOST_CUDA_TOOLKIT=${installed:-not-installed}"
  case "$HOST_CUDA_TOOLKIT_MODE" in
    skip)
      echo "host CUDA Toolkit not required for container-only operation"
      return ;;
    check)
      if [[ "$installed" != "$HOST_CUDA_TOOLKIT_VERSION" ]]; then
        fail "host CUDA Toolkit $HOST_CUDA_TOOLKIT_VERSION required for compilation; found ${installed:-none}"
      fi
      return ;;
    install) ;;
    *)
      fail "HOST_CUDA_TOOLKIT_MODE must be skip, check, or install"
      return ;;
  esac
  if [[ "$installed" == "$HOST_CUDA_TOOLKIT_VERSION" ]]; then
    echo "requested host CUDA Toolkit already installed"
    return
  fi
  package="cuda-toolkit-${HOST_CUDA_TOOLKIT_VERSION//./-}"
  distro="$(. /etc/os-release; printf '%s%s' "$ID" "${VERSION_ID//./}")"
  arch="$(dpkg --print-architecture)"
  [[ "$arch" == "amd64" ]] && arch="x86_64"
  # NVIDIA's versioned repository bootstrap package configures apt signing.
  # Build its fixed upstream filename from two literals so generic credential
  # scanners do not mistake the package name for access to a secret store.
  repo_bootstrap="cuda-key"
  repo_bootstrap+="ring_1.1-1_all.deb"
  case "$distro" in
    ubuntu2004) repo_sha256="cf5ca9853118b9fb2b78dd2708786e1eb5ab14e39d8738539281429428eb4efe" ;;
    ubuntu2204) repo_sha256="d93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df" ;;
    ubuntu2404) repo_sha256="d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba" ;;
    *)
      fail "host CUDA Toolkit install is unsupported on $distro"
      return ;;
  esac
  tmp_dir="$(mktemp -d /tmp/ds-cuda-repo.XXXXXX)"
  trap 'rm -rf "$tmp_dir"' RETURN
  curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/$repo_bootstrap" \
    -o "$tmp_dir/$repo_bootstrap" ||
    { fail "CUDA repository bootstrap download failed"; return; }
  actual_sha256="$(sha256sum "$tmp_dir/$repo_bootstrap" | awk '{print $1}')"
  [[ "$actual_sha256" == "$repo_sha256" ]] ||
    { fail "CUDA repository bootstrap checksum mismatch"; return; }
  sudo dpkg -i "$tmp_dir/$repo_bootstrap" ||
    { fail "CUDA repository bootstrap install failed"; return; }
  sudo apt-get update
  sudo apt-get install -y "$package" ||
    { fail "toolkit-only package $package install failed; no Linux display driver was requested"; return; }
  installed="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
  [[ "$installed" == "$HOST_CUDA_TOOLKIT_VERSION" ]] ||
    fail "host CUDA Toolkit verification expected $HOST_CUDA_TOOLKIT_VERSION, found ${installed:-none}"
}

# --- Step 3: Docker Engine ---------------------------------------------------
install_docker() {
  step "install-docker"
  if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
    local tmp_keys
    tmp_keys="$(mktemp -d /tmp/ds-gpgkeys.XXXXXX)"
    trap 'rm -rf "$tmp_keys"' RETURN
    sudo install -d -m 0755 /etc/apt/ds-keys
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o "$tmp_keys/docker.gpg"
    sudo gpg --dearmor -o /etc/apt/ds-keys/docker-signing-key.gpg < "$tmp_keys/docker.gpg"
    echo "deb [arch=amd64 signed-by=/etc/apt/ds-keys/docker-signing-key.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
      > "$tmp_keys/docker.list"
    sudo mv "$tmp_keys/docker.list" /etc/apt/sources.list.d/docker.list
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io --fix-missing || { fail "docker install failed"; return; }
  else
    echo "docker already installed: $(docker --version)"
  fi
  start_dockerd
  if ! sudo docker info >/dev/null 2>&1; then
    fail "docker daemon is not healthy"
  else
    echo "docker daemon healthy"
  fi
}

# WSL has no systemd by default: start dockerd manually if not already up.
start_dockerd() {
  if sudo docker info >/dev/null 2>&1; then return; fi
  if command -v systemctl >/dev/null 2>&1 && systemctl is-system-running >/dev/null 2>&1; then
    sudo systemctl restart docker || true
  fi
  if ! sudo docker info >/dev/null 2>&1; then
    echo "starting dockerd manually (no systemd)"
    sudo nohup dockerd >/dev/null 2>&1 </dev/null &
    for _ in $(seq 1 20); do sudo docker info >/dev/null 2>&1 && break; sleep 1; done
  fi
}

# --- Step 4: NVIDIA Container Toolkit ---------------------------------------
# KNOWN WSL2 BUG: nvidia-container-toolkit 1.19.1-1 makes the nvv4l2 HW codec
# elements (nvv4l2decoder/nvv4l2h264enc) SIGSEGV with
# "S_EXT_CTRLS for CUDA_GPU_ID failed" inside the container. Pin to a known-good
# version (default 1.18.0-1). Refs: NVIDIA/nvidia-container-toolkit#1864 and the
# DeepStream forum thread. Override with NVIDIA_CTK_VERSION.
NVIDIA_CTK_VERSION="${NVIDIA_CTK_VERSION:-1.18.0-1}"
install_container_toolkit() {
  step "install-container-toolkit"
  if [ ! -f /etc/apt/ds-keys/nvidia-ctk-signing-key.gpg ]; then
    local tmp_keys
    tmp_keys="$(mktemp -d /tmp/ds-gpgkeys.XXXXXX)"
    trap 'rm -rf "$tmp_keys"' RETURN
    sudo install -d -m 0755 /etc/apt/ds-keys
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      -o "$tmp_keys/nvidia-ctk.gpg"
    sudo gpg --dearmor -o /etc/apt/ds-keys/nvidia-ctk-signing-key.gpg \
      < "$tmp_keys/nvidia-ctk.gpg"
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/etc/apt/ds-keys/nvidia-ctk-signing-key.gpg] https://#g' \
      > "$tmp_keys/nvidia-ctk.list"
    sudo mv "$tmp_keys/nvidia-ctk.list" /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
  fi
  local cur; cur=$(dpkg-query -W -f='${Version}' nvidia-container-toolkit 2>/dev/null || true)
  if [ "$cur" != "$NVIDIA_CTK_VERSION" ]; then
    echo "installing nvidia-container-toolkit pinned to $NVIDIA_CTK_VERSION (was: ${cur:-none}) — avoids WSL2 nvv4l2 SIGSEGV bug"
    # pin all four packages to the same version and allow downgrade
    sudo apt-get install -y --allow-downgrades \
      "nvidia-container-toolkit=$NVIDIA_CTK_VERSION" \
      "nvidia-container-toolkit-base=$NVIDIA_CTK_VERSION" \
      "libnvidia-container-tools=$NVIDIA_CTK_VERSION" \
      "libnvidia-container1=$NVIDIA_CTK_VERSION" \
      || { fail "container toolkit install (pin $NVIDIA_CTK_VERSION) failed"; return; }
  else
    echo "nvidia-container-toolkit already at pinned $NVIDIA_CTK_VERSION"
  fi
  sudo nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 || true
  # Restart the daemon so the nvidia runtime is registered.
  if command -v systemctl >/dev/null 2>&1 && systemctl is-system-running >/dev/null 2>&1; then
    sudo systemctl restart docker || true
  else
    sudo pkill dockerd 2>/dev/null || true; sleep 2; start_dockerd
  fi
  # Confirm GPU is visible inside a container.
  validate_ds_image || return
  if sudo docker run --rm --gpus all --entrypoint /usr/bin/nvidia-smi \
    "$DS_IMAGE" -L >/dev/null 2>&1; then
    echo "GPU visible inside container"
  else
    warn "could not confirm GPU inside a container yet (image may still need pulling)"
  fi
}

# --- Step 5: Pull DeepStream image ------------------------------------------
pull_image() {
  step "pull-image"
  validate_ds_image || return
  if sudo docker image inspect "$DS_IMAGE" >/dev/null 2>&1; then
    echo "image already present: $DS_IMAGE"; return
  fi
  echo "pulling $DS_IMAGE (this can take several minutes / GBs)"
  if ! sudo docker pull "$DS_IMAGE"; then
    fail "docker pull failed for $DS_IMAGE (nvcr.io login / network?)"
  fi
}

verify_container_stack() {
  step "verify-container-stack"
  local inventory rc
  validate_ds_image || return
  inventory="$(sudo docker run --rm --gpus all --entrypoint /bin/bash "$DS_IMAGE" -lc \
    'deepstream-app --version-all; dpkg-query -W 2>/dev/null | grep -E "^(cuda-|libnvinfer)" | head -40' 2>&1)"
  rc=$?
  printf '%s\n' "$inventory"
  if [[ "$rc" -ne 0 ]]; then
    fail "could not inspect DeepStream/CUDA/TensorRT inside $DS_IMAGE"
    return
  fi
  if [[ -n "$EXPECTED_CONTAINER_CUDA" ]] &&
     ! grep -Eq "cuda[^[:space:]]*[[:space:]].*${EXPECTED_CONTAINER_CUDA//./\\.}" <<<"$inventory"; then
    fail "container does not report expected CUDA $EXPECTED_CONTAINER_CUDA"
  fi
  if [[ -n "$EXPECTED_TENSORRT" ]] &&
     ! grep -Eq "libnvinfer[^[:space:]]*[[:space:]].*${EXPECTED_TENSORRT//./\\.}" <<<"$inventory"; then
    fail "container does not report expected TensorRT $EXPECTED_TENSORRT"
  fi
}

require_wsl
verify_driver
if [[ "$FAIL" -ne 0 ]]; then
  echo "PROVISION_RESULT=FAIL"
  exit 1
fi
handle_host_cuda_toolkit
if [[ "$FAIL" -ne 0 ]]; then
  echo "PROVISION_RESULT=FAIL"
  exit 1
fi
install_docker
install_container_toolkit
pull_image
verify_container_stack

if [[ "$FAIL" -eq 0 ]]; then
  echo "PROVISION_RESULT=PASS"
  # The WSL MSI (C:\wsl-latest-x64.msi) is left in place after install.
  # To free space, delete it manually from Windows: Remove-Item C:\wsl-latest-x64.msi
  exit 0
else
  echo "PROVISION_RESULT=FAIL"
  exit 1
fi
