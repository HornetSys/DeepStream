#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Extracts the Tegra multimedia stack out of the public NVIDIA Jetson Linux
# Driver Package (BSP) so it can be baked into the SBSA container image.
#
# SBSA hosts (GB200 / Grace / DGX Spark) carry no Jetson BSP and the container
# runtime does not inject Tegra libraries the way it does on Jetson, so the
# libraries have to ship inside the image. The OpenRM multimedia variant is the
# one that works against a discrete GPU.
#
# Usage: extract_packages.sh <Jetson_Linux_R<ver>_aarch64.tbz2 | Linux_for_Tegra dir>
#
# Output lands in /tmp/output_extract/loc and is copied into the image by the
# Dockerfile. The BSP's Debian package version is written to
# /tmp/output_extract/l4t_deb_version so the Dockerfile can fetch the matching
# nvidia-l4t-gstreamer package, which the public BSP does not include.
# To try the script outside docker, override OUTPUT_DIR:
#   OUTPUT_DIR="$PWD/output_extract" ./extract_packages.sh <bsp>

BSP_PACKAGE=$1
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/output_extract}"
TEMP_DIR="${TEMP_DIR:-/tmp/temp_extract}"

set -o errtrace
trap 'echo "CUSTOM MESSAGE: ERROR occured in this file: $BASH_SOURCE"' ERR
set -o errexit
set -o nounset

if [[ -z "${BSP_PACKAGE}" ]]; then
    echo "error: no Jetson Linux BSP supplied." >&2
    echo "       usage: $0 <Jetson_Linux_R<ver>_aarch64.tbz2 | Linux_for_Tegra dir>" >&2
    echo "       Download the Driver Package (BSP) from" >&2
    echo "       https://developer.nvidia.com/embedded/jetson-linux and stage it" >&2
    echo "       with setup_sbsa_build.sh." >&2
    exit 1
fi

if [[ ! -e "${BSP_PACKAGE}" ]]; then
    echo "error: '${BSP_PACKAGE}' does not exist." >&2
    exit 1
fi

echo "creating output dir"
mkdir -p "$OUTPUT_DIR/loc" "$TEMP_DIR"
echo "created output dir"
ls -la "$OUTPUT_DIR/loc"

# Accept either the downloaded BSP tarball or an already-extracted tree. The
# public tarball unpacks straight to Linux_for_Tegra/; some internal packages
# nest it one level deeper inside linux_for_tegra.tbz2, so unwrap that too.
if [[ -d "${BSP_PACKAGE}" ]]; then
    echo "Using pre-extracted BSP tree: ${BSP_PACKAGE}"
    BSP_ROOT="${BSP_PACKAGE}"
else
    echo "Extracting BSP: ${BSP_PACKAGE}"
    tar -xf "${BSP_PACKAGE}" -C "$TEMP_DIR"

    NESTED_BSP=$(find "$TEMP_DIR" -name 'linux_for_tegra.tbz2' -print -quit)
    if [[ -n "${NESTED_BSP}" ]]; then
        echo "Unwrapping nested BSP: ${NESTED_BSP}"
        tar -xf "${NESTED_BSP}" -C "$(dirname "${NESTED_BSP}")"
    fi

    BSP_ROOT=$(find "$TEMP_DIR" -maxdepth 3 -type d -name 'Linux_for_Tegra' -print -quit)
    if [[ -z "${BSP_ROOT}" ]]; then
        echo "error: no Linux_for_Tegra directory found inside ${BSP_PACKAGE}." >&2
        echo "       Expected the NVIDIA Jetson Linux Driver Package (BSP)." >&2
        exit 1
    fi
fi

echo "BSP root: ${BSP_ROOT}"

# The archives sit directly in nv_tegra/ in the public BSP, but some internal
# packages nest the OpenRM ones under nv_tegra/openrm/, so locate them by name
# instead of by path.
extract_bsp_archive() {
    local archive_name=$1
    local archive_path

    archive_path=$(find "${BSP_ROOT}/nv_tegra" -name "${archive_name}" -print -quit)
    if [[ -z "${archive_path}" ]]; then
        echo "error: ${archive_name} not found under ${BSP_ROOT}/nv_tegra." >&2
        if [[ "${archive_name}" == *_openrm.tbz2 ]]; then
            echo "       Some BSP downloads only generate the OpenRM archives once" >&2
            echo "       apply_binaries.sh has run. From ${BSP_ROOT} try:" >&2
            echo "         sudo ./apply_binaries.sh --openrm" >&2
        fi
        exit 1
    fi

    echo "Extracting ${archive_path}"
    tar -xf "${archive_path}" -C "$OUTPUT_DIR/loc/"
}

extract_bsp_archive nvidia_drivers.tbz2
extract_bsp_archive nvidia_l4t_multimedia_openrm.tbz2

# Every Debian package in a BSP release carries the same version string, so
# read it off nvidia-l4t-multimedia for the apt download in the Dockerfile.
MULTIMEDIA_DEB=$(find "${BSP_ROOT}/nv_tegra" -name 'nvidia-l4t-multimedia_*_arm64.deb' -print -quit)
if [[ -z "${MULTIMEDIA_DEB}" ]]; then
    echo "error: nvidia-l4t-multimedia_*_arm64.deb not found under ${BSP_ROOT}/nv_tegra;" >&2
    echo "       cannot determine the BSP package version." >&2
    exit 1
fi
L4T_DEB_VERSION=$(dpkg-deb -f "${MULTIMEDIA_DEB}" Version)
echo "BSP package version: ${L4T_DEB_VERSION}"
echo "${L4T_DEB_VERSION}" > "$OUTPUT_DIR/l4t_deb_version"

rm -rf "$TEMP_DIR"

echo "BSP extraction complete: $OUTPUT_DIR/loc"
