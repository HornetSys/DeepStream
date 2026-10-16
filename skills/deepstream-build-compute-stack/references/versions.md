<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Version reference

How each version parameter maps to concrete apt packages and pins in the bundled
`compute-stack-provisioning/` playbook, and how to confirm a requested version actually exists.

## Defaults

| Parameter | Default | Set in |
|-----------|---------|--------|
| `nv_driver_version` | `595.45.04` | `assets/compute-stack-provisioning/defaults/main.yml` |
| `cuda_version` | `13-2` | `assets/compute-stack-provisioning/defaults/main.yml` |
| `cudnn_version` | `9` | `assets/compute-stack-provisioning/defaults/main.yml` |
| `cudnn_sub_version` | `9.20` | `assets/compute-stack-provisioning/defaults/main.yml` |
| `trt_version` | `10` | `assets/compute-stack-provisioning/defaults/main.yml` |
| `trt_sub_version` | `10.16` | `assets/compute-stack-provisioning/defaults/main.yml` |

`otel_version=v1.23.0` is checked out at immutable commit
`ea1f0d61ce5baa5584b097266bf133d1f31e3607`; update both values together when upgrading.

All are overridable with `-e "<var>=<value>"` on the `ansible-playbook` command line.

## How `ubuntu_setup.yml` transforms the inputs

Derived in the play `vars:` block (`assets/compute-stack-provisioning/ubuntu_setup.yml`):

- `cuda_version_arr = cuda_version.split('-')` → e.g. `13-2` → `['13','2']`.
- `cuda_major_version = cuda_version_arr[0] | int` → `13`. This major version drives the
  **cuDNN pin** so cuDNN stays matched to the installed CUDA.
- `cuda_dot_version_prod = cuda_version | replace('-','.')` → `13.2`.
- The remaining `*_prod` vars are pass-throughs of the driver/TRT/cuDNN inputs.

The `init` role builds the OS token as `<distro><major>` → `ubuntu24`; the `cuda` role
appends `04` → `ubuntu2404` for the NVIDIA repo path.

## Per-component mapping

### NVIDIA display driver (`roles/nv_display_driver`)

- `nv_driver_branch` = first dot-segment of `nv_driver_version` → `595.45.04` → `595`.
- Installs `nvidia-driver-pinning-<nv_driver_version>` (e.g. `nvidia-driver-pinning-595.45.04`)
  and the **open** kernel driver `nvidia-open-<branch>` (e.g. `nvidia-open-595`), plus
  `nvidia-xconfig`.
- Reboots remote hosts after install; skips reboot on `localhost`.

### CUDA toolkit (`roles/cuda`)

- Adds the CUDA keyring from
  `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/<arch>/cuda-keyring_1.1-1_all.deb`
  where `<arch>` is `sbsa` on `aarch64`, else the raw arch (`x86_64`).
- Installs `cuda-toolkit-<cuda_version>` (e.g. `cuda-toolkit-13-2`).

### cuDNN (`roles/cudnn`)

- Builds the pin `-cuda-<cuda_major>=<cudnn_sub_version>*` (e.g. `-cuda-13=9.20*`).
- Installs `libcudnn<cudnn_version>-headers-cuda-<major>`, `libcudnn<cudnn_version>...`,
  and `libcudnn<cudnn_version>-dev...`, each at the pinned sub-version (e.g. `libcudnn9...-cuda-13=9.20*`).

### TensorRT (`roles/tensorrt`)

- Builds the pin `=<trt_sub_version>*` (e.g. `=10.16*`).
- Installs the full `libnvinfer<trt_version>` family: `libnvinfer10`, `-dev`, `-plugin`,
  `-plugin-dev`, `libnvonnxparsers`, `-lean`, `-dispatch`, `-vc-plugin`, the `-headers-*-dev`
  variants, `python3-libnvinfer*` bindings, `libnvinfer-bin`, and `tensorrt`/`tensorrt-libs`/`tensorrt-dev`.
- `libnvinfer-win-builder-resource<trt_version>` is installed **only on x86_64**.

### NVIDIA Docker (`roles/nvidia-docker`)

- Not version-parameterized. Installs `docker.io`,
  `nvidia-container-toolkit`, runs `nvidia-ctk runtime configure`, and adds the user to
  the `docker` group.

### DeepStream OS deps (`roles/deepstream-deps`) — optional

DeepStream-specific and **opt-in** (tagged `never`): excluded from the default run, installed
only with `--tags deepstream-deps`. Other teams can ignore this role.

- GStreamer 1.0 (good/bad/ugly/libav + tools), `libssl-dev`, `libyaml-cpp-dev`,
  `libjansson4`, `libmosquitto-dev`, `python3-pip`, etc.
- OpenTelemetry-C++ built from source at `otel_version` (default `v1.23.0`, in
  `roles/deepstream-deps/defaults/main.yml`). Uses `.configured.<ver>`/`.installed.<ver>`
  stamp files to skip already-completed steps on re-run.

## Installing individual components

For the component → `--tags` mapping and the install-scope rules, see the canonical table in
`SKILL.md` §2.1. This file covers only the version → apt-package mapping.

## Confirming a version exists before running

A pin that doesn't exist in the NVIDIA apt repo fails the run partway through. To check:

- **CUDA / cuDNN / TensorRT**: after the CUDA keyring is present, on the target run
  `apt-cache madison <package>` (e.g. `apt-cache madison cuda-toolkit-13-2`,
  `apt-cache madison libnvinfer10`) to list available versions.
- **Driver**: `apt-cache search nvidia-open-<branch>` and check the driver's release notes
  for the exact `<nv_driver_version>`.
- Browse the repo directly at
  `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/<arch>/`.
