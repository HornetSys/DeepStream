<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Provisioning and verification

Read this before running the playbook on the target machine or verifying an
installation. The push-mode procedure remains in `SKILL.md` because it is the
default for a supplied remote host.

## Run-on-target mode (local machine)

Use when no remote `host` is given (configure the local machine), or when
pushing over SSH is not desired and the playbook is instead run directly on
the box.

For a true localhost run, use `assets/compute-stack-provisioning/` in place.
To run on a remote box in this mode, copy the tree over first (for example,
`scp -r assets/compute-stack-provisioning <user>@<host>:~/`). Install Ansible
on the machine as described in `SKILL.md` section 4.1; `ansible-playbook
--version` must print a version.

```bash
cd assets/compute-stack-provisioning
ansible-playbook -i inventory.ini ubuntu_setup.yml \
  -e "cuda_version=13-2"
  # ...only include -e lines for versions the operator overrode
  # Single component? append --tags <tag>, e.g. --tags cuda
```

The shipped `inventory.ini` targets `localhost` via `ansible_connection=local`
as `$USER`.

- The driver role skips the reboot on `localhost`; a manual reboot may be
  required before the driver loads (`nvidia-smi` may fail until then).
- `docker`-group membership for the user takes effect only after a new login
  session.

**Pass:** the play recap ends with `failed=0`.

## Verify

Run on the target after the play completes (for push mode, via SSH):

| Check | Command | Pass |
|-------|---------|------|
| Driver + GPU | `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` | GPU + requested driver version shown |
| CUDA toolkit | `dpkg -l \| grep cuda-toolkit-<cuda_version>` (or `/usr/local/cuda/bin/nvcc --version`) | package present at requested version |
| cuDNN | `dpkg -l \| grep libcudnn<major>` | pinned `cudnn_sub_version` present |
| TensorRT | `dpkg -l \| grep libnvinfer<major>` | pinned `trt_sub_version` present |
| Container runtime | `docker run --rm --gpus all ubuntu:24.04@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254 nvidia-smi` | container sees the GPU (the toolkit injects `nvidia-smi`) |

Verify only the components that were installed. For a single-component run,
run just that component's check (for example, driver-only → `nvidia-smi`). If
`nvidia-smi` fails on a host that has not rebooted yet (localhost, or a remote
host whose driver package did not change), flag that a reboot is required
before the driver loads, then re-verify.

## Examples

- **Full generic stack, remote, defaults:** “Provision the compute stack on
  `gpu-host-01` (SSH key at `/path/to/private-key`).” → push mode, no
  overrides, no `--tags`.
- **Remote, password auth (no key):** “Provision `gpu-host-01`, I only have a
  username/password.” → push mode with `--ask-pass` instead of `--private-key`.
- **Specific versions:** “Configure `10.0.0.5` with CUDA 13-2, driver
  595.45.04, TensorRT 10.16.” → matching `-e` overrides for each.
- **Single component:** “Install just the NVIDIA driver on this host.” →
  `--tags driver`.
- **Local machine:** “Set up the compute stack on this box.” → run-on-target
  mode with the shipped `localhost` inventory.
- **Add DeepStream layer:** “Also install the DeepStream OS deps.” →
  `--tags deepstream-deps` (or `--tags all,deepstream-deps`).
