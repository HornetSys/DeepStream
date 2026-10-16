<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepStream Build Compute Stack

`deepstream-build-compute-stack` provisions an Ubuntu 24.04 host (x86_64 or ARM SBSA) with a chosen
NVIDIA compute stack — the **open** display driver, CUDA toolkit, cuDNN, TensorRT, and the
NVIDIA container/Docker runtime — by running a bundled, version-pinned Ansible playbook. It
helps an agent collect the target and desired versions, run the playbook (pushed over SSH or
on the target), handle the driver reboot, verify the result, and emit a fixed build report.
It installs the **full stack** by default or a **single component** on request, and exposes an
optional, opt-in DeepStream OS-deps layer.

The skill is designed for agents that support the Agent Skills folder convention. It targets **x86_64** and **ARM SBSA** only; Jetson is out of scope (use
[NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) / JetPack).

## What is included

```
skills/deepstream-build-compute-stack/
├── SKILL.md                         # the main workflow the agent follows
├── README.md                        # this file
├── skill-card.md                    # catalog card (generated)
├── BENCHMARK.md                     # evaluation metrics (generated)
├── evals/evals.json                 # trigger/behavior evals
├── references/
│   ├── versions.md                  # version → apt-package mapping; how to confirm a version exists
│   ├── provisioning.md               # on-target and verification procedures
│   └── troubleshooting.md           # error → cause → solution table
└── assets/compute-stack-provisioning/
    ├── ubuntu_setup.yml             # the playbook
    ├── inventory.ini                # localhost inventory (on-target mode)
    ├── defaults/main.yml            # default versions (driver/CUDA/cuDNN/TensorRT)
    └── roles/                       # init, cuda, nv_display_driver, nvidia-docker,
                                     #   cudnn, tensorrt, deepstream-deps (opt-in)
```

The main workflow is in `SKILL.md`. The provisioning logic is entirely Ansible; the component
→ tag mapping and install-scope rules are documented once in `SKILL.md` §2.1, the on-target
and verification procedures in `references/provisioning.md`, the version → package mapping in
`references/versions.md`, and error handling in
`references/troubleshooting.md`.

## Safety

Provisioning changes system repositories, packages, kernel drivers, Docker configuration,
and display settings with elevated privileges. It can reboot remote hosts and interrupt
services or graphical sessions. Prefer a non-production host; otherwise take a recoverable
snapshot or backup and confirm the exact target before running the playbook.

## Prerequisites

**Control machine** (where you run the playbook from, for push mode):

- Ansible (`ansible-playbook`).
  ```bash
  sudo apt update
  sudo apt install -y ansible
  ```
  No sudo, or a PEP 668 "externally-managed-environment" error (default on Ubuntu 24.04)?
  Install Ansible without root via `pipx install --include-deps ansible`, a venv
  (`python3 -m venv` — needs `python3-venv`), or `pip install --user --break-system-packages ansible`.
- `sshpass`, **only if you'll use username/password SSH auth** instead of a key:
  ```bash
  sudo apt install -y sshpass
  ```

**Target host:**

- Ubuntu 24.04 (x86_64 or ARM SBSA) — the playbook asserts this and fails fast otherwise.
- `sudo`/`become` privileges.
- Outbound internet to the NVIDIA apt repos (plus `nvidia.github.io` + GitHub only if the
  optional DeepStream deps are requested).

**Credentials:** remote push mode accepts either an SSH private key (`--private-key <path>`,
the default) or a username/password, collected interactively with `--ask-pass` (requires
`sshpass` on the control machine, above) — pick whichever the operator has available. If sudo
requires a password, add `--ask-become-pass` interactively as well. Never commit or paste
credentials into an agent prompt, inventory, or command argument; passwords are only ever
entered at the interactive prompt or sourced from an approved vault/secret-manager lookup.

## Install the skill

Copy the skill folder into the skills directory of your agent (Claude Code, Codex or Cursor),
for example for a user-level Codex install:

```bash
mkdir -p ~/.codex/skills
cp -r skills/deepstream-build-compute-stack ~/.codex/skills/
```

Restart the agent after the first installation so the skill is added to its available-skills
list. See [skills/README.md](../README.md) for the per-agent locations.

## Invoke the skill by name

Provide the target and, optionally, versions. Examples:

**Full stack on a remote host (defaults):**
```
Use deepstream-build-compute-stack to provision the NVIDIA compute stack on gpu-host-01
(SSH key at /path/to/private-key). Use default versions.

First confirm the host is Ubuntu 24.04 and get my approval before provisioning —
it reboots the machine. Then install the full stack and verify each component.
```

**Specific versions:**
```
Use deepstream-build-compute-stack to configure 10.0.0.5 with CUDA 13-2, driver 595.45.04,
and TensorRT 10.16. Verify with nvidia-smi / nvcc / docker-GPU when done.
```

**Single component (others must NOT be installed):**
```
Use deepstream-build-compute-stack to install just the NVIDIA driver on 10.0.0.5.
Do not install CUDA, cuDNN, TensorRT, or Docker.
```

**Local machine:**
```
Use deepstream-build-compute-stack to set up the compute stack on this box (on-target mode).
```

**Remote host, password auth (no SSH key):**
```
Use deepstream-build-compute-stack to provision the NVIDIA compute stack on gpu-host-01.
I don't have an SSH key for it, only a username and password — prompt me for the
password interactively rather than a key.
```

## Approval gate

Provisioning **reboots the host and rewrites its GPU/display config**, so the skill pauses for
one explicit approval before touching a shared/lab machine:

> Confirm the target host identity and that it is safe to reboot and reconfigure before the
> playbook runs.

In push mode the driver install reboots the remote host and reconnects automatically — treat
the mid-run disconnect + reconnect as expected, not a failure.

## Configuration

Versions are fully parameterized; pass `-e` only for the versions the operator explicitly
named, otherwise the defaults apply.

| Variable | Default | Notes |
|----------|---------|-------|
| `nv_driver_version` | `595.45.04` | NVIDIA open driver; branch derived from the major |
| `cuda_version` | `13-2` | format `XX-X`; drives the cuDNN pin |
| `cudnn_version` / `cudnn_sub_version` | `9` / `9.20` | apt pin |
| `trt_version` / `trt_sub_version` | `10` / `10.16` | apt pin |

Install scope is controlled by `--tags`: no tag → the full generic stack; a single tag
(`driver`, `cuda`, `cudnn`, `tensorrt`, `docker`) → that component only; `deepstream-deps` →
the opt-in DeepStream layer. See `SKILL.md` §2.1 for the full mapping and rules, and
`references/versions.md` for the version → package mapping and how to confirm a pin exists.

## Run and verify

Run these from the skill directory (`skills/deepstream-build-compute-stack/`).

**Push mode (remote host).** The bundled `assets/compute-stack-provisioning/inventory.ini`
targets `localhost` and is only for the on-target case — for a remote host, write your own
inventory (e.g. `/tmp/csb-inventory.ini`) and never put a password in it:

```ini
[target]
gpu-host-01 ansible_user=your-username

[all:vars]
ansible_ssh_common_args='-o StrictHostKeyChecking=accept-new'
```

`accept-new` still fails on a *changed* host key (MITM protection); if a re-imaged host
trips `REMOTE HOST IDENTIFICATION HAS CHANGED`, clear its stale key with `ssh-keygen -R <host>`.

Preflight (confirm reachability + Ubuntu 24.04 before provisioning), then run the playbook
(full stack, or add `--tags <component>` for one component). Use an SSH key (default) or,
if the operator only has a username/password, swap `--private-key <key>` for `--ask-pass`
(requires `sshpass` on the control machine):

```bash
# Key auth
ansible -i /tmp/csb-inventory.ini target --private-key <key> \
  -m setup -a 'filter=ansible_distribution*'

ansible-playbook -i /tmp/csb-inventory.ini \
  assets/compute-stack-provisioning/ubuntu_setup.yml \
  --private-key <key> --ask-become-pass \
  -e "cuda_version=13-2"          # only for versions the operator overrode

# Password auth (alternative to a key)
ansible -i /tmp/csb-inventory.ini target --ask-pass \
  -m setup -a 'filter=ansible_distribution*'

ansible-playbook -i /tmp/csb-inventory.ini \
  assets/compute-stack-provisioning/ubuntu_setup.yml \
  --ask-pass --ask-become-pass \
  -e "cuda_version=13-2"          # only for versions the operator overrode
```

`--ask-pass` prompts for the SSH password interactively — it is never written to the
inventory, a `-e` value, or any log.

**On-target mode (local machine).** Use the bundled localhost inventory in place:

```bash
cd assets/compute-stack-provisioning
ansible-playbook -i inventory.ini ubuntu_setup.yml
```

A green run ends with `failed=0` for the target. Verify on the target:

| Check | Command |
|-------|---------|
| Driver + GPU | `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` |
| CUDA toolkit | `/usr/local/cuda/bin/nvcc --version` |
| cuDNN | `dpkg -l \| grep libcudnn<major>` |
| TensorRT | `dpkg -l \| grep libnvinfer<major>` |
| Container runtime | `docker run --rm --gpus all ubuntu:24.04@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254 nvidia-smi` |

## Expected deliverables

- The requested components installed at the requested (or default) versions.
- The driver reboot handled (or flagged as required, on localhost).
- Per-component verification (`nvidia-smi` / `nvcc` / cuDNN / TensorRT / docker-GPU).
- The fixed build-complete report (host, arch, mode, per-component status, reboot, verify).
## Validate changes locally

Run these from the **repo root** (the paths below are repo-root-relative):

```bash
# playbook parses
ansible-playbook --syntax-check \
  skills/deepstream-build-compute-stack/assets/compute-stack-provisioning/ubuntu_setup.yml

# optional lint
ansible-lint skills/deepstream-build-compute-stack/assets/compute-stack-provisioning/

# skill schema/quality
python3 /path/to/skill-creator/scripts/quick_validate.py \
  skills/deepstream-build-compute-stack

git diff --check
```

Validate against a disposable Ubuntu 24.04 host — a full run reboots the target and rewrites
its GPU/display config, so never point it at a machine you are not authorized to reprovision.
