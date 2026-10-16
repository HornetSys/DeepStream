---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: deepstream-build-compute-stack
description: "Provision Ubuntu 24.04 NVIDIA stacks with Ansible on x86_64 or ARM SBSA, with Xorg setup and optional DeepStream dependencies. Not for Jetson."
license: Apache-2.0
metadata:
  author: "Kamal Mansata <kmansata@nvidia.com>"
  version: 1.0.0
  tags:
    - ansible
    - cuda
    - tensorrt
    - cudnn
    - nvidia-driver
    - nvidia-docker
    - provisioning
    - gpu
    - ubuntu
    - x86
    - sbsa
  domain: provisioning
---

# DeepStream Build Compute Stack

Configure a target system with a chosen NVIDIA compute stack by running the bundled
Ansible playbook (`assets/compute-stack-provisioning/`). The agent collects the target and the desired
versions, runs the playbook, handles the driver reboot, verifies the result, and reports.

This is a **general-purpose** skill for any team that needs a GPU host provisioned. The
default (generic) stack installs the NVIDIA **open** display driver, the CUDA toolkit,
cuDNN, TensorRT, and the NVIDIA container and Docker runtime. Versions are fully parameterized.
An optional DeepStream-specific layer (GStreamer + OpenTelemetry-C++ built from source) is
available opt-in via `--tags deepstream-deps` and is **not** part of the default run.

After the stack is built, install your framework or SDK on top — e.g. for DeepStream, the
`deepstream-setup` skill; other teams layer their own runtime.

---

## Purpose

Give any team a repeatable, version-pinned way to provision the NVIDIA compute stack on a
GPU host, driven from an AI tool, instead of hand-running driver, CUDA, and TensorRT install steps.
Use it to stand up a fresh host, pin a specific stack for reproducing an issue, or add a
single missing component to an existing host.

## Instructions

1. Collect the target (`host` for push over SSH; omit to configure the local machine) and any
   versions the operator explicitly named — otherwise use the defaults.
2. Preflight: ensure Ansible on the control machine (§4.1) and confirm the target is Ubuntu 24.04.
   For a shared or lab host, confirm its identity and get approval before provisioning — the run
   reboots and reconfigures it (§4.2, hard rule 4).
3. Run the bundled playbook — the **full stack** by default, or a **single component** with
   `--tags <tag>` (see §2.1). Only pass `-e` overrides for versions the operator specified.
4. In push mode the driver install reboots the host; the run reconnects on its own — do not ask
   the operator for input, just poll for availability at a reasonable interval and continue once
   the host is back. (On-target/`localhost` the reboot is **skipped** — a manual reboot may be
   needed before the driver loads, so a post-install `nvidia-smi` failure there is not a run failure.)
5. Read [`references/provisioning.md`](references/provisioning.md) before an on-target run or verification, then emit the final report (§7).

Full procedure: push mode → §4; on-target and verification → [`references/provisioning.md`](references/provisioning.md). Hard rules → §5.

---

## 1. Prerequisites — scope, requirements, and knowledge

| Aspect | Supported / required |
|--------|----------------------|
| OS | **Ubuntu 24.04 only** — the playbook asserts `24.04 <= version < 25.00` and fails fast otherwise |
| Architecture | **x86_64** and **ARM SBSA** (`aarch64`) |
| Jetson | **Out of scope** — use [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) / JetPack instead |
| Privileges | Requires `sudo`/`become` on the target |
| Network | Target needs internet access (NVIDIA apt repos; https://nvidia.github.io + GitHub only if the optional DeepStream deps are requested) |
| Driver | Installs the NVIDIA **open** kernel driver (not the proprietary module) |
| Reboot | Driver install **reboots** remote hosts; on `localhost` the reboot is skipped and must be done manually before the driver loads |

Role execution order: `init` → `cuda` → `nv_display_driver` → `nvidia-docker` → `cudnn` →
`tensorrt` run by default; `deepstream-deps` is **opt-in only** (`--tags deepstream-deps`,
tagged `never`). What each role/tag installs is in §2.1's tag-mapping table.

---

## 2. Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `host` | No | SSH target `user@host` or IP. **Omit** → configure the local machine | `ubuntu@10.0.0.5` |
| `ssh_key` | If remote (push) and no password | Path to SSH private key | `/path/to/private-key` |
| `ssh_auth` | If remote (push) | `key` (default, uses `ssh_key`) or `password` — password is always collected interactively, never as a parameter value | `password` |
| `mode` | No | `push` or `on-target`. Auto: remote `host` → `push`; else `on-target` | `push` |
| `nv_driver_version` | No | NVIDIA open driver version. Default `595.45.04` | `595.45.04` |
| `cuda_version` | No | CUDA version, format `XX-X`. Default `13-2` | `13-2` |
| `cudnn_version` | No | cuDNN major version. Default `9` | `9` |
| `cudnn_sub_version` | No | cuDNN sub-version (apt pin). Default `9.20` | `9.20` |
| `trt_version` | No | TensorRT major version. Default `10` | `10` |
| `trt_sub_version` | No | TensorRT sub-version (apt pin). Default `10.16` | `10.16` |
| `components` | No | Which components to install. Default: the **generic stack** (driver, CUDA, cuDNN, TensorRT, Docker). Set to install a single component or a subset, or `deepstream-deps` for the optional DeepStream layer | `driver` |

**Input source precedence:** a value the operator explicitly names in their prompt always
wins; anything not named falls back to the playbook's built-in defaults
(`assets/compute-stack-provisioning/defaults/main.yml`). There is no state file or
persisted config between runs — every invocation resolves its inputs fresh from what the
operator says this time plus those defaults. Collect the target first, then the versions.
**Only pass `-e` overrides for versions the operator explicitly specified** — everything
else uses the playbook defaults. Never invent version strings; if the operator names a
version, use it verbatim.

SSH auth for push mode is **either** a private key (default, `--private-key`) **or** a
username/password (`--ask-pass`, entered interactively) — never both requested at once. Prefer
the key when the operator has one; fall back to password auth only when they say they don't
have (or don't want to use) a key. See §4.3 for how password auth is collected and passed.

### 2.1 Install scope (full stack or single component)

The playbook is tagged so it can install the **generic stack** (default) or **one component**.
Map the operator's request to a tag:

| Component the operator asks for | Tag | What runs |
|---------------------------------|-----|-----------|
| Generic stack (no component named) | *(none)* | driver + cuda + cudnn + tensorrt + docker (excludes `deepstream-deps`) |
| NVIDIA display driver | `driver` | CUDA repo and keyring + `nvidia-open` plus pinning + Xorg config |
| CUDA toolkit | `cuda` | CUDA repo and keyring + `cuda-toolkit-<ver>` |
| cuDNN | `cudnn` | CUDA repo and keyring + cuDNN packages |
| TensorRT | `tensorrt` | CUDA repo and keyring + TensorRT packages |
| NVIDIA Docker | `docker` | `nvidia-docker` only |
| DeepStream OS deps (optional) | `deepstream-deps` | GStreamer + OS deps + OpenTelemetry — **opt-in only** |
| Everything incl. DeepStream | `all,deepstream-deps` | generic stack + the DeepStream layer |

Rules:
- **Default is the generic compute stack** (driver, CUDA, cuDNN, TensorRT, Docker). The
  DeepStream layer is DeepStream-specific and tagged `never`, so it is **excluded** unless
  explicitly requested with `--tags deepstream-deps` (or `--tags all,deepstream-deps` for everything).
- Only scope down to a single component when the operator names it (e.g. "install *just* the
  driver", "install cuDNN only").
- Fact-gathering (`init`) and the Ubuntu-24.04 assert are tagged `always` and run regardless.
- `driver`, `cudnn`, and `tensorrt` automatically pull in the shared CUDA repo and keyring setup —
  the `nvidia-open`, cuDNN, and TensorRT packages all ship from the CUDA repo — but **not**
  the full CUDA toolkit.
- To install several components, pass a comma-separated tag list (e.g. `--tags driver,docker`).

---

## 3. Version reference

Each version parameter maps to concrete apt packages and pins. The playbook derives
`cuda_major_version` from `cuda_version` and couples the cuDNN pin to it. Full mapping,
plus how to check that a requested version exists in the NVIDIA repo, is in
[`references/versions.md`](references/versions.md). Quick summary:

- `cuda_version=13-2` → `cuda-toolkit-13-2`; major `13` also drives the cuDNN pin.
- `nv_driver_version=595.45.04` → branch `595` → `nvidia-open-595` + `nvidia-driver-pinning-595.45.04`.
- `cudnn_version=9` / `cudnn_sub_version=9.20` → `libcudnn9...-cuda-13=9.20*`.
- `trt_version=10` / `trt_sub_version=10.16` → `libnvinfer10=10.16*` family.

---

## 4. Procedure — Mode A: push (remote host over SSH)

Use when the operator supplies a remote `host` with credentials.

### 4.1 Ensure Ansible on the control machine

```bash
ansible-playbook --version
```

If missing, run `sudo apt update`, then `sudo apt install -y ansible`. No sudo, or a PEP 668
"externally-managed-environment" error (default on Ubuntu 24.04)? Install without root via
`pipx install --include-deps ansible`, a venv (`python3 -m venv` - needs `python3-venv`), or
`pip install --user --break-system-packages ansible`.
**Pass:** `ansible-playbook --version` prints a version.

### 4.2 Confirm the target before provisioning

For a shared or lab host, confirm the host identity with the operator before proceeding —
provisioning reboots the machine and rewrites its GPU and display config.

### 4.3 Write the inventory (no plaintext secrets)

Create a working inventory (e.g. `/tmp/csb-inventory.ini`) with only the host and user. Do
**not** write passwords into it, and do not edit the bundled
`assets/compute-stack-provisioning/inventory.ini` in place.

```ini
[target]
target-host ansible_user=your-username

[all:vars]
ansible_python_interpreter=/usr/bin/python3
ansible_ssh_common_args='-o StrictHostKeyChecking=accept-new'
```

`accept-new` auto-accepts a first-seen host but still fails on a *changed* key (MITM
protection). If a re-imaged target trips `REMOTE HOST IDENTIFICATION HAS CHANGED`, clear its
stale key with `ssh-keygen -R <host>` rather than disabling verification;
`-o UserKnownHostsFile=/dev/null` (off entirely) is a disposable-lab fallback only.

Supply credentials **without persisting them to disk**:
- **SSH key (default):** add `--private-key <path>` on the command line.
- **SSH username/password (alternative to a key):** do **not** put the password in the
  inventory, a `-e` value, or any file. Use Ansible's interactive prompt, `--ask-pass` (short
  `-k`), so the operator (or the terminal session driving this skill) is prompted at run time;
  for unattended automation, source the password from an approved vault or secret-manager
  environment lookup instead of a literal. Password auth requires `sshpass` on the **control
  machine** (`sudo apt install -y sshpass` if `ansible-playbook` reports it missing).
- **Become password:** use `--ask-become-pass`; for unattended automation, use an
  approved vault or secret-manager-backed environment lookup.

Do not ask the operator to paste a password into chat — direct them to enter it at the
`--ask-pass`/`--ask-become-pass` prompt (or the vault flow) instead. Only stop and ask for a
different credential if neither a key nor password auth is available.

Never hard-code `ansible_ssh_pass` or `ansible_become_pass` in an inventory or commit secrets.

### 4.4 Preflight

```bash
ansible -i /tmp/csb-inventory.ini target --private-key <path> \
  -m setup -a 'filter=ansible_distribution*'
```

Password auth: swap `--private-key <path>` for `--ask-pass`.

**Pass:** connection succeeds and reports `Ubuntu` `24.04`. If the target is not Ubuntu
24.04, stop — the playbook will assert-fail (see Troubleshooting).

### 4.5 Run the playbook

```bash
ansible-playbook -i /tmp/csb-inventory.ini \
  assets/compute-stack-provisioning/ubuntu_setup.yml \
  --private-key <path> --ask-become-pass \
  -e "cuda_version=13-2" \
  -e "nv_driver_version=595.45.04"
  # ...only include -e lines for versions the operator overrode
  # Single component? append --tags <tag>, e.g. --tags driver  (see §2.1)
```

Password auth: swap `--private-key <path>` for `--ask-pass` — it prompts once per
`ansible-playbook` invocation and reuses that connection password for the whole run; it is
never written to the generated inventory or logs.

The `nv_display_driver` role **reboots the remote host** and then `wait_for_connection`s;
the play resumes automatically. Treat the mid-run disconnect + reconnect as **expected**,
not a failure. This is long-running (~10–40 min). Run it detached and poll its log rather
than blocking the session.

**Do not stop to ask the operator for input after the reboot.** The play polls for the host
itself and continues on its own; if you drive the reboot outside the playbook, poll for
SSH and host availability at a reasonable interval (e.g. every 15–30 s, for up to 9 minutes) and
resume the workflow automatically once the host answers — no operator confirmation is needed.
Stop after 9 minutes if the host does not return; report a reboot timeout and the last
successful playbook task, and do not restart provisioning automatically.

**Pass:** the play recap ends with `failed=0` for the target host.
If the recap reports `failed>0` or `unreachable>0`, stop, report the failed task and recap,
and troubleshoot the specific error before rerunning. Never emit a successful final report.

---

## 5. Hard rules

1. **Only override what the operator specified.** Pass `-e` for a version only when the
   operator named it; otherwise let the playbook defaults apply. *Why: silent version
   drift produces a stack the operator did not ask for.*
2. **Never invent version strings.** Use the operator's values verbatim; if unsure a
   version exists, check the NVIDIA repo (see `references/versions.md`) rather than guess.
   *Why: a nonexistent apt pin fails the run partway through.*
3. **Ubuntu 24.04 only; never Jetson.** Do not target other distros or releases, or Jetson —
   the playbook asserts and fails fast. Redirect Jetson requests to SDK Manager (JetPack).
4. **Confirm shared-host identity before provisioning.** Provisioning reboots the host and
   rewrites GPU and display config — get operator approval before touching a shared or lab host.
5. **Treat the driver reboot as expected, and never pause for input on it.** In push mode the
   mid-run disconnect + reconnect is normal; do not abort or re-run from scratch on seeing it,
   and do not ask the operator to confirm the reboot or that the host is back. Poll for host
   availability at a reasonable interval and resume the workflow automatically once it answers.
6. **Do not edit the bundled playbook.** Parameterize via `-e` and a generated inventory;
   the `assets/compute-stack-provisioning/` tree is the shipped source of truth.
7. **Never persist an SSH or become password.** Password auth is collected only via
   `--ask-pass`/`--ask-become-pass` (or a vault/secret-manager lookup) — never write a
   password into the inventory, a `-e` value, a command echoed to a log, or chat.
   *Why: an inventory or transcript containing a plaintext password is a standing credential
   leak.*

---

## 6. Troubleshooting

If a run fails or an operator reports an error, consult
[`references/troubleshooting.md`](references/troubleshooting.md) for the Error → Cause →
Solution table (assertion failures, SSH/sudo auth, apt/CUDA-repo errors, `sshpass`, driver
reboot/verify issues) before improvising a fix.

---

## 7. Final report

Emit a fixed-width summary at the end:

```
=== Compute Stack Build Complete ===
Host        : <host or localhost>
Arch        : <x86_64 | aarch64 (sbsa)>
Mode        : <push | on-target>
Driver      : <nv_driver_version>        [installed | FAILED | not installed]
CUDA        : cuda-toolkit-<cuda_version> [installed | FAILED | not installed]
cuDNN       : <cudnn_sub_version>          [installed | FAILED | not installed]
TensorRT    : <trt_sub_version>            [installed | FAILED | not installed]
NVIDIA Docker: <version>                   [installed | FAILED | not installed]
DeepStream deps: <version | not requested>  [installed | FAILED | skipped]
Reboot      : <performed | required | skipped (localhost)>
Verify      : nvidia-smi <PASS or FAIL> | nvcc <PASS or FAIL> | docker-gpu <PASS or FAIL>
Next step   : install your framework or SDK on top (e.g. `deepstream-setup` for DeepStream)
=====================================
```

For a single-component run, mark the components that were not part of the run as
`not installed` (only report status for what was actually installed and verified).

---

## Limitations

- **Ubuntu 24.04 only; no Jetson.** Other distros, releases, and all Jetson/JetPack targets
  are out of scope — the playbook asserts and fails fast (§1, hard rule 3). Use NVIDIA SDK
  Manager / JetPack for Jetson.
- **x86_64 and ARM SBSA only.** No other CPU architectures are supported.
- **Installs the NVIDIA open kernel driver only** — not the proprietary driver module, and
  not a specific vendor's custom driver build.
- **No uninstall, downgrade, or config-drift repair.** The playbook installs/upgrades to the
  pinned versions; it does not remove packages, roll back to an older version, or reconcile
  a host that was hand-modified outside this skill.
- **Requires outbound internet on the target** to the NVIDIA apt repos (plus
  `nvidia.github.io` and GitHub only when the opt-in `deepstream-deps` layer is requested).
  Air-gapped targets are not supported.
- **Push mode reboots the target automatically; on-target/localhost does not.** A localhost
  run leaves the driver unloaded until a manual reboot; see [`references/provisioning.md`](references/provisioning.md).
- **Password-based SSH auth needs `sshpass` on the control machine**; without it, `--ask-pass`
  fails (§6).
- **No component-removal or partial-rollback path.** `--tags` controls what gets *installed*
  in a given run; it cannot be used to remove a previously-installed component.
