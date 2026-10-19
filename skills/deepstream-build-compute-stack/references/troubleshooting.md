<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

Error → cause → solution for `deepstream-build-compute-stack`. See `SKILL.md` §6 for when to consult
this file.

| Error | Cause | Solution |
|-------|-------|----------|
| Supported-host assertion fails its OS check | Target is not Ubuntu 24.04 | Only run on Ubuntu 24.04; for Jetson use SDK Manager |
| Supported-host assertion fails its architecture check | Target architecture is not x86_64 or ARM SBSA | Use an x86_64 or aarch64 SBSA host; for Jetson use SDK Manager |
| Supported-host assertion fails on a Jetson device | `nvidia-l4t-core` package detected on an `aarch64` host | Jetson is out of scope regardless of Ubuntu version; use NVIDIA SDK Manager / JetPack |
| Host does not reconnect after driver reboot | Reboot or SSH recovery exceeded 9 minutes | Stop and report the timeout; inspect the host out of band before rerunning |
| Play recap reports `failed>0` or `unreachable>0` | A provisioning task or target connection failed | Report the failed task and recap, resolve that cause, then rerun |
| `Permission denied` / SSH auth fails | Wrong user, key, or key permissions (key auth); wrong username/password (password auth) | Key auth: verify the user and private-key path, and that the key file is readable only by the operator. Password auth: re-prompt for the correct username/password via `--ask-pass` |
| `to use the 'ssh' connection type with passwords, you must install the sshpass program` | `--ask-pass`/`--ask-become-pass` used without `sshpass` on the control machine | `sudo apt install -y sshpass`, then rerun |
| CUDA keyring download or checksum fails | Arch mismatch or wrong `ubuntu_release_token` | Confirm arch (`aarch64` → `sbsa`) and that the repo path `ubuntu2404` exists for it |
| `E: Version '...' not found` on cuDNN or TensorRT | Requested `*_sub_version` doesn't exist in the repo for that CUDA major | Pick a version that exists (see `references/versions.md`); check the NVIDIA apt repo |
| OpenTelemetry build fails | Missing build deps / clone failure | Ensure `git`, `cmake`, `build-essential`, internet; re-run (stamp files skip completed steps) |
| `docker: permission denied` after install | `docker` group membership not yet active | Log out and back in (or `newgrp docker`), or use `sudo docker` |
| `nvidia-smi` fails right after install | Host has not rebooted (localhost, or driver unchanged) | Reboot the host, then re-verify |
