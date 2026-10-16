<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NVIDIA Compute Stack Provisioning
Ansible playbooks for provisioning Ubuntu hosts with the NVIDIA compute stack — NVIDIA driver, CUDA, cuDNN, TensorRT, and NVIDIA Docker — plus optional DeepStream OS dependencies. Supported platforms: **x86_64** & **ARM SBSA**. For Jetson configuration, use [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) with a supported JetPack version.

### Setup Details

Packages | Status | Versions (default values shown; all configurable)
 --- | --- | ---
OS Supported | **Ubuntu-24.04** | -
NVIDIA Device Driver (Open) | *CONFIGURABLE* | **nv_driver_version=** 595.45.04
CUDA | *CONFIGURABLE* | **cuda_version=** 13-2
CuDNN    | *CONFIGURABLE* | **cudnn_version=** 9  **cudnn_sub_version=** 9.20
TensorRT | *CONFIGURABLE* | **trt_version=** 10  **trt_sub_version=** 10.16
Nvidia Docker | **ADDED** | -
DeepStream OS deps | **OPTIONAL** | opt-in via `--tags deepstream-deps`

## Safety warning

This playbook runs with elevated privileges, changes APT repositories and system packages,
may replace an existing NVIDIA driver, and can reboot remote hosts. Run it on a test host
or take a recoverable snapshot or backup first. Expect service and graphical-session
interruption when driver packages or display configuration change. The driver role stops
Xorg sessions and regenerates `/etc/X11/xorg.conf`; restore the file from the host backup
to roll back. The Docker role adds the connecting user to the `docker` group, which grants
root-equivalent host access, and restarts the Docker daemon, which may interrupt running
containers.

## Running the Playbook

### Prerequisites

Ansible on the control machine, SSH access to an Ubuntu 24.04 target, and `sudo` on that
target. See the skill README (`../../README.md`) for control-machine install fallbacks and
credential handling.

### Local Installation
1. Clone this repository:
   ```bash
   git clone <REPO_URL>
   cd <REPO_DIR>/skills/deepstream-build-compute-stack/assets/compute-stack-provisioning
   ```

2. Create an inventory file (e.g., `inventory.ini`):
   ```ini
   [target]
   your-server-ip ansible_user=your-username
   ```

3. Run the playbook:
   ```bash
   ansible-playbook -i inventory.ini ubuntu_setup.yml
   ```

   Note: Adjust the version numbers according to your needs. The values shown above are examples.

### Available Variables

The supported variables and current defaults are defined in `defaults/main.yml`. Override only
the values needed for a run with `-e variable=value`; for example, `-e cuda_version=13-2`.

### Basic Usage

#### Local System Configuration
The repository includes a default inventory file (`inventory.ini`) configured for local system setup. To configure your local system:

```bash
ansible-playbook -i inventory.ini ubuntu_setup.yml
```

This will use all default values and configure the local system. The playbook will run with elevated privileges (sudo) as required.

#### Custom Configuration
To override specific values while configuring local system:
```bash
ansible-playbook -i inventory.ini ubuntu_setup.yml \
  -e "cuda_version=13-2" \
  -e "nv_driver_version=595.45.04"
```

#### Installing Individual Components
By default the playbook installs the generic compute stack (driver + CUDA + cuDNN + TensorRT +
NVIDIA Docker). To install a single component, add `--tags <component>`; combine tags with a
comma. For the full component → tag mapping and the scope rules, see `SKILL.md` §2.1.

```bash
# just the driver:
ansible-playbook -i inventory.ini ubuntu_setup.yml --tags driver
# several components at once:
ansible-playbook -i inventory.ini ubuntu_setup.yml --tags driver,docker
# add the DeepStream OS deps (opt-in):
ansible-playbook -i inventory.ini ubuntu_setup.yml --tags deepstream-deps
```

#### Remote System Configuration
For configuring remote systems, modify the inventory.ini file with your target host details:
```ini
[target]
your-server-ip ansible_user=your-username
```
