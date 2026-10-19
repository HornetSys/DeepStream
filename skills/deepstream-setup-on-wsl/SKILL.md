---
name: deepstream-setup-on-wsl
description: >
  Set up and verify NVIDIA DeepStream in Docker inside WSL2 on a local or remote
  Windows machine. Use when a user wants to install WSL2, Ubuntu, Docker, NVIDIA
  Container Toolkit, and a requested DeepStream image; validate a supported WDDM
  workstation GPU; or run built-in or user-supplied DeepStream pipelines.
service: deepstream
reviewed: 2026-07-30
license: CC-BY-4.0 AND Apache-2.0
permissions:
  - env
  - file_read
  - file_write
  - network
  - shell
metadata:
  tags: [deepstream, wsl2, wsl, windows, docker, gpu, setup, verification, nvidia-container-toolkit]
  languages: [bash, python]
  domain: devops
  version: "2.1.1"
  data-classification: "public"
---

# DeepStream setup on WSL2

Set up DeepStream in Docker inside WSL2, then optionally verify built-in or
user-supplied pipelines. All supported targets use the same preflight,
provisioning, verification, and report steps.

## Safety contract

- Ask before every state-changing phase. Show the exact plan before installing.
- Never ask for or accept a password, API key, or token in chat.
- For remote Windows, use WinRM HTTPS with certificate validation or SSH. Never
  use external WinRM HTTP.
- Treat the GPU check as a hard gate. Require a WDDM GeForce/RTX/Quadro
  workstation GPU; reject H100/A100/H200/L40/L4/T4/Tesla and TCC mode.
- Never install a Linux NVIDIA driver in WSL; Windows projects the driver.
- Treat host CUDA Toolkit as optional: install a versioned toolkit-only package
  only when the user requests host-side CUDA compilation.
- Prefer existing compatible components. Retry failed provisioning at most once.
- Run optional verification only after asking. Require a second confirmation for
  user-supplied commands or network access.
- Never place credentials in a custom command or CLI flag. Use an application-
  specific private file or interactive authentication flow outside chat.

## Invocation

Display:

```text
DeepStream on WSL2

Where should it run?
1. This Windows computer (no remote credentials).
2. Remote Windows (WinRM HTTPS; hidden local password prompt).
3. Existing WSL over SSH (existing key or ssh-agent).

I will preflight, show changes, and ask before installing. Never paste secrets.
```

This skill supports an existing local or remote target.

## Step 1 — Collect non-secret inputs

Ask for:

- target mode: local, remote HTTPS, or SSH;
- requested DeepStream image or version;
- installed Windows version, GPU, driver/CUDA, WSL distro, and whether host-side
  CUDA compilation is required;
- whether to run optional built-in or custom verification.

Hostnames, IP addresses, and usernames are not authentication secrets, but offer
a target profile when the user prefers to keep them out of chat.

### Remote Windows target profile

Require Python 3.10+ on the controller. Ask the user to run locally:

```bash
python scripts/configure_target.py create --name <TARGET_NAME>
```

The helper prompts for host, username, and password. Password input is hidden and
stored only in a private file; it prints `TARGET_PROFILE=<path>`. Ask the user to
reply only that the profile is ready or provide its path. Never read or display
the password yourself.

The password file must be a regular, user-owned file with mode `0600`.

## Step 2 — Read-only preflight

For WinRM mode, run:

```bash
python scripts/bootstrap.py --check
```

Check, without changing state:

- controller Python 3.10+ and the exact private dependency environment for WinRM;
- target reachability and secure transport;
- Windows 11, WDDM driver mode, GPU name, driver/CUDA, WSL version and distro;
- disk space and whether Docker, NVIDIA Container Toolkit, and the requested
  image already exist.

Stop before installation if the target, transport, or GPU gate fails.

Resolve the CUDA-compatible Windows minimum separately from the WSL guide's
validated driver; never reuse a Linux driver number. Resolve the remaining stack
from `references/version-matrix.md` and official docs. The container owns
CUDA/TensorRT; `nvidia-smi` does not prove Toolkit installation.

## Step 3 — Show and confirm the plan

Display target/connection, GPU/WDDM, distro, digest-preferred image, resolved
driver/CUDA/TensorRT stack, and exact changes: controller environment, WSL,
Ubuntu, optional host Toolkit, Docker, Container Toolkit, and image pull.

Use `AskUserQuestion` for explicit confirmation. Use separate confirmations for
firewall/WinRM configuration, driver replacement, reboot, custom commands,
network-enabled containers, and deletion.

## Step 4 — Controller prerequisites

For WinRM mode, after confirmation, run:

```bash
python scripts/bootstrap.py --install --yes
```

Use the printed `PYTHON=<path>` later. Bootstrap creates
`~/.deepstream-wsl/venv`, never modifies system Python, and leaves authentication
to the user. SSH requires an existing client; local mode needs no controller.

## Step 5 — WSL2 and Ubuntu

Follow `references/wsl2-deepstream-steps.md`. Install the standalone Microsoft WSL
package and selected Ubuntu distro only when missing or incompatible. Verify the
download's trusted source, SHA-256, and Authenticode signature before MSI
installation. Announce and confirm reboot before initiating it.

After reconnecting, run the hard gate:

```powershell
nvidia-smi --query-gpu=name,driver_model.current,driver_version --format=csv,noheader
wsl -d <DISTRO> -u root -- nvidia-smi
```

Stop if the GPU is unsupported, reports TCC, or does not project into WSL.
Stop below the CUDA-compatible minimum. Mark an older-than-validated driver
unvalidated and require container verification. Guide public replacement through
the official Windows installer; never silently replace it. Confirm reboot
separately.

## Step 6 — Provision DeepStream

For remote HTTPS:

```bash
<PYTHON> scripts/winrm_exec.py --target-profile <PROFILE> \
  run-wsl --distro <DISTRO> --timeout 3600 \
  --env DS_IMAGE=<IMAGE> --env NVIDIA_CTK_VERSION=<PINNED> \
  --env MIN_WINDOWS_DRIVER=<CUDA_MINIMUM> \
  --env VALIDATED_WINDOWS_DRIVER=<WSL_VALIDATED> \
  --env EXPECTED_CONTAINER_CUDA=<VERSION> --env EXPECTED_TENSORRT=<VERSION> \
  --env HOST_CUDA_TOOLKIT_MODE=<skip|check|install> \
  --env HOST_CUDA_TOOLKIT_VERSION=<VERSION> \
  --script scripts/wsl_provision_deepstream.sh
```

The script installs missing Docker and Container Toolkit packages, configures the
runtime, optionally installs only a versioned host Toolkit, pulls the image, and
checks its DeepStream/CUDA/TensorRT inventory. Default host Toolkit mode is
`skip`; use `install` only after explicit confirmation of host compilation.
Require the final line `PROVISION_RESULT=PASS`.
On failure, diagnose from `STEP=` and `FAIL:` lines and retry at most once.

NGC authentication, when required, is an interactive user step. Never collect an
NGC key in chat or pass it as a command argument.

## Step 7 — Optional verification

Ask separately before each option.

Built-in sanity checks:

```bash
<PYTHON> scripts/run_verification.py --target-profile <PROFILE> \
  --ds-image <IMAGE> --label <LABEL> --timeout 1800
```

For a custom pipeline, collect command/inputs/pass criteria, show the redacted
command, confirm execution and any network access separately, then pass `--yes`.

```bash
<PYTHON> scripts/run_custom_pipeline.py --target-profile <PROFILE> \
  --ds-image <IMAGE> --name <CHECK_NAME> \
  --cmd '<COMMAND>' --expect-regex '<PATTERN>' --yes \
  [--allow-network] [--input <FILE>]
```

Immediately reproduce the complete result table and local run directory in the
user-facing response; never rely on tool output alone. A PASS needs real evidence
such as frames, objects, or validated output—not EOS alone.

## Step 8 — Report and cleanup

Report target OS/GPU/driver and driver CUDA capability separately from the WSL
host Toolkit and container CUDA/TensorRT versions. Include changes, image,
verification results, and log path. Then:

- remove Windows staging files using the same secure target profile;
- stop temporary services;
- retain private verification logs only when the user wants them;
- offer to delete the target profile:

```bash
python scripts/configure_target.py delete --profile <PROFILE> --yes
```


## Resources

- `scripts/configure_target.py`, `bootstrap.py`, `winrm_exec.py` — secure target
  profile, controller environment, and remote execution.
- `scripts/wsl_provision_deepstream.sh` — driver/Toolkit/container compatibility
  gates and provisioning.
- `scripts/run_verification.py`, `run_custom_pipeline.py` — fixed or confirmed
  custom verification.
- `references/` — setup, secure WinRM, version routing, and troubleshooting.
