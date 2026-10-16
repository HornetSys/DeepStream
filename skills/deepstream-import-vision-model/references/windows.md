<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running on Windows (and cross-platform)

This skill runs on **Windows with Docker Desktop** with **no host packages**: every compute step runs
**inside one named DeepStream Linux campaign container**. The host only needs `docker` + the NVIDIA
driver. The single exception is the host-side install (copying the skill into `<project>\.claude\skills\`),
which cannot run in a container — so the skill ships exactly one PowerShell script, **`install.ps1`**,
the twin of `install.sh`. There are no `.ps1` duplicates of the compute scripts.

## Why this works
The ONNX export, TensorRT engine build, custom nvinfer parser compile, DeepStream run, and PDF
report all execute **inside** `nvcr.io/nvidia/deepstream:9.1-triton-multiarch` (the venv
`build/.venv_optimum` + `wkhtmltopdf` are installed into that container by `setup.sh`). The container
is the portability layer. The only per-shell difference is the `docker create` **bind-mount token**.

## Prerequisites (Windows)
1. **Docker Desktop** with the **WSL2 backend enabled** (Settings → General → *Use the WSL 2 based
   engine*) — required for GPU (`--gpus all` works only through the WSL2 backend).
2. A recent **NVIDIA driver** with WSL/CUDA support. No CUDA toolkit / TensorRT / DeepStream needed
   on the host — the container ships them.
3. Docker Desktop → Settings → Resources → **File Sharing**: share the drive holding your working dir.
4. A verified immutable DeepStream repository digest supplied as `DS_IMAGE`.

## The one thing that differs per shell: the mount token
Claude Code fills this in based on the host OS:

| Shell | working-dir mount |
|-------|-------------------|
| **PowerShell** | `-v "${PWD}:/work"` |
| **cmd** | `-v "%cd%:/work"` |
| **WSL2 / Linux bash** | `-v "$PWD":/work` |

## Bootstrap + preflight (PowerShell example)
```powershell
$env:DS_IMAGE = "nvcr.io/nvidia/deepstream@sha256:<verified-64-hex-digest>"
$env:CAMPAIGN_ID = "vision-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$container = "ds-import-$env:CAMPAIGN_ID"
docker pull $env:DS_IMAGE
docker create --name $container --gpus all --shm-size=16g -v "${PWD}:/work" -w /work `
  --entrypoint sleep $env:DS_IMAGE infinity
docker start $container
$env:DS_CONTAINER = $container
docker exec $container .claude/skills/deepstream-import-vision-model/setup.sh
docker exec $container .claude/skills/deepstream-import-vision-model/scripts/preflight.sh
```
Set `DS_CONTAINER` to `$container`; every later phase uses `docker exec` directly
or `.claude/skills/deepstream-import-vision-model/scripts/dsrun.sh`.

## Notes
- The skill ships a `.gitattributes` forcing **LF** on all scripts, so a Windows checkout won't
  CRLF-corrupt them (CRLF breaks bash-in-container).
- `--shm-size=16g` works on the WSL2 backend.
- **Install:** on native Windows run the bundled **`install.ps1`** — the twin of `install.sh`, same
  sequence and flags (`-Target`=`--target`, `-NoCursor`=`--no-cursor`, `-DryRun`=`--dry-run`):
  `.\install.ps1 -Target C:\path\to\project` (copies the skill into `<project>\.claude\skills\`). On
  Linux/WSL2/Git Bash use `bash install.sh --target <project>`.
- Prefer a **WSL2 Ubuntu terminal** for the exact Linux experience — inside WSL2 everything runs
  unchanged.
