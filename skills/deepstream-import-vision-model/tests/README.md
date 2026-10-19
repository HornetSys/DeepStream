<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# deepstream-import-vision-model tests

Unit / validation tests for the hardened scripts in this skill. These run
locally with only `python3` (stdlib + `bash`) — no GPU, no network, no external
package install required.

## What is covered

| Test class | Script | What it verifies |
|---|---|---|
| `TestInstallScript` | `install.sh` | `--target` rejects `/`, `..`, empty, missing; dry-run previews the self-contained skill copy without writing to target |
| `TestCleanupScript` | `scripts/model/cleanup.sh` | `MODEL_NAME` regex enforcement; shell meta-chars / slashes rejected; dry-run does not touch real files |
| `TestHFScripts` | `scripts/model/hf-list-files.sh`, `hf-download-config.sh` | `HF_ORG` / `MODEL_NAME` / `DEST` validation |
| `TestNGCScripts` | `scripts/model/ngc-list-files.sh`, `ngc-download.sh` | NGC arg validation; `DEST_DIR` refuses `""`, `/`, path-traversal |
| `TestKittiDumpUsage` | `scripts/deepstream/ds-kitti-dump.sh` | Usage message printed when required args missing |
| `TestEmbedImages` | `scripts/report/md-to-html-pdf.py` | Local images inlined as `data:` URIs; remote / absolute / traversal paths left alone (proves `--enable-local-file-access` is safe to drop) |
| `TestPowerShellInstaller` | Vision skill `install.ps1` | Exact installed file-set match, Claude/Codex targets, `-NoCursor`, safe reinstall, missing entrypoint guard |

### Note on `TestPowerShellInstaller`

This one exists because of a shipped bug: `install.ps1` used
`Copy-Item -LiteralPath $SkillDir -Destination $dest -Recurse`, whose behaviour depends on whether
the destination already exists, and which on **Windows PowerShell 5.1** created the directory tree
without copying any files — an empty skill dir, no `SKILL.md`, and a skill the agent runtime
silently refused to load. Nothing caught it because the suites only ever exercised `install.sh`.

It needs a PowerShell, discovered in this order: `pwsh`, `powershell`, then the
`mcr.microsoft.com/powershell` docker image if already pulled. With none available it **skips** —
it will never fail on a machine without PowerShell. To enable it on a Linux dev box:

```bash
docker pull mcr.microsoft.com/powershell:latest
```

Caveat: the dev-box/container runner is PowerShell **7**. The original bug was 5.1-specific, so
this pins the copy *contract* rather than reproducing every 5.1 quirk — run it on Windows for full
fidelity.

## Run

From the repository root:

```bash
python3 -m unittest discover -s skills/deepstream-import-vision-model/tests -v
```

Or a single test file:

```bash
python3 -m unittest discover -s skills/deepstream-import-vision-model/tests -p test_hardened_scripts.py -v
```

## Adding tests

Tests shell out to the real scripts and assert on exit codes + stderr/stdout,
or import Python helpers directly. Keep them hermetic: use `tempfile.TemporaryDirectory()`,
don't rely on network access, and don't shell out to tools that may not be
installed on every dev machine (e.g. `trtexec`, `deepstream-app`, `wkhtmltopdf`,
`mmdc`, `ngc`).
