---
name: deepstream-inspect-stack
description: Use when diagnosing NVIDIA DeepStream SDK installation, version compatibility, container tag, JetPack, CUDA, TensorRT, driver, GPU/Jetson, WSL2, pyservicemaker, or pyds issues; also use to reroute DeepStream pipeline-building or model-import requests to sibling skills without producing code.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA DeepStream Team <deepstream-sdk@nvidia.com>
  version: "1.0.0"
  status: public-ready
  team: deepstream-sdk
  domain: deepstream
  languages:
    - en
  tags:
    - deepstream
    - compatibility
    - diagnostics
    - jetson
    - containers
    - versioning
owner: NVIDIA CORPORATION
service: deepstream-sdk
version: "1.0.0"
reviewed: "2026-07-21"
---

# DeepStream Inspect Stack

Diagnose DeepStream installation and version-compatibility problems before debugging starts. Return a risk level, the most likely mismatch, verification commands, a known-good setup, and one concrete next step.

## Purpose

Use this skill for DeepStream setup sanity checks, upgrade or downgrade questions, runtime errors that look version-related, container tag confusion, Jetson or GPU support questions, WSL2 path questions, and Python binding import failures.

This is a triage skill only. It must not build pipelines, author configs, import models, convert ONNX/TLT/SafeTensors, run benchmarks, or generate implementation code.

## Public Source Rule

Always read the relevant reference before answering. Do not cite DeepStream, JetPack, CUDA, TensorRT, driver, Python binding, or container-tag pairings from memory.

All compatibility claims in this skill must be based on public NVIDIA documentation, the public NVIDIA NGC catalog, or public NVIDIA repositories/forums explicitly linked in the references. If a row is missing or the user's stack is newer than the public matrix, say "unverified - check NVIDIA release notes" and link the official public documentation.

## Modes

Default to Mode A, client-mediated diagnosis. Ask the user to run verify commands and paste output unless the user clearly opts into local autonomous detection with phrases such as "auto-detect", "autonomous mode", "I'm on the box", "just run it", or `mode=autonomous`.

Use Mode B only when the current environment is the target DeepStream machine and the user opted in. In Mode B, announce each read-only command before running it, show concise output, parse the eight intake fields, and never install, remove, write config, pull containers, SSH to boards, or change system state. Docker use in Mode B is limited to `docker ps` for observing running container image names; do not run `docker exec`, `docker run`, `docker pull`, `docker rm`, `docker rmi`, or `docker inspect` autonomously. Python use in Mode B is limited to `python3 --version`; DeepStream app use is limited to `deepstream-app --version-all`. Never autonomously run Python import probes: all `python3 -c "import ..."` commands in the references are Mode A, client-mediated checks and require pasted user output.

Collect only missing intake fields, preferably in one batched request:

| Field | Verify command |
| --- | --- |
| Platform: Jetson model or discrete GPU | Jetson: `cat /proc/device-tree/model`; discrete: `nvidia-smi --query-gpu=name --format=csv,noheader` |
| DeepStream SDK version | `deepstream-app --version-all` or `cat /opt/nvidia/deepstream/deepstream/version` |
| JetPack/L4T version for Jetson | `cat /etc/nv_tegra_release`; `dpkg -l \| grep nvidia-l4t-core` |
| NVIDIA driver for discrete GPU | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| CUDA toolkit | `nvcc --version` |
| TensorRT | `dpkg -l \| grep -i tensorrt \| head` |
| Python and venv status | `python3 --version`; `echo $VIRTUAL_ENV` |
| Container vs bare metal | `docker ps --format '{{.Image}}'`; `uname -m` |

Classify risk exactly:

| Risk | Use when |
| --- | --- |
| `BLOCKED` | Known-impossible stack or unsupported native platform, such as native Windows or a GPU/Jetson family first supported after the reported DeepStream line. |
| `HIGH` | Known-incompatible component, such as driver too old for the DS row or wrong container architecture. |
| `MEDIUM` | Unverified combination, newer companion stack than the matrix row, or missing data that blocks certainty. |
| `LOW` | Matches a public known-good row or known non-version setup issue such as a missing pyservicemaker wheel in a venv. |

## Scripts

No deterministic helper scripts are bundled. Do not call `run_script(...)` for this skill; use the reference docs and read-only shell commands only when Mode B is explicitly selected.

## Examples

```text
My RTX Pro 6000 Blackwell workstation has DeepStream 7.0 and nvinfer fails. What is wrong?
```

```text
I am on a Jetson Thor box. Auto-detect what is installed and tell me if the DeepStream stack is consistent.
```

```text
Inside a DS 9.x container, `from pyservicemaker import Pipeline` fails in my Python venv.
```

```text
I pulled a DeepStream base container, but the samples folder is empty.
```

## Instructions

1. If the request asks for pipeline construction, GStreamer or pyservicemaker code, nvinfer config authoring, model import, ONNX/TensorRT conversion, or benchmarking implementation, answer only with the sibling-skill reroute:
   - Pipeline/config/code requests: "That's pipeline construction - `deepstream-dev` is the right skill. Activate it and re-ask."
   - Model import/conversion requests: "That's model onboarding - `deepstream-import-vision-model` is the right skill. Activate it with the model URL."
2. Otherwise, read the relevant reference:
   - `references/compatibility-matrix.md` for public version stack and platform support.
   - `references/known-good-setups.md` for recommended public target stacks.
   - `references/container-tags.md` for NGC tag suffixes, multi-arch behavior, and DGX Spark SBSA tags.
   - `references/python-bindings.md` for `pyservicemaker`, `pyds`, venv, and `libpython3.12` import failures.
   - `references/verify-commands.md` for copy-paste intake commands.
   - `references/troubleshooting.md` for specific error-message mapping.
3. If a needed reference cannot be read, do not diagnose from memory. Say that the matrix is unavailable and ask for pasted verified ground truth.
4. Compare the provided or detected stack against the matrix. Explicitly mark rows as NVIDIA-sourced, best-effort, or unverified when the reference does.
5. Return the five-section report. If key fields are missing, ask for those fields before producing a verdict unless the user only needs a reroute.

## Report Shape

Use exactly these section headers for diagnostics:

````markdown
**Risk:** LOW | MEDIUM | HIGH | BLOCKED

**Most-likely mismatch:** <one-line cause, citing the relevant reference row>

**Verify (run these and paste output):**
```bash
<command 1>
<command 2>
```

**Known-good setup for your platform:**
<specific row or closest supported path, including container tag when relevant>

**Next step:**
<one concrete action>
````

For Mode B, change the verify header to `**Verify (already ran in autonomous mode):**` and summarize the commands already shown.

## Quality Bar

- Never invent compatibility claims or unstated version pairings.
- Never produce DeepStream implementation code under any framing, including "just an example" or "since the sibling skill is unavailable."
- Do not recommend native Windows DeepStream. Native Windows is `BLOCKED`; WSL2 plus Ubuntu plus NGC container is the supported path to investigate.
- Do not infer a Jetson board from L4T alone; use `cat /proc/device-tree/model` or GPU name.
- Keep commands pasteable and label whether they are user-run or autonomously run.
- Never include credentials, API keys, RTSP URLs, private board IPs, reservation-sheet content, non-public registry URLs, or unpublished build identifiers in output or committed files.
- Treat `BENCHMARK.md` as NVCARPS-generated evaluation evidence; never edit it manually.
