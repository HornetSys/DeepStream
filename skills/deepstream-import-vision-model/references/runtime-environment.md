<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

## Container runtime and preflight

Run every phase in the same digest-pinned DeepStream image; the host needs only
Docker, NVIDIA Container Toolkit, and the driver. The caller supplies the
verified repository digest as `DS_IMAGE`; validate its `@sha256:` form and
record it, but do not resolve a mutable tag on the caller's behalf. Keep artifacts under the mounted
work root. See [windows.md](windows.md) for
PowerShell/cmd mount syntax. Use only least-privilege source credentials and
never print or place them in command arguments.

For a gated Hugging Face source, provide `HF_TOKEN` only as a masked secret
environment variable inherited by the persistent container (for example, add
`--env HF_TOKEN` to `docker create` after exporting it in the caller environment).
Never append its value to the command line, URL, logs, or generated files, and
disable shell tracing while it is in scope. The `hf-*` scripts consume
`HF_TOKEN` from the environment and send it only in the Authorization header.

Start one execution container with one selected NVIDIA GPU and keep it alive for setup, preflight, build,
runtime, and reporting so setup-installed system packages remain available:

```bash
: "${DS_IMAGE:?set DS_IMAGE to verified nvcr.io/nvidia/deepstream@sha256:<64-hex-digest>}"
DS_DIGEST="${DS_IMAGE##*@sha256:}"
case "$DS_IMAGE" in nvcr.io/nvidia/deepstream@sha256:*) ;; *) echo "BLOCKED: invalid DS_IMAGE repository" >&2; exit 4 ;; esac
case "$DS_DIGEST" in ""|*[!0-9a-f]*) echo "BLOCKED: invalid DS_IMAGE digest" >&2; exit 4 ;; esac
[ "${#DS_DIGEST}" -eq 64 ] || { echo "BLOCKED: invalid DS_IMAGE digest length" >&2; exit 4; }
docker pull "$DS_IMAGE"
CAMPAIGN_ID="${CAMPAIGN_ID:-vision-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
case "$CAMPAIGN_ID" in ""|*[!A-Za-z0-9_.-]*) echo "BLOCKED: invalid CAMPAIGN_ID" >&2; exit 4 ;; esac
export DS_CONTAINER="ds-import-$CAMPAIGN_ID"
DS_GPU_DEVICE="${DS_GPU_DEVICE:-0}"
if [[ ! "$DS_GPU_DEVICE" =~ ^([0-9]+|GPU-[0-9a-fA-F-]{36}|MIG-GPU-[0-9a-fA-F-]{36}/[0-9]+/[0-9]+)$ ]]; then
  echo "BLOCKED: DS_GPU_DEVICE must select exactly one GPU index, GPU UUID, or MIG device" >&2
  exit 4
fi
docker create --name "$DS_CONTAINER" --gpus "device=$DS_GPU_DEVICE" --shm-size=16g \
  -v "$PWD":/work -w /work --entrypoint sleep "$DS_IMAGE" infinity
docker start "$DS_CONTAINER"
docker exec "$DS_CONTAINER" .claude/skills/deepstream-import-vision-model/setup.sh
docker exec "$DS_CONTAINER" .claude/skills/deepstream-import-vision-model/scripts/preflight.sh
```

The selected device is GPU `0` inside this one-GPU container, so generated
`nvinfer` configs retain `gpu-id=0`. Do not infer that a four-GPU host or
multi-GPU container is required. A multi-stream benchmark measures multiple
logical video streams on this selected GPU; it does not require multiple GPUs.
Hardware capability is checked at preflight/build/runtime: CUDA-incompatible
drivers, unsupported model operators, insufficient memory, or unavailable
encoders are reported per case, not generalized into a claim that every model
or NVIDIA device is supported.

Use `docker exec "$DS_CONTAINER" ...` for every phase. The named container
survives shell exits; if it stops, restart it and rerun setup and preflight.
After evidence is complete, stop and remove only this campaign container.

`setup.sh` creates a shared venv with system GStreamer bindings and, when
needed, invokes the image-provided PyDS installer before bridging its
site-packages into that same venv. `preflight.sh` imports GI, Gst, PyDS, and
Pillow from `build/.venv_optimum` and initializes a CUDA context, catching
host-driver/container-runtime incompatibility before any engine build. This is
a hard gate for the classification
reference runner. If it fails, return `BLOCKED` with the preflight error rather
than running the runner and emitting a Python traceback.

For a native C++ detection or segmentation **diagnostic pipeline only**, set
`DS_PREFLIGHT_PROFILE=native-pipeline` when running `preflight.sh`. It still
checks the selected GPU, CUDA context, TensorRT, encoder, and source video, but
defers Python/GI/PyDS and PDF checks that the native runner does not use. This
profile cannot authorize adapter promotion or Step 8 completion; rerun the
default `full` profile after provisioning those dependencies. Never use the
native profile for a Python/Service Maker runner.

Use `build/.venv_optimum/bin/python` and `/usr/src/tensorrt/bin/trtexec` in that
container. `scripts/dsrun.sh` requires `DS_CONTAINER` and executes only in that
named campaign container. Build and run TensorRT there. Preflight failure stops.

## Encoder fallback

1. Use `nvv4l2h264enc` (NVENC) for `.mp4`. If unavailable, use
   `theoraenc + oggmux` for `.ogv` from `gstreamer1.0-plugins-base`;
   `x264enc` and `openh264enc` are prohibited.
2. If the fallback is absent, rerun `setup.sh` exactly once inside the existing
   persistent execution container and repeat preflight. Its apt-installed
   packages persist; never use a fresh `--rm` container for this retry.
3. If there is no persistent container or no encoder after that retry, emit
   `DS_SINGLE_STREAM_MODE=skipped`, return encoder exit `4`
   (`ENCODER_EXIT_4`), mark pipeline `BLOCKED` and visual `UNTESTED`, and stop.
4. A skipped run cannot be `PASS`; use an NVENC-capable system if needed.
