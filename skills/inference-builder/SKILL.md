---
name: inference-builder
description: |
  Generate deployable Vision AI pipelines with high-performance video and
  streaming capabilities on NVIDIA GPUs. Activate when users want to: create
  GPU-accelerated inference microservices or standalone apps for vision,
  video, or streaming workloads; write or edit pipeline YAML configs; build
  Docker images for GPU inference; work with models from NGC or HuggingFace;
  or deploy with DeepStream, Triton, vLLM, TensorRT-LLM, or PyTorch backends.
owner: "chunlinl@nvidia.com"
metadata:
  author: "Chunlin Li <chunlinl@nvidia.com>"
  version: "1.0.0"
  tags:
    - inference
    - nvidia
    - deepstream
    - vllm
    - tensorrt-llm
    - triton
    - docker
    - gpu
service: "inference"
reviewed: "2026-04-09"
---

# NVIDIA Inference Builder

A code generator for Vision AI pipelines with high-performance video and
streaming capabilities on NVIDIA GPUs. It turns a YAML config (+ optional
OpenAPI spec and custom processors) into a deployable inference pipeline,
packaged as a Docker container or standalone application.

## Setup

Inference Builder is part of the DeepStream mono-repo. Resolve it from the
repository root so the skill works from any subdirectory:

```bash
DEEPSTREAM_ROOT="$(git rev-parse --show-toplevel)"
INFERENCE_BUILDER_ROOT="$DEEPSTREAM_ROOT/tools/inference_builder"
source "$INFERENCE_BUILDER_ROOT/.venv/bin/activate"
python "$INFERENCE_BUILDER_ROOT/builder/main.py" --help
```

### First-time venv setup

```bash
cd "$INFERENCE_BUILDER_ROOT"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Getting current CLI usage

Always run with `--help` for authoritative flags — do NOT rely on memorized
syntax:

```bash
python "$INFERENCE_BUILDER_ROOT/builder/main.py" --help
```

All paths below are relative to `tools/inference_builder/` unless stated otherwise.

## Key Directories

| Path | What lives there |
|------|-----------------|
| `builder/samples/` | Working examples for every backend — read `builder/samples/README.md` for the full index, then start from the closest matching example |
| `schemas/config.schema.json` | Main config schema — all fields, data types, and validation rules |
| `schemas/README.md` | Comprehensive reference: backends, parameters, data types, preprocessors, responders, routing |
| `schemas/backends/` | Per-backend schemas and parameter definitions |
| `templates/backend/` | Jinja templates for each inference backend |
| `templates/responder/` | Jinja templates for server endpoint handlers |
| `lib/` | Shared runtime libraries included in generated pipelines |
| `builder/tests/` | Integration test framework (see `run_tests.sh --help`) |

## Instructions

1. **Browse samples & schemas** — find the closest match in `builder/samples/`;
   consult `schemas/config.schema.json` and `schemas/backends/` for field reference
2. **Write YAML config** — validate against the schema; consult
   `schemas/README.md` for data types, preprocessors, responders, and routing.
   Prefer choices that keep the hot path on GPU and skip avoidable GPU↔CPU
   round-trips; prefer GPU acceleration across the whole pipeline — decode/
   encode, scale & color convert, batch, preprocess, infer, track, and overlay
3. **Generate pipeline** — `python builder/main.py <config.yaml> [options]`
4. **Prepare model repository** — download models from NGC/HuggingFace into
   the model_repo directory
5. **Generate nvinfer config** (DeepStream only) — check if the downloaded
   model already includes runtime config files (nvdsinfer_config.yaml,
   nvdspreprocess_config.yaml); if not, generate them and place alongside
   the model. Skip this step entirely for non-DeepStream backends
6. **Select platform & build Docker image** — read `doc/platform-guide.md` to
   choose the correct base image and Dockerfile for the target hardware
   (x86_64, Jetson/Tegra, DGX Spark); then build the image
7. **Run & test** — run the container with the model repository mounted;
   for serverless pipelines the container runs to completion; for HTTP
   servers (fastapi/triton/nim) send test requests to verify. If GPU
   resources are available, perform an end-to-end smoke test against the
   running container to validate the full pipeline on real hardware

## Non-Obvious Knowledge

- If the user's model request is ambiguous (e.g., unclear model variant,
  version, precision, or source), always ask for clarification before
  proceeding — do not guess.
- Before using any model, gather as much information as possible about it:
  version, architecture, input/output shapes, precision, and expected
  preprocessing. Cross-check this information against the pipeline YAML
  config and any runtime config (e.g., nvdsinfer_config.yaml) to ensure
  consistency — mismatches in input dims, scale factors, or label files
  are common sources of silent failures.
- Config YAML supports runtime env-var substitution: `$ENV_VAR|default_value`
  (auto-typed based on the default).
- Server type `serverless` does NOT need an OpenAPI spec or server definition.
- Custom processor class `name` attribute must exactly match the processor
  name in the YAML config.
- Default to bridge networking plus explicit `-p` port mappings. Recommend
  Docker host-network or host-IPC modes only when the deployment explicitly
  needs host namespace sharing, and call out the security tradeoff before
  opting in.
- Before running containers, check `nvidia-smi` for free GPU memory. If GPUs
  are occupied or low on memory, select a free GPU with `--gpus device=N` or
  ask the user to free resources — OOM failures can be silent or misleading.
- Key runtime env vars: `HTTP_PORT` (default 8000), `LOG_LEVEL` (default INFO),
  `DEBUG` (0/1), `N_CODEC_INSTANCES` (parallel video decoders),
  `MAX_BATCH_SIZE` (live stream frame batching).
- DeepStream backends require a separate `nvdsinfer_config.yaml` in the
  model directory.
- When writing new configs, the required top-level fields are only `name`,
  `model_repo`, and `models` — everything else is optional.

## Examples

- "Generate an inference-builder YAML config for a DeepStream detection service
  that reads RTSP, uses Triton, and exposes a FastAPI endpoint."
- "Adapt the dummy sample into a serverless image-inference pipeline and show
  the Docker build and run commands."
