# Known-Good Setups by Platform

Use this file when the user wants a concrete public recommendation: DeepStream version, OS, container tag, driver, Python, and next step.

Authoritative public sources:

- DeepStream installation guide: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html
- NGC DeepStream catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream
- Public DeepStream coding-agent examples: https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent

## Discrete GPU - Blackwell

Examples: RTX Pro 6000 Blackwell, RTX 50-series, B100, B200, GB200.

Recommended current path:

| Component | Value |
| --- | --- |
| DeepStream | 9.1 |
| OS | Ubuntu 24.04 LTS |
| NVIDIA driver | 595.58.03 or newer in the same branch family |
| CUDA | 13.2 |
| TensorRT | 10.16.0.72 |
| Container | `nvcr.io/nvidia/deepstream:9.1-triton-multiarch` |
| Python API | `pyservicemaker` |
| Authority | [NVIDIA-sourced] |

Valid older public rows:

- DS 9.0 with Ubuntu 24.04, Driver 590.48.01, CUDA 13.1, TensorRT 10.14.1.48, container `nvcr.io/nvidia/deepstream:9.0-triton-multiarch`.
- DS 8.0 with Ubuntu 24.04, Driver 570.133.20, CUDA 12.8, TensorRT 10.9.0.34. DS 8.0 is the first public row in this matrix that lists Blackwell.

If a Blackwell user reports DS 7.x or earlier, return Risk: HIGH or BLOCKED and recommend DS 9.1 unless they have a release-specific reason to pin to DS 8.0.

## Discrete GPU - Hopper, Ada, Ampere, Turing

Recommended current path:

| Component | Value |
| --- | --- |
| DeepStream | 9.1 |
| OS | Ubuntu 24.04 LTS |
| NVIDIA driver | 595.58.03 or newer in the same branch family |
| CUDA | 13.2 |
| TensorRT | 10.16.0.72 |
| Container | `nvcr.io/nvidia/deepstream:9.1-triton-multiarch` |
| Authority | [NVIDIA-sourced] |

Use older release rows only when the user is constrained to an older Ubuntu, driver, or JetPack line. When recommending an older row, cite `compatibility-matrix.md` and tell the user to verify the release-specific public notes.

## Jetson Thor

Recommended current path:

| Component | Value |
| --- | --- |
| DeepStream | 9.1 |
| JetPack | 7.2 |
| L4T | 39.2 |
| OS | Ubuntu 24.04 |
| CUDA | 13.2 |
| TensorRT | 10.16.1.7 |
| cuDNN | 9.20.0.46 |
| Container | `nvcr.io/nvidia/deepstream:9.1-triton-multiarch` |
| Python API | `pyservicemaker` |
| Authority | [NVIDIA-sourced] |

Valid older public rows:

- DS 9.0 on JetPack 7.1 / L4T 38.4.
- DS 8.0 on JetPack 7.0 / L4T 38.2.

If a Thor user reports DS 7.x or earlier, return Risk: BLOCKED unless the user provides an NVIDIA release note proving support for that exact board and release.

## Jetson Orin

Recommended current path:

| Component | Value |
| --- | --- |
| DeepStream | 9.1 |
| JetPack | 7.2 |
| L4T | 39.2 |
| OS | Ubuntu 24.04 |
| CUDA | 13.2 |
| TensorRT | 10.16.1.7 |
| cuDNN | 9.20.0.46 |
| Container | `nvcr.io/nvidia/deepstream:9.1-triton-multiarch` |
| Python API | `pyservicemaker` for new DS 9.x work |
| Authority | [NVIDIA-sourced] |

For JetPack 6.x Orin systems, use the DS 7.1 or DS 7.0 rows in `compatibility-matrix.md` unless public release notes confirm a newer DeepStream pairing.

## Jetson Xavier and Nano

These are legacy platforms for new DeepStream work. Use release-specific public documentation before recommending a stack. If the user is starting a new project, recommend migrating to Orin or Thor rather than forcing a new DS 9.x workflow onto legacy Jetson hardware.

## DGX Spark

Recommended starting path:

| Component | Value |
| --- | --- |
| DeepStream | 9.1 |
| Platform | DGX Spark / GB10 Blackwell |
| Container | `nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark` |
| Authority | [NVIDIA-sourced] |

Do not recommend the generic x86_64 or Jetson multi-arch tag for DGX Spark when the SBSA/DGX Spark tag is available.

## Windows hosts

DeepStream is not supported natively on Windows.

| Path | Risk |
| --- | --- |
| Native Windows install | BLOCKED |
| WSL2 plus Ubuntu plus NVIDIA container runtime plus DeepStream NGC container | LOW after GPU/container runtime is verified |

When answering Windows questions, keep the distinction clear: WSL2/Linux/container is the path to investigate; native Windows is not.

## Quick decision tree

1. Blackwell discrete GPU? Recommend DS 9.1, Driver 595.58.03+, Ubuntu 24.04, `9.1-triton-multiarch`.
2. Jetson Thor or Orin on JetPack 7.2? Recommend DS 9.1 and `9.1-triton-multiarch`.
3. DGX Spark? Recommend `9.1-triton-sbsa-dgx-spark`.
4. Older JetPack or older Ubuntu? Use `compatibility-matrix.md` and mark the row as release-specific.
5. Unknown or newer-than-matrix combination? Risk: MEDIUM and ask the user to verify against public NVIDIA release notes.
