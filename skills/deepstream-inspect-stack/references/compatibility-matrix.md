# DeepStream Compatibility Matrix

Use this file to compare a user's reported DeepStream stack against public NVIDIA documentation.

Authoritative public sources:

- Current DeepStream documentation: https://nvidia-deepstream.docs.buildwithfern.com/deepstream/9.2
- Current DeepStream installation guide: https://nvidia-deepstream.docs.buildwithfern.com/deepstream/9.2/text/DS_Installation
- NGC DeepStream catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream
- Public DeepStream coding-agent examples: https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent

Authority labels:

- `[NVIDIA-sourced]` means the row is from public NVIDIA documentation or public NVIDIA repositories.
- `[best-effort - verify]` means the row is useful triage guidance but the user should verify it against the release-specific public notes.
- `[unverified]` means do not claim compatibility. Return Risk: MEDIUM unless another row proves the stack is impossible.

## Discrete GPU stack (x86_64)

| DS version | Ubuntu | NVIDIA driver | CUDA | TensorRT | Platforms listed in public docs | Authority |
| --- | --- | --- | --- | --- | --- | --- |
| 9.1 | 24.04 | 595.58.03 | 13.2 | 10.16.0.72 | Turing, Ampere, Hopper, Ada, Blackwell | [NVIDIA-sourced] |
| 9.0 | 24.04 | 590.48.01 | 13.1 | 10.14.1.48 | Turing, Ampere, Hopper, Ada, Blackwell | [NVIDIA-sourced] |
| 8.0 | 24.04 | 570.133.20 | 12.8 | 10.9.0.34 | Turing, Ampere, Hopper, Ada, Blackwell | [NVIDIA-sourced] |
| 7.1 | 22.04 | 535.183.06 datacenter / 560.35.03 RTX | 12.6 | 10.3.0.26 | Turing, Ampere, Hopper, Ada | [NVIDIA-sourced] |
| 7.0 and earlier | Release-specific | Release-specific | Release-specific | Release-specific | See release-specific install guide | [best-effort - verify] |

## Jetson stack (ARM64)

The Jetson companion stack is not the same as the x86_64 stack at the same DeepStream major version. Always use the Jetson row when diagnosing Jetson Thor, Orin, Xavier, or Nano.

| DS version | JetPack | L4T | Ubuntu | CUDA | TensorRT | cuDNN | Publicly listed Jetson platforms | Authority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9.1 | 7.2 | 39.2 | 24.04 | 13.2 | 10.16.1.7 | 9.20.0.46 | Jetson AGX Thor, Jetson Orin | [NVIDIA-sourced] |
| 9.0 | 7.1 | 38.4 | 24.04 | 13.0 | 10.13.2.6 | 9.12.0 | Jetson AGX Thor | [NVIDIA-sourced] |
| 8.0 | 7.0 | 38.2 | 24.04 | 13.0 | 10.13.2.6 | 9.12.0 | Jetson AGX Thor | [NVIDIA-sourced] |
| 7.1 | 6.1 | 36.4 | 22.04 | 12.6 | 10.3.0.31 | 9.3.0 | Jetson AGX Orin, Orin NX, Orin Nano | [NVIDIA-sourced] |
| 7.0 | 6.0 | 36.3 | 22.04 | 12.2 | 8.6.2.3 | 8.9.4.25 | Jetson AGX Orin, Orin NX, Orin Nano | [NVIDIA-sourced] |
| 6.4 | 6.0 DP | 36.2 | 22.04 | 12.2 | 8.6.2.3 | 8.9.4.25 | Jetson AGX Orin, Orin NX, Orin Nano | [NVIDIA-sourced] |
| 6.3 | 5.1.2 | 35.4 | 20.04 | 11.4 | 8.5.2.2 | 8.6.0.166+ | Xavier, Orin, Nano | [NVIDIA-sourced] |
| 6.2 | 5.1 | 35.2.1 | 20.04 | 11.4 | 8.5.2.2 | 8.6.0.166+ | Xavier, Orin, Orin NX | [NVIDIA-sourced] |
| 6.1.1 | 5.0.2 | 35.1 | 20.04 | 11.4 | 8.4.1.5 | 8.4.1.50+ | Xavier, Jetson NX, Jetson Orin | [NVIDIA-sourced] |

## GPU architecture support

Use this table as an architecture gate. If the user's GPU architecture first appears after their DeepStream version, the risk is HIGH or BLOCKED.

| Architecture | Compute capability | First DS row in current public matrix | Example products | Authority |
| --- | --- | --- | --- | --- |
| Blackwell | 10.x / 12.x | 8.0 | RTX Pro 6000 Blackwell, RTX 50-series, B100, B200, GB200 | [NVIDIA-sourced] |
| Hopper | 9.0 | 6.2 | H100, H200 | [NVIDIA-sourced] |
| Ada Lovelace | 8.9 | 6.2 | L4, L40, RTX 6000 Ada, RTX 40-series | [NVIDIA-sourced] |
| Ampere | 8.0 / 8.6 | 6.1.1 | A100, A30, A10, RTX A6000, RTX 30-series | [NVIDIA-sourced] |
| Turing | 7.5 | 6.1.1 | T4, RTX 20-series | [NVIDIA-sourced] |

## Jetson SoC support

| Jetson SoC | First public DeepStream row in this matrix | Notes |
| --- | --- | --- |
| Thor | 8.0 | DS 9.1 is the current public recommendation for Thor. |
| Orin | 6.1.1 | DS 9.1 is the current public recommendation for Orin on JetPack 7.2. |
| Xavier | 6.1.1 | Legacy platform; verify against release-specific docs before recommending new work. |
| Nano | 6.3 | Legacy platform; verify release-specific support. |

## Specialty platforms

| Platform | GPU/SoC | Public DeepStream path | Authority |
| --- | --- | --- | --- |
| DGX Spark | GB10 Blackwell | DS 9.1 ARM SBSA container path: `nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark` | [NVIDIA-sourced] |

## How to use this matrix

1. Choose the discrete GPU table or Jetson table.
2. Find the user's reported DeepStream version.
3. Compare driver, JetPack/L4T, CUDA, TensorRT, Ubuntu, and platform.
4. If a required companion component is older than the public row, return Risk: HIGH.
5. If the user's hardware first appears after their DeepStream version, return Risk: HIGH or BLOCKED.
6. If the stack is newer than the matrix or missing key fields, return Risk: MEDIUM and ask for verification.
7. If the stack matches a public row, return Risk: LOW unless the symptom maps to a known non-version issue in `troubleshooting.md`.

## What this matrix does not cover

- Native Windows DeepStream support. DeepStream is not supported natively on Windows; use Linux or WSL2 plus containers.
- Customer-built, patched, or private containers. Verify the installed version inside the container.
- TensorRT engines copied across platforms. Engines are not portable across TensorRT versions or CPU/GPU architectures.
- Product drift after publication. Always prefer current public NVIDIA release notes if they differ from this file.
