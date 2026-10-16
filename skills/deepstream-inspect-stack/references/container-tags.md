# DeepStream NGC Container Tags

Use this file when the user reports a container image (`nvcr.io/nvidia/deepstream:*`) or asks which public NGC DeepStream tag to use.

Authoritative public source: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream

Container tags can change between releases. If a tag is not listed here, verify it in the NGC catalog before claiming it exists.

## Current public DS 9.1 tags

| Tag | Contains | Use when |
| --- | --- | --- |
| `9.1-triton-multiarch` | DeepStream SDK plus Triton support in a multi-architecture image | Default for first-look validation, Triton-backed inference, and general diagnostics on x86_64 or Jetson |
| `9.1-samples-multiarch` | DeepStream SDK plus sample applications and assets in a multi-architecture image | First-look sample validation when Triton is not required |
| `9.1-triton-sbsa-dgx-spark` | DeepStream SDK plus Triton for DGX Spark / ARM SBSA | DGX Spark / GB10 systems |

For DS 9.1, prefer these public tags over older DS 9.0 tags unless the user is intentionally pinned to an older release.

## Older public DS 9.0 tags

| Tag | Contains | Use when |
| --- | --- | --- |
| `9.0-triton-multiarch` | DeepStream SDK plus Triton support in a multi-architecture image | DS 9.0 validation on x86_64 or Jetson |
| `9.0-devel` | SDK, headers, samples, and build tools | Building custom plugins or parsers on DS 9.0 |
| `9.0-base` | Runtime-oriented image with fewer development assets | Production-style custom app runtime; not for first-look samples |
| `9.0-triton-sbsa-dgx-spark` | DS 9.0 ARM SBSA path for DGX Spark | DGX Spark users pinned to DS 9.0 |

## Tag suffix meanings

| Suffix | Typical contents | Use when |
| --- | --- | --- |
| `-base` | Runtime libraries with minimal extra assets | Production-style deployment of an already-built app |
| `-samples` or `-samples-multiarch` | Samples, configs, and sample media | First-look exploration and sanity checks |
| `-devel` | Development headers, samples, and build tools | Building custom plugins, custom parsers, or source examples |
| `-triton` or `-triton-multiarch` | Triton Inference Server components plus DeepStream | `nvinferserver` or Triton-backed inference |
| `-sbsa-dgx-spark` | ARM SBSA image for DGX Spark | DGX Spark / GB10 |

## Multi-architecture behavior

Modern DeepStream tags marked `multiarch` can resolve to the correct architecture for x86_64 or Jetson ARM64. If the user reports `exec format error`, still verify the actual image manifest and host architecture.

Verify:

```bash
uname -m
docker ps --format '{{.Image}}'
docker manifest inspect <image> 2>&1 | head
```

## Common mistakes

### Base image when the user wanted samples

Symptom: `/opt/nvidia/deepstream/deepstream/samples/` is missing or sparse.

Likely cause: the user selected a runtime-style tag.

Fix: choose a samples, devel, or triton tag that includes the assets needed for first-look validation.

Risk: LOW.

### Wrong architecture image

Symptom: `exec format error` on startup.

Likely cause: image architecture does not match the host.

Fix: use a public multi-arch DeepStream tag for the user's DeepStream release, or the release-specific Jetson/L4T image documented for that release.

Risk: HIGH.

### Trusting the tag instead of the installed version

Symptom: the user says the tag is DS 9.x, but `deepstream-app --version-all` or the version file shows a different SDK.

Verify:

```bash
cat /opt/nvidia/deepstream/deepstream/version
deepstream-app --version-all
```

Risk: MEDIUM until the installed version is confirmed.

## Recommended defaults

| Platform | Default public container |
| --- | --- |
| Discrete GPU workstation/server | `nvcr.io/nvidia/deepstream:9.1-triton-multiarch` |
| Jetson Thor / Orin | `nvcr.io/nvidia/deepstream:9.1-triton-multiarch` |
| DGX Spark / GB10 | `nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark` |
| DS 9.0 pinned environment | `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` |

## Verify commands for the user

```bash
docker ps --format '{{.Image}}'
docker exec <container> cat /opt/nvidia/deepstream/deepstream/version
docker exec <container> ls /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/ 2>&1
docker exec <container> ls /opt/tritonserver/ 2>/dev/null && echo "Triton present"
```
