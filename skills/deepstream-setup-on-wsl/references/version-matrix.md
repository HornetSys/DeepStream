# Per-release version matrix — resolve & confirm before provisioning

**Each DeepStream release is tested against a specific stack** (Windows driver,
CUDA, Ubuntu / WSL distro, TensorRT, WSL2 version). Using the wrong OS/driver for a
given DeepStream version wastes a lease and produces bogus results. So for the
requested version, resolve the matrix from **that release's own doc** and **confirm
with the user** before leasing/provisioning.

## Where the authoritative versions live

- **Version-specific WSL doc** (lists what was validated for that release):
  `https://docs.nvidia.com/metropolis/deepstream/<VER>/dev-guide/text/DS_on_WSL2.html`
  e.g. `.../deepstream/7.1/dev-guide/text/DS_on_WSL2.html`. The **unversioned**
  `https://nvidia-deepstream.docs.buildwithfern.com/deepstream/9.2/text/DS_on_WSL2` is
  the **current/latest** release.
- **DeepStream release notes / "Platform and OS Compatibility"** for that version
  (driver, CUDA, TensorRT, Ubuntu).
- **NGC** for the exact container tag: `nvcr.io/nvidia/deepstream` → tags list.
- If the **user supplies a doc page**, read it — it names the versions tested for
  that release.

## What to resolve (and confirm) per request

| Field | Notes |
| --- | --- |
| DeepStream image tag | e.g. `nvcr.io/nvidia/deepstream:<VER>-triton-multiarch` — confirm exact tag on NGC (`-triton-multiarch` / `-triton` / `-samples` vary by release) |
| **Ubuntu / WSL distro** | **differs by release** — DS 9.0 → Ubuntu-24.04; DS 7.x → Ubuntu-22.04; DS 6.x → Ubuntu-20.04 (verify from the release doc). Sets `--distro` + `wsl --install -d Ubuntu-XX.04` |
| Windows driver minimum | derive from the container CUDA release's compatibility table; hard gate |
| WSL-validated driver | Windows version from the release's WSL guide; never copy the Linux driver number |
| Driver CUDA capability | reported by `nvidia-smi`; do not confuse it with an installed Toolkit |
| Container CUDA/TensorRT | shipped in the image; verify their package versions after pull |
| Host CUDA Toolkit | absent by default; resolve a versioned toolkit-only package only for host compilation |
| WSL2 version | default latest stable from microsoft/WSL, or the doc's validated one |

## Known-good rows (verify against the release doc before use)

| DeepStream | Ubuntu/WSL distro | Windows driver: CUDA minimum / WSL validated | Container CUDA / TensorRT | WSL2 | Status |
| --- | --- | --- | --- | --- | --- |
| 9.1-triton-multiarch | Ubuntu-24.04 | R580 / 595.79 | 13.2 / 10.16 | 2.7.3 validated | official baseline |
| 9.0-triton-multiarch | Ubuntu-24.04 | resolve archived 9.0 docs | resolve archived 9.0 docs | release doc | resolve from doc |
| 7.x | Ubuntu-22.04 (confirm) | per release doc | per release doc | release doc | resolve from doc |
| 6.x | Ubuntu-20.04 (confirm) | per release doc | per release doc | release doc | resolve from doc |

Live observations belong in test evidence, not minimum requirements. A different
driver may work, but do not weaken the official baseline without authoritative
compatibility evidence.

> Treat the table above as a starting hint only. **Always reconcile with the
> requested release's doc / release notes and confirm with the user** — do not
> assume older/newer releases match the rows above.
