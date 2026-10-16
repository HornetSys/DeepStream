# DeepStream on WSL2 — FAQ & known-issue learnings

Distilled from the official
[DeepStream on WSL2](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_on_WSL2.html)
guide, its
[FAQ](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_WSL2_FAQ.html),
and the NVIDIA issue tracker / forums. These drive several skill behaviors.

## WARNING: nvv4l2 codec SIGSEGV — pin nvidia-container-toolkit

**Symptom (seen live 2026-07 on RTX 3090):** any pipeline using the hardware codec
elements `nvv4l2decoder` / `nvv4l2h264enc` crashes inside the container with:

```text
Error while setting IOCTL
Invalid control
S_EXT_CTRLS for CUDA_GPU_ID failed
Caught SIGSEGV
```

**Root cause:** a bug in **nvidia-container-toolkit `1.19.1-1`** on WSL2. CUDA /
TensorRT compute is unaffected (nvinfer builds engines fine); only the V4L2 HW
codecs crash.

**Fix:** pin the container toolkit to a known-good version — **`1.18.0-1`**:

```bash
sudo apt-get install -y --allow-downgrades \
  nvidia-container-toolkit=1.18.0-1 nvidia-container-toolkit-base=1.18.0-1 \
  libnvidia-container-tools=1.18.0-1 libnvidia-container1=1.18.0-1
sudo nvidia-ctk runtime configure --runtime=docker
# restart the docker daemon (and `wsl --shutdown` if needed)
```

`wsl_provision_deepstream.sh` does this automatically (`NVIDIA_CTK_VERSION`,
default `1.18.0-1`). Refs:
[nvidia-container-toolkit#1864](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1864),
[DeepStream forum](https://forums.developer.nvidia.com/t/nvv4l2decoder-nvv4l2h264enc-crash-in-container-nvidia-container-toolkit-1-19-1-1-s-ext-ctrls-for-cuda-gpu-id-failed/372158).

## FAQ points that shape the skill

- **Driver:** install only the **Windows** GPU driver; never a Linux driver in WSL.
  Prefer a **clean install** (driver "Advanced" → clean) to avoid lib mismatches.
- **`could not select device driver` (docker GPU):**
  Run in **Ubuntu WSL**: `sudo apt install --reinstall nvidia-container-toolkit`
  then restart the daemon — if systemd is enabled: `sudo systemctl restart docker`;
  otherwise: `sudo pkill dockerd 2>/dev/null || true` then `sudo dockerd >/var/log/dockerd.log 2>&1 &`
  Then in **Windows PowerShell**: `wsl --shutdown` and reopen: `wsl -d Ubuntu-24.04`.
- **`GPU access blocked by operating system`:** unsupported GPU — see the
  [WSL2 support constraints](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#wsl-2-support-constraints).
  (Datacenter GPUs unsupported; GeForce/Quadro in WDDM only.)
- **No video output / display sink errors** (`MESA: Failed to attach to x11 shm`,
  `nveglglessink` on Ubuntu-24.04 docker): known WSL2 rendering issue → **use
  `filesink`** (this skill's checks are already headless/filesink/fakesink-based).
  Also: do **not** launch WSL through an SSH session (e.g. MobaXterm) if you need
  display output — use VNC; `wsl --shutdown` first if already launched via SSH.
- **Ubuntu install "service cannot be started":** set the Windows **Windows
  Update** service (services.msc) startup to Automatic and start it.
- **Harmless startup noise:** GLib pthread error, and warnings about missing OSS
  plugin libs on first `deepstream-app` runs — safe to ignore.
- **`wsl --install` caveat:** on some Windows builds it installs the limited inbox
  WSL — install the **standalone latest stable** from
  [microsoft/WSL releases](https://github.com/microsoft/WSL/releases) instead
  (this skill does).
- **Docker in WSL has no systemd by default:** start with `sudo dockerd &`.

## Maintenance — re-check the upstream doc each DeepStream release

The official pages change per release. Before running for a new DeepStream version,
re-read and reconcile this skill against:

- Guide: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_on_WSL2.html>
- FAQ:   <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_WSL2_FAQ.html>

Watch specifically for changes to: the verification commands (Step 6), the DeepStream
image tag, supported GPUs/driver, the recommended WSL2 version, and any new
known-issues/workarounds — then update `pipelines/` and
`wsl_provision_deepstream.sh` accordingly.

**Last reconciled 2026-07 against DeepStream 9.1**, with this validated stack:
DS image `nvcr.io/nvidia/deepstream:9.1-triton-multiarch`; Windows 11 22H2 (build 22621);
Windows driver 591.86 (CUDA 13.1 in WSL, 13.2 in container); standalone WSL2 2.7.11; Ubuntu-24.04;
**nvidia-container-toolkit pinned 1.18.0-1** (1.19.1-1 is broken on WSL2).
All 4 sanity pipelines passed (version, decode, transcode, deepstream-app-infer) on RTX 3080.
Previously also validated against DS 9.0 on RTX 3090 with WSL2 2.7.10 — same stack.

## Troubleshooting table (symptom → cause → recovery)

Recovery is **bounded and forward-only** — retry a failed provision at most once
(or swap the machine at most twice), then stop and report. Never an open loop.

| Symptom | Likely cause | Recovery |
| --- | --- | --- |
| GPU-check fails (datacenter part) | Unsupported target GPU | Stop; use a WDDM GeForce/RTX/Quadro workstation box |
| `nvidia-smi` missing in WSL | Windows driver missing / TCC mode | Update the Windows driver; ensure WDDM (`nvidia-smi --query-gpu=driver_model.current`) |
| `docker info` fails | No systemd; daemon not started | Script starts `dockerd` manually; check `/var/log/dockerd.log` |
| container `nvidia-smi` fails | Container Toolkit not wired to docker | Re-run provisioning (re-runs `nvidia-ctk runtime configure` + daemon restart) |
| `docker pull` fails | nvcr.io auth / network | `docker login nvcr.io` on the host; retry |
| `S_EXT_CTRLS for CUDA_GPU_ID failed` → **SIGSEGV** in nvv4l2decoder/nvv4l2h264enc | **nvidia-container-toolkit 1.19.1-1 bug on WSL2** | Pin the toolkit to `1.18.0-1` (provisioning does this via `NVIDIA_CTK_VERSION`); see the critical note above |
| `could not select device driver` (GPU) | container runtime not registered | `apt install --reinstall nvidia-container-toolkit`; if systemd is enabled: `sudo systemctl restart docker`; otherwise: `sudo pkill dockerd 2>/dev/null || true` then `sudo dockerd >/var/log/dockerd.log 2>&1 &`; run `wsl --shutdown` then reopen |
| display-sink / `x11 shm` / `nveglglessink` error | WSL2 rendering limitation | use `filesink`/`fakesink` (skill already does); for display use VNC not SSH-launched WSL |
| nvinfer FAIL but engine "serialize … successfully" | crash is in the nvv4l2 decoder, not TensorRT | apply the container-toolkit pin above |
