# DeepStream on WSL2 — Documented Steps (source of truth)

Security-conscious implementation of the official guide, used as the source of
truth for this skill. It preserves the supported stack while narrowing broad
container, network, and X11 permissions. Record any further deviation in the run
report.

Source: <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_on_WSL2.html>

## Prerequisites (validated matrix from the guide)

- **Windows 11** with an NVIDIA GPU in **WDDM mode** (GeForce or RTX/Quadro
  workstation). **Tesla / datacenter GPUs are NOT supported** for the WSL2 path.
- **NVIDIA Windows driver** (current guide validates GameReady 595.79 with an RTX-3050;
  any current GeForce/Studio/Workstation DCH driver that reports WDDM works).
- **WSL2** from the Microsoft Store (guide validated a recent stable release; check `wsl --version`).
- **Ubuntu 24.04** as the WSL distro.

> The NVIDIA driver is installed **on Windows only**. Do **not** install a Linux
> NVIDIA driver inside WSL — the GPU is projected into WSL by the Windows driver.

## Step 1 — Install WSL2 and Ubuntu (run on Windows, admin terminal)

**Install the latest stable STANDALONE WSL2 package — do NOT rely on the Windows
inbox default / `wsl --update`.** Even if `wsl.exe` already exists, download and
install the latest stable release from `microsoft/WSL` (the standalone x64 MSI),
matching the guide's use of the Store/GitHub WSL2 package:

```powershell
# Discover + download the latest stable x64 MSI from microsoft/WSL releases.
# Require the release API's SHA-256 digest and a valid Microsoft Authenticode signature.
$rel = Invoke-RestMethod https://api.github.com/repos/microsoft/WSL/releases/latest
$asset = $rel.assets | Where-Object { $_.name -match '\.x64\.msi$' } | Select-Object -First 1
if (-not $asset -or -not $asset.digest -or $asset.digest -notmatch '^sha256:') {
  throw 'WSL release has no official SHA-256 digest; pin a release with a published digest'
}
curl.exe -fL -o C:\wsl-latest-x64.msi $asset.browser_download_url
$expected = $asset.digest.Substring(7).ToLowerInvariant()
$actual = (Get-FileHash C:\wsl-latest-x64.msi -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "WSL MSI SHA-256 mismatch: expected $expected got $actual" }
$signature = Get-AuthenticodeSignature C:\wsl-latest-x64.msi
if ($signature.Status -ne 'Valid') { throw "WSL MSI signature invalid: $($signature.Status)" }
msiexec /i C:\wsl-latest-x64.msi /quiet /norestart
wsl --version                 # confirm the standalone version is active
```

> **GitHub API rate limit:** from a shared corporate NAT egress IP, the
> unauthenticated `api.github.com` call can return HTTP 403 (rate limited). If so,
> resolve the latest release from your side and use the direct
> `https://github.com/microsoft/WSL/releases/download/<tag>/wsl.<ver>.0.x64.msi`
> URL (release *downloads* are not rate limited), or pin a known-good version.

Then install the distro on WSL 2:

```bash
wsl --set-default-version 2
wsl --list --online
wsl --install --no-launch -d Ubuntu-24.04
wsl --list --verbose          # confirm Ubuntu-24.04, VERSION 2
```

> Run first-time setup non-interactively as root (avoids the interactive
> user-creation prompt): `wsl -d Ubuntu-24.04 -u root -- <cmd>`.

## Step 2 — Verify the NVIDIA driver (inside Ubuntu WSL)

```bash
nvidia-smi
```

`nvidia-smi` must list the GPU and a CUDA version. If it fails, the Windows
driver is missing/old or the GPU is in TCC (not WDDM) mode.

The displayed CUDA version is the driver's capability, not proof of an installed
WSL CUDA Toolkit. Compare the Windows driver with the selected release baseline.
If it is incompatible, use the official [NVIDIA driver
selector](https://www.nvidia.com/Download/index.aspx), choose the matching GPU
and Windows release, verify NVIDIA's publisher signature, install interactively,
reboot, and repeat both Windows and WSL `nvidia-smi` checks.

## Optional — CUDA Toolkit for host compilation

Skip this for container-only DeepStream. If the user explicitly needs to compile
CUDA outside the container, install only the release-resolved versioned
`cuda-toolkit-X-Y` package from NVIDIA's WSL repository. Never install a CUDA or
driver metapackage that brings a Linux display driver into WSL. Follow the
[CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## Step 3 — Install Docker Engine (inside Ubuntu WSL)

```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
# apt-key is deprecated (Ubuntu 22.04+); use a local signing-key file — matches wsl_provision_deepstream.sh
mkdir -p /tmp/ds-gpgkeys
sudo install -d -m 0755 /etc/apt/ds-keys
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /tmp/ds-gpgkeys/docker.gpg
sudo gpg --dearmor -o /etc/apt/ds-keys/docker-signing-key.gpg < /tmp/ds-gpgkeys/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/ds-keys/docker-signing-key.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /tmp/ds-gpgkeys/docker.list
sudo mv /tmp/ds-gpgkeys/docker.list /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io --fix-missing
# WSL has no systemd by default — start the daemon BEFORE any docker client command
sudo dockerd >/var/log/dockerd.log 2>&1 &
sudo docker info
```

> On WSL there is no systemd by default, so the Docker daemon is started manually
> with `sudo dockerd &` (or enable systemd in `/etc/wsl.conf`).

## Step 4 — Install the NVIDIA Container Toolkit (inside Ubuntu WSL)

```bash
mkdir -p /tmp/ds-gpgkeys
sudo install -d -m 0755 /etc/apt/ds-keys
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o /tmp/ds-gpgkeys/nvidia-ctk.gpg
sudo gpg --dearmor -o /etc/apt/ds-keys/nvidia-ctk-signing-key.gpg < /tmp/ds-gpgkeys/nvidia-ctk.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/etc/apt/ds-keys/nvidia-ctk-signing-key.gpg] https://#g' \
  > /tmp/ds-gpgkeys/nvidia-ctk.list
sudo mv /tmp/ds-gpgkeys/nvidia-ctk.list /etc/apt/sources.list.d/nvidia-container-toolkit.list
NVIDIA_CTK_VERSION="${NVIDIA_CTK_VERSION:-1.18.0-1}"
sudo apt-get update
sudo apt-get install -y "nvidia-container-toolkit=${NVIDIA_CTK_VERSION}"
sudo nvidia-ctk runtime configure --runtime=docker
# default WSL has no systemd — restart the manually-launched daemon:
sudo pkill dockerd 2>/dev/null || true
sudo dockerd >/var/log/dockerd.log 2>&1 &
```

> Only if you enabled systemd in `/etc/wsl.conf` can you use
> `sudo systemctl restart docker` instead of the manual restart above.

## Step 5 — Pull and run the DeepStream container (inside Ubuntu WSL)

> **SUPPLY-CHAIN WARNING:** Always pin by digest in production to prevent tag substitution.
> Get the digest: `docker inspect --format='{{index .RepoDigests 0}}' nvcr.io/nvidia/deepstream:9.1-triton-multiarch`

```bash
# RECOMMENDED (digest-pinned, production):
sudo docker pull nvcr.io/nvidia/deepstream:9.1-triton-multiarch@sha256:<verified-from-ngc>
# Development/testing only (mutable tag):
# sudo docker pull nvcr.io/nvidia/deepstream:9.1-triton-multiarch
```

> **NOTE:** The official NVIDIA guide uses additional runtime flags for display/audio access.
> For that form, see: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_on_WSL2.html
> Those host interfaces are not granted by this skill. The default smoke run is
> headless, has no network, and does not expose the host display or sound device.

```bash
sudo docker run --rm --name=deepstream-smoke --gpus all \
  --network none --security-opt=no-new-privileges --cap-drop=ALL \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v /usr/lib/wsl/lib/libnvcuvid.so.1:/usr/lib/x86_64-linux-gnu/libnvcuvid.so.1:ro \
  -v /usr/lib/wsl/lib/libnvidia-encode.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1:ro \
  --entrypoint deepstream-app \
  nvcr.io/nvidia/deepstream:9.1-triton-multiarch@sha256:<verified-from-ngc> \
  --version-all
```

## Step 6 — Verify DeepStream (inside the container)

```text
deepstream-app --version-all
cd /opt/nvidia/deepstream/deepstream/samples/streams
gst-launch-1.0 filesrc location=sample_720p.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=filedump.mp4 -v
cd /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app
deepstream-app -c source30_1080p_dec_infer-resnet_tiled_display.txt
```

> **Headless note (from the guide):** for Ubuntu 24.04 docker, prefer **filesink**
> output for verification rather than the on-screen rendering plugins
> (`nveglglessink` / tiled display), which have known issues without a display.
> This skill's verification runs the filesink transcode plus an `nvinfer`
> inference smoke test to a `fakesink`, so it produces a deterministic
> PASS/FAIL with no display attached.
