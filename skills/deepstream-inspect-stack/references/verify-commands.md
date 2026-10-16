# Verify Commands

Use this file when building the "Verify (run these and paste output)" section of a diagnostic, or when the user has not provided enough environment data.

In Mode A, render commands as copy-paste shell snippets for the user to run. In Mode B, run only announced read-only system commands after user opt-in. Never autonomously run a `python3 -c "import ..."` probe: it is client-mediated and requires pasted user output. In both modes, never run commands that install, remove, download, pull containers, edit config, or modify state.

Before asking for or relaying container output, tell the user to redact non-public registry hostnames, private project paths, unpublished build identifiers, and credentials from image names. Keep public NGC image names such as `nvcr.io/nvidia/deepstream:9.1-triton-multiarch`; replace internal or private image prefixes with placeholders such as `<private-registry>/<private-image>:<tag>`. In Mode B, redact those fields before including command output in the final diagnostic.

## One-shot intake script

Tell the user to paste the entire output back after redacting any private/internal registry URLs or unpublished build identifiers in the container section.

### Discrete GPU host (x86_64)

```bash
echo "=== PLATFORM ==="
uname -a
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv 2>&1
echo
echo "=== OS ==="
grep -E "^(NAME|VERSION_ID)" /etc/os-release
echo
echo "=== DEEPSTREAM ==="
cat /opt/nvidia/deepstream/deepstream/version 2>&1
deepstream-app --version-all 2>&1 | head
echo
echo "=== CUDA TOOLKIT ==="
nvcc --version 2>&1 | grep -i release
ls -1 /usr/local/cuda* 2>&1
echo
echo "=== TENSORRT ==="
dpkg -l 2>&1 | grep -i tensorrt | head
echo
echo "=== PYTHON ==="
python3 --version 2>&1
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
python3 -c "import pyservicemaker; print('pyservicemaker:', pyservicemaker.__file__)" 2>&1
python3 -c "import pyds; print('pyds:', pyds.__file__)" 2>&1
echo
echo "=== CONTAINERS ==="
docker ps --format 'image={{.Image}} status={{.Status}}' 2>&1
nvidia-container-cli info 2>&1 | head
```

### Jetson host (ARM64)

```bash
echo "=== PLATFORM ==="
uname -a
cat /proc/device-tree/model 2>/dev/null || true
cat /etc/nv_tegra_release 2>&1
echo
echo "=== JETPACK / L4T ==="
dpkg -l 2>&1 | grep -E "nvidia-jetpack|nvidia-l4t-core" | head
echo
echo "=== OS ==="
grep -E "^(NAME|VERSION_ID)" /etc/os-release
echo
echo "=== DEEPSTREAM ==="
cat /opt/nvidia/deepstream/deepstream/version 2>&1
deepstream-app --version-all 2>&1 | head
echo
echo "=== CUDA ==="
nvcc --version 2>&1 | grep -i release
ls -1 /usr/local/cuda* 2>&1
echo
echo "=== TENSORRT ==="
dpkg -l 2>&1 | grep -i tensorrt | head
echo
echo "=== PYTHON ==="
python3 --version 2>&1
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
python3 -c "import pyservicemaker; print('pyservicemaker:', pyservicemaker.__file__)" 2>&1
python3 -c "import pyds; print('pyds:', pyds.__file__)" 2>&1
echo
echo "=== CONTAINERS ==="
docker ps --format 'image={{.Image}} status={{.Status}}' 2>&1
```

## Targeted commands

### GPU or board

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
cat /proc/device-tree/model 2>/dev/null || true
cat /etc/nv_tegra_release 2>/dev/null || true
```

### DeepStream version

```bash
cat /opt/nvidia/deepstream/deepstream/version
deepstream-app --version-all
```

### Driver

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

### CUDA

```bash
nvcc --version | grep release
ls -l /usr/local/cuda
```

### TensorRT

```bash
dpkg -l | grep -i tensorrt | head
python3 -c "import tensorrt; print(tensorrt.__version__)" 2>&1
```

### JetPack / L4T

```bash
dpkg -l | grep -E "nvidia-jetpack|nvidia-l4t-core"
cat /etc/nv_tegra_release
```

### Python and venv

```bash
python3 --version
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
which python3
python3 -c "import pyservicemaker; print(pyservicemaker.__file__)" 2>&1
python3 -c "import pyds; print(pyds.__file__)" 2>&1
```

### Container

```bash
docker ps --format '{{.Image}}'
docker exec <container_id> cat /opt/nvidia/deepstream/deepstream/version
```

### NVIDIA Container Toolkit

```bash
nvidia-container-cli info 2>&1 | head
which nvidia-container-runtime
```

### Samples

```bash
ls /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/ 2>&1
ls /opt/nvidia/deepstream/deepstream/samples/streams/*.mp4 2>&1
```

## Commands not to recommend as verification

- No `apt install`, `apt purge`, `rm -rf`, `dpkg --remove`, `pip install`, or `docker pull` as autonomous verification.
- Install commands may appear only as the user's manual next step after diagnosis.
- If a verify command requires elevated privileges, show `sudo` explicitly.

## Reading output

| You see | It means |
| --- | --- |
| Blackwell GPU with DS 7.x or earlier | Public matrix says the DS line is too old. Risk: HIGH or BLOCKED. |
| DS 9.1 on dGPU with driver below the public DS 9.1 row | Driver likely too old. Risk: HIGH. |
| `deepstream-app: command not found` | DS not installed or PATH not set. Verify `/opt/nvidia/deepstream/`. |
| `pyservicemaker` missing in a venv | Install the bundled wheel into the venv. Risk: LOW. |
| `nvidia-container-cli: command not found` | NVIDIA Container Toolkit missing. Risk: HIGH. |
| `cat: /opt/nvidia/deepstream/deepstream/version: No such file` | DS not installed at the standard path or running only inside a container. |
| `tegra_release` line present | Jetson host. Use board model plus L4T, not L4T alone. |

## Gaps

- The skill does not SSH to remote machines.
- The skill does not parse output with a deterministic parser.
- If DeepStream exists only inside a stopped container, host probes may report no DS install. Ask the user to start or inspect the relevant container manually.
