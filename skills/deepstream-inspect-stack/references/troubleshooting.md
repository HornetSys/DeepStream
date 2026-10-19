# DeepStream Version-Related Troubleshooting

Use this file when the user reports a specific error message and you need to map it to a likely version, packaging, or container cause.

Scope: version/install/compatibility errors only. For pipeline-construction bugs, reroute to `deepstream-dev`. For model import, ONNX export, TensorRT engine building, or benchmarking, reroute to `deepstream-import-vision-model`.

## Python and venv errors

### `ModuleNotFoundError: No module named 'pyservicemaker'` in a venv

Likely cause: the venv does not inherit the DeepStream-installed `pyservicemaker` wheel.

Verify:

```bash
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
ls /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl
cat /opt/nvidia/deepstream/deepstream/version
```

Fix:

```bash
pip install /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl pyyaml
```

Risk: LOW.

### `ModuleNotFoundError: No module named 'pyservicemaker'` outside a venv

Likely cause: the user is not in a DS 9.x environment, or DeepStream is not installed where Python is running.

Verify:

```bash
cat /opt/nvidia/deepstream/deepstream/version 2>&1
python3 --version
docker ps --format '{{.Image}}'
```

Fix: move into a DS 9.x environment or use the release-specific Python binding for the installed DeepStream line.

Risk: HIGH if the code targets the wrong API for the installed DS version; MEDIUM if the environment is simply not activated.

### `ImportError: libpython3.12.so.1.0`

Likely cause: `pyservicemaker`'s native extension can be found, but the Python 3.12 shared library is absent from the environment.

Verify:

```bash
python3 --version
dpkg -l 2>&1 | grep -E "libpython3\.12|python3\.12"
python3 -c "import pyservicemaker; print(pyservicemaker.__file__)" 2>&1
```

Fix:

```bash
sudo apt-get update
sudo apt-get install -y libpython3.12
```

Risk: LOW for compatibility; HIGH for Python app execution until fixed.

### `ModuleNotFoundError: No module named 'pyds'`

Likely cause: legacy binding is not installed, or the user is on DS 9.x where `pyservicemaker` is the preferred API for new work.

Verify:

```bash
cat /opt/nvidia/deepstream/deepstream/version
python3 -c "import pyds; print(pyds.__file__)" 2>&1
```

Fix: use `pyservicemaker` for new DS 9.x work. For older DS releases, check the release-specific Python bindings package.

Risk: MEDIUM.

## Driver and CUDA errors

### `cuInit failed: CUDA driver version is insufficient for CUDA runtime version`

Likely cause: the host driver is older than the container or DeepStream row requires.

Verify:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
deepstream-app --version-all 2>&1 | head
nvcc --version 2>&1 | grep -i release
```

Fix: upgrade the host driver to the public row required by the installed DeepStream version, or pin DeepStream to a release compatible with the installed driver.

Risk: HIGH.

### `Unsupported SM: 0xc00` or `Unsupported SM: 0xa00`

Likely cause: Blackwell GPU running with a DeepStream/TensorRT stack too old to support that compute capability.

Verify:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
cat /opt/nvidia/deepstream/deepstream/version
```

Fix: use DS 9.1 as the current public path for Blackwell, or DS 8.0+ if pinned to an older DeepStream release that publicly lists Blackwell.

Risk: HIGH. If the user must stay on DS 7.x or earlier, Risk: BLOCKED.

### cuDNN or TensorRT version mismatch

Likely cause: a bare-metal install has mixed components from different release rows.

Verify:

```bash
dpkg -l | grep -E "cudnn|tensorrt|nvinfer" | head
cat /opt/nvidia/deepstream/deepstream/version
```

Fix: use the matched NGC DeepStream container for the target release, or reinstall components according to the release-specific public installation guide.

Risk: HIGH on bare metal; LOW after switching to a matched container.

## TensorRT errors

### `setDimensions: Error Code 3: Internal Error (Assertion mDims.d[i] > 0)`

Likely cause: dynamic ONNX input dimensions are not specified for `nvinfer`.

This is not primarily a version mismatch. Reroute to `deepstream-dev` for config guidance, or to `deepstream-import-vision-model` if the user is importing the model.

Risk: MEDIUM.

### TensorRT engine built in one environment, run in another

Likely cause: TensorRT engines are tied to TensorRT version, platform, and GPU architecture.

Verify:

```bash
dpkg -l | grep -i libnvinfer-bin | head
uname -m
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Fix: build and run the engine in the same target environment.

Risk: HIGH.

## Container errors

### `nvidia-container-cli: initialization error`

Likely cause: NVIDIA Container Toolkit is not installed/configured, or the container was launched without GPU access.

Verify:

```bash
nvidia-container-cli info 2>&1 | head
docker ps --format '{{.Image}}'
```

Fix: configure NVIDIA Container Toolkit and launch with GPU access according to NVIDIA Container Toolkit documentation.

Risk: HIGH.

### `exec format error`

Likely cause: image architecture does not match the host architecture.

Verify:

```bash
uname -m
docker ps --format '{{.Image}}'
docker manifest inspect <image> 2>&1 | head
```

Fix: use a public DeepStream tag that supports the host architecture, such as `9.1-triton-multiarch` for x86_64 or Jetson, or `9.1-triton-sbsa-dgx-spark` for DGX Spark.

Risk: HIGH.

### Samples missing

Likely cause: the user selected a runtime/base-style image instead of a samples/devel/triton image.

Verify:

```bash
docker ps --format '{{.Image}}'
ls /opt/nvidia/deepstream/deepstream/samples/ 2>&1
```

Fix: switch to a samples, devel, or triton tag for first-look validation.

Risk: LOW.

## Jetson-specific errors

### Jetson Thor with DS 7.x or earlier

Likely cause: Thor requires a newer public DeepStream/JetPack row than the user has installed.

Verify:

```bash
cat /proc/device-tree/model 2>/dev/null
cat /etc/nv_tegra_release
cat /opt/nvidia/deepstream/deepstream/version 2>&1
```

Fix: use the current public Thor path in `known-good-setups.md`.

Risk: BLOCKED.

### Jetson Orin with engine built on x86_64

Likely cause: TensorRT engines are not portable between x86_64 and Jetson ARM64.

Verify:

```bash
uname -m
cat /opt/nvidia/deepstream/deepstream/version
```

Fix: build the engine in the target Jetson environment. Reroute implementation steps to `deepstream-import-vision-model`.

Risk: HIGH.

## Windows

### Native Windows install request

Likely cause: the user assumes DeepStream runs natively on Windows.

Fix: DeepStream is not supported natively on Windows. Recommend Linux or WSL2 plus Ubuntu plus NVIDIA container runtime plus an NGC DeepStream container.

Risk: BLOCKED for native Windows; LOW for a verified WSL2/container path.

## Fallback

If the symptom does not match a row above:

1. Ask for the one-shot intake script from `verify-commands.md`.
2. Compare the output against `compatibility-matrix.md`.
3. If the stack matches, say the issue is probably not a version mismatch and reroute to the appropriate sibling skill.

Default fallback risk: MEDIUM.
