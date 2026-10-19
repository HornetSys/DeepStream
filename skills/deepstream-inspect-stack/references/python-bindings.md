# DeepStream Python Bindings

Use this file when the user reports `ModuleNotFoundError`, mixes `pyservicemaker` and `pyds`, asks which binding to use, or has Python import failures inside a virtual environment.

## Binding families

| Binding | DeepStream line | Module | Notes |
| --- | --- | --- | --- |
| `pyservicemaker` | DS 9.x | `from pyservicemaker import Pipeline, Flow` | Current high-level Python API. The wheel is bundled under `/opt/nvidia/deepstream/deepstream/service-maker/python/`. |
| `pyds` | Older DS Python workflows | `import pyds` | Legacy binding used by many DS 6.x/7.x Python examples. Verify against the user's release-specific docs before recommending it. |

For new DS 9.x work, prefer `pyservicemaker`. Do not tell users to install `pyservicemaker` or `pyds` from random third-party packages.

## The venv pitfall

Symptom inside an activated venv:

```python
from pyservicemaker import Pipeline
ModuleNotFoundError: No module named 'pyservicemaker'
```

Likely cause: the DeepStream container or host install has the wheel, but a normal `python3 -m venv` does not inherit system site-packages.

Fix inside the activated venv:

```bash
pip install /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl pyyaml
```

Risk: LOW. This is a packaging/venv setup issue, not a DeepStream version incompatibility.

## Shared-library import failure

Symptom:

```text
ImportError: libpython3.12.so.1.0: cannot open shared object file: No such file or directory
```

Likely cause: `pyservicemaker` is installed, but the matching Python shared library is missing from the environment where the import runs.

Verify:

```bash
python3 --version
python3 -c "import pyservicemaker; print(pyservicemaker.__file__)" 2>&1
dpkg -l 2>&1 | grep -E "libpython3\.12|python3\.12"
```

Fix in the affected environment:

```bash
sudo apt-get update
sudo apt-get install -y libpython3.12
```

The skill may recommend this as the user's explicitly approved next step, but must not run it autonomously. In a root container, the user may omit `sudo`.

## pyds notes

If the user reports `ModuleNotFoundError: No module named 'pyds'`:

- On DS 9.x, this is usually expected for new workflows; use `pyservicemaker`.
- On older DS releases, verify the release-specific Python bindings package and install method.
- If the request turns into Python app construction or migration, reroute to `deepstream-dev`.

## Verify commands

```bash
cat /opt/nvidia/deepstream/deepstream/version
which python3
python3 --version
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
ls -1 /opt/nvidia/deepstream/deepstream/service-maker/python/ 2>&1
python3 -c "import pyservicemaker; print(pyservicemaker.__file__)" 2>&1
python3 -c "import pyds; print(pyds.__file__)" 2>&1
```

## What not to do

- Do not build Python bindings from source as a first response.
- Do not mix `pyds` and `pyservicemaker` in one generated workflow.
- Do not generate pipeline code from this skill; reroute pipeline work to `deepstream-dev`.
