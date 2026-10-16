#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check or install controller-side dependencies in a private virtual environment."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


MIN_PYTHON = (3, 10)
REQUIREMENTS = Path(__file__).resolve().parent / "requirements-controller.txt"
DEFAULT_VENV = Path(
    os.environ.get("DEEPSTREAM_WSL_VENV", "~/.deepstream-wsl/venv")
).expanduser()


def _venv_python(venv_dir: Path) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return venv_dir / directory / executable


def _locked_versions() -> dict[str, str]:
    locked: dict[str, str] = {}
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            locked[name] = version
    return locked


def _has_locked_requirements(python: Path) -> bool:
    if not python.exists():
        return False
    expected = json.dumps(_locked_versions())
    code = (
        "import json; from importlib.metadata import version; "
        f"e=json.loads({expected!r}); "
        "raise SystemExit(0 if all(version(k)==v for k,v in e.items()) else 1)"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def check(venv_dir: Path) -> bool:
    if sys.version_info < MIN_PYTHON:
        raise SystemExit(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; "
            f"found {sys.version_info.major}.{sys.version_info.minor}"
        )
    python = _venv_python(venv_dir)
    ready = _has_locked_requirements(python)
    print(f"PYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"VENV={venv_dir}")
    print(f"PYWINRM_READY={'true' if ready else 'false'}")
    print(f"OPENSSH_CLIENT={'true' if shutil.which('ssh') else 'false'}")
    print(f"BOOTSTRAP_READY={'true' if ready else 'false'}")
    if ready:
        print(f"PYTHON={python}")
    return ready


def install(args: argparse.Namespace, venv_dir: Path) -> int:
    if not args.yes:
        raise SystemExit("refusing dependency installation without --yes")
    if sys.version_info < MIN_PYTHON:
        check(venv_dir)
    print(
        f"INSTALL_PLAN=create private venv {venv_dir}; "
        f"install exact versions from {REQUIREMENTS}"
    )
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    if venv_dir.exists():
        st = venv_dir.lstat()
        if venv_dir.is_symlink() or not venv_dir.is_dir():
            raise SystemExit(f"refusing unsafe virtual environment path: {venv_dir}")
        if st.st_uid != os.getuid():
            raise SystemExit(f"virtual environment is not owned by the current user: {venv_dir}")
    python = _venv_python(venv_dir)
    if not python.exists():
        venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)
    os.chmod(venv_dir, 0o700)
    result = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "-r",
            str(REQUIREMENTS),
        ],
        check=False,
    )
    if result.returncode != 0 or not _has_locked_requirements(python):
        raise SystemExit("controller dependency installation failed")
    print("BOOTSTRAP_READY=true")
    print(f"PYTHON={python}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap DeepStream WSL controller tools")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--install", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--venv", default=str(DEFAULT_VENV))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    venv_dir = Path(args.venv).expanduser().resolve()
    if args.check:
        return 0 if check(venv_dir) else 1
    return install(args, venv_dir)


if __name__ == "__main__":
    raise SystemExit(main())
