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

"""Create and delete private WinRM HTTPS target profiles.

Passwords are accepted only from a hidden local terminal prompt. They are never
accepted as arguments, environment variables, or chat input.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import stat
import sys
from pathlib import Path


DEFAULT_TARGETS_DIR = Path(
    os.environ.get(
        "DEEPSTREAM_WSL_TARGETS_DIR",
        "~/.deepstream-wsl/targets",
    )
).expanduser()
NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,254}")


def _private_dir(path: Path) -> None:
    if path.exists():
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise SystemExit(f"refusing unsafe target directory: {path}")
        if st.st_uid != os.getuid():
            raise SystemExit(f"target directory is not owned by the current user: {path}")
        if st.st_mode & 0o077:
            raise SystemExit(
                f"target directory must not be group/world accessible: {path}"
            )
        return
    path.mkdir(parents=True, mode=0o700)
    os.chmod(path, 0o700)


def _write_private(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def _validate_text(value: str, label: str) -> str:
    if not value or len(value) > 255 or any(ord(ch) < 32 for ch in value):
        raise SystemExit(f"invalid {label}")
    return value


def create(args: argparse.Namespace) -> int:
    if not NAME_RE.fullmatch(args.name):
        raise SystemExit("--name must be lowercase letters, digits, and hyphens")
    if not sys.stdin.isatty():
        raise SystemExit("target creation requires an interactive local terminal")

    host = args.host or input("Windows hostname or IP: ").strip()
    user = args.user or input("Windows username: ").strip()
    if not HOST_RE.fullmatch(host):
        raise SystemExit("invalid host: use a hostname or IP address")
    _validate_text(user, "Windows username")

    password = getpass.getpass("Windows password (hidden): ")
    confirm = getpass.getpass("Confirm Windows password: ")
    if not password:
        raise SystemExit("password must not be empty")
    if password != confirm:
        raise SystemExit("passwords do not match")

    root = Path(args.targets_dir).expanduser().resolve()
    _private_dir(root)
    target_dir = root / args.name
    if target_dir.exists():
        raise SystemExit(
            f"target {args.name!r} already exists; delete it explicitly before recreating"
        )
    _private_dir(target_dir)

    pass_path = target_dir / "winrm.pass"
    profile_path = target_dir / "target.json"
    try:
        _write_private(pass_path, password.encode("utf-8"))
        profile = {
            "schema_version": 1,
            "host": host,
            "user": user,
            "transport": "https",
            "port": 5986,
            "pass_file": "winrm.pass",
        }
        _write_private(
            profile_path,
            (json.dumps(profile, indent=2) + "\n").encode("utf-8"),
        )
    except BaseException:
        pass_path.unlink(missing_ok=True)
        profile_path.unlink(missing_ok=True)
        try:
            target_dir.rmdir()
        except OSError:
            pass
        raise
    finally:
        password = ""
        confirm = ""

    print(f"TARGET_PROFILE={profile_path}")
    print("CREDENTIAL_STORED=true")
    return 0


def _load_profile_path(raw: str) -> tuple[Path, dict]:
    profile_path = Path(raw).expanduser().resolve()
    st = profile_path.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise SystemExit(f"refusing unsafe profile: {profile_path}")
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise SystemExit(f"profile must be user-owned and mode 0600: {profile_path}")
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid target profile: {exc}") from exc
    return profile_path, profile


def delete(args: argparse.Namespace) -> int:
    if not args.yes:
        raise SystemExit("refusing to delete credentials without --yes")
    profile_path, profile = _load_profile_path(args.profile)
    target_dir = profile_path.parent.resolve()
    pass_path = (target_dir / str(profile.get("pass_file", ""))).resolve()
    if pass_path.parent != target_dir:
        raise SystemExit("profile pass_file escapes its target directory")

    if pass_path.exists():
        st = pass_path.lstat()
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            raise SystemExit(f"refusing unsafe password path: {pass_path}")
        if st.st_uid != os.getuid():
            raise SystemExit(f"password file is not owned by the current user: {pass_path}")
        pass_path.unlink()
    profile_path.unlink()
    try:
        target_dir.rmdir()
    except OSError:
        pass
    print("CREDENTIAL_DELETED=true")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage private DeepStream WSL target profiles")
    sub = parser.add_subparsers(dest="command", required=True)

    create_parser = sub.add_parser("create")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--host", help="non-secret hostname/IP; prompt if omitted")
    create_parser.add_argument("--user", help="non-secret Windows username; prompt if omitted")
    create_parser.add_argument("--targets-dir", default=str(DEFAULT_TARGETS_DIR))
    create_parser.set_defaults(func=create)

    delete_parser = sub.add_parser("delete")
    delete_parser.add_argument("--profile", required=True)
    delete_parser.add_argument("--yes", action="store_true")
    delete_parser.set_defaults(func=delete)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
