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

"""Robust WinRM executor for the DeepStream-on-WSL2 skill.

Why this exists: the ansible/win_command path is fragile (heavy dep, UTF-8 locale
requirement, and brittle quoting when nesting `wsl … -- bash -lc "…"`). This helper
uses **pywinrm** directly with PowerShell **-EncodedCommand** (base64 UTF-16LE), so
there is zero cmd/PowerShell quoting to get wrong, and it sets generous timeouts for
long installs.

Subcommands:
  run-ps   --host H [--user U] [--pass-file F] (--command PS | --script FILE) [--timeout N]
  run-wsl  --host H [--distro Ubuntu-24.04] (--command BASH | --script FILE) [--timeout N]
           runs a bash command/script inside the WSL distro as root (base64-piped,
           so no nested-quote issues): wsl -d <distro> -u root -- bash -lc '…'
  wait     --host H [--timeout N]   wait for WinRM to answer (e.g. after a reboot)

Auth: password from a private target profile or --pass-file. Never pass the
password on the command line. Public remote targets require HTTPS.

Usage:
  python scripts/winrm_exec.py --target-profile PROFILE run-ps --command "PS"
  python scripts/winrm_exec.py --host HOST --user USER --use-https run-wsl --command "bash cmd"

Output:
  Stdout/stderr from the remote command; key=value lines for structured responses.

Arguments:
  --host HOST         Windows host IP or hostname
  --user USER         Local Windows admin username
  --pass-file FILE    Path to chmod-600 password file

Subcommands:
  run-ps   --command PS | --script FILE   Run a PowerShell command/script
  run-wsl  --distro D  --command CMD      Run a bash command inside WSL
  wait     --timeout N                    Poll until WinRM responds
  push     --local L --remote R           Upload a file to the Windows host
  pull     --distro D --wsl-path P        Download a file from WSL

Exit codes:
  0  Success
  1  Connection or execution error
  2  Argument error
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import time


def _protocol_class():
    """Import pywinrm lazily so bootstrap/help work before dependencies exist."""
    try:
        from winrm.protocol import Protocol
        return Protocol
    except ImportError as exc:
        raise SystemExit(
            "pywinrm is not installed in this interpreter. Run "
            "`python scripts/bootstrap.py --check`, then confirm and run "
            "`python scripts/bootstrap.py --install --yes`; use the PYTHON path it prints."
        ) from exc

# Where the remote Windows password is read from (single line = the password).
# Prefer --target-profile. The legacy path remains for HTTPS-only compatibility.
DEFAULT_PASS_FILE = os.path.expanduser(
    os.environ.get("WINRM_PASS_FILE", "~/.deepstream-wsl/winrm_pass"))

WINRM_HTTP_PORT = 5985
WINRM_HTTPS_PORT = 5986
WINRM_READ_TIMEOUT_EXTRA_S = 30   # added to operation timeout for read_timeout_sec
CHUNK_BYTES = 1800   # keeps the UTF-16 EncodedCommand below the Windows limit
UPLOAD_VERIFY_TIMEOUT_S = 60
UPLOAD_DECODE_TIMEOUT_S = 180
DOCKERD_WAIT_ITER = 25
DOCKERD_SLEEP_S = 1
WINRM_WAIT_PROBE_TIMEOUT_S = 20  # timeout for each WinRM probe during wait-for-reboot
RUN_PS_DEFAULT_TIMEOUT_S = 300   # default timeout for run-ps subcommand (s)


def _env_prefix(pairs) -> str:
    """Build a safe `export K='V' …; ` prefix from --env KEY=VALUE pairs.

    Values are single-quoted and restricted to a safe charset (no quotes/spaces/
    shell metachars) so a crafted value can't break out of the wsl bash -lc "…".
    """
    if not pairs:
        return ""
    parts = []
    for p in pairs:
        if "=" not in p:
            sys.exit(f"--env must be KEY=VALUE (got {p!r})")
        k, v = p.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            sys.exit(f"invalid --env key {k!r}")
        if not re.fullmatch(r"[A-Za-z0-9._:/@%,=+-]*", v):
            sys.exit(f"invalid --env value for {k}: {v!r} (safe chars only)")
        parts.append(f"{k}='{v}'")
    return "export " + " ".join(parts) + "; "


def _private_regular_file(raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser()
    try:
        st = path.lstat()
    except FileNotFoundError as exc:
        raise SystemExit(f"{label} not found: {path}") from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise SystemExit(f"{label} must be a regular file, not a symlink: {path}")
    if st.st_uid != os.getuid():
        raise SystemExit(f"{label} is not owned by the current user: {path}")
    if st.st_mode & 0o077:
        raise SystemExit(
            f"{label} must not be group/world accessible; run `chmod 600 {path}`"
        )
    return path.resolve()


def _password(args) -> str:
    path = _private_regular_file(args.pass_file, "password file")
    data = path.read_bytes()
    if data.endswith(b"\r\n"):
        data = data[:-2]
    elif data.endswith(b"\n"):
        data = data[:-1]
    if not data:
        raise SystemExit(f"password file is empty: {path}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"password file is not valid UTF-8: {path}") from exc


def _apply_target_profile(args) -> None:
    """Resolve a private target profile onto argparse connection fields."""
    if not getattr(args, "target_profile", None):
        args.allow_http = False
        if not getattr(args, "use_https", False):
            raise SystemExit(
                "remote WinRM requires --target-profile or an explicit --use-https"
            )
        return

    profile_path = _private_regular_file(args.target_profile, "target profile")
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid target profile JSON: {exc}") from exc
    if profile.get("schema_version") != 1:
        raise SystemExit("unsupported target profile schema_version")

    host = str(profile.get("host", ""))
    user = str(profile.get("user", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,254}", host):
        raise SystemExit("invalid host in target profile")
    if not user or len(user) > 255 or any(ord(ch) < 32 for ch in user):
        raise SystemExit("invalid user in target profile")
    pass_path = (profile_path.parent / str(profile.get("pass_file", ""))).resolve()
    if pass_path.parent != profile_path.parent:
        raise SystemExit("target profile pass_file escapes its private directory")

    transport = profile.get("transport")
    args.use_https = transport == "https"
    args.allow_http = False
    if not args.use_https:
        raise SystemExit("target profile transport must be https")
    expected_port = WINRM_HTTPS_PORT
    if profile.get("port", expected_port) != expected_port:
        raise SystemExit(f"target profile port must be {expected_port} for {transport}")

    args.host = host
    args.user = user
    args.pass_file = str(pass_path)


def resolve_connection(args) -> str:
    _apply_target_profile(args)
    if not getattr(args, "host", None):
        raise SystemExit("no Windows host: pass --target-profile or --host")
    if not getattr(args, "user", None):
        raise SystemExit("no Windows user: pass --target-profile or --user")
    return _password(args)


def _decode(b: bytes) -> str:
    # wsl.exe / some tools emit UTF-16LE; PowerShell text is usually UTF-8.
    if len(b) > 1 and b[1] == 0:
        return b.decode("utf-16-le", errors="replace")
    return b.decode("utf-8", errors="replace")


def _run_ps(host, user, pw, ps, timeout, transports=("ntlm", "basic"),
            use_https=False, allow_http=False):
    """Run a PowerShell script via -EncodedCommand. Returns (rc, out, err)."""
    enc = base64.b64encode(ps.encode("utf-16-le")).decode()
    last = None
    if use_https:
        scheme, port = "https", WINRM_HTTPS_PORT
    elif allow_http:
        scheme, port = "http", WINRM_HTTP_PORT
        print(f"\n[SECURITY WARNING] WinRM to {host}:{port} has no TLS server identity.\n"
              "  NTLM message encryption is required, but HTTPS remains the public default.\n"
              "  This mode is restricted to an installed private adapter and isolated network.",
              file=sys.stderr)
        transports = ("ntlm",)
    else:
        sys.exit(
            f"ERROR: WinRM transport to {host} requires an explicit choice:\n"
            "  --use-https    HTTPS port 5986 (recommended for external/untrusted networks)\n"
            "  private adapter profile  HTTP may be enabled only by an installed adapter\n"
            "Pass --use-https or a valid target profile to proceed.")
    for tp in transports:
        try:
            Protocol = _protocol_class()
            protocol_args = {
                "transport": tp,
                "username": user,
                "password": pw,
                "read_timeout_sec": timeout + WINRM_READ_TIMEOUT_EXTRA_S,
                "operation_timeout_sec": timeout,
            }
            if use_https:
                protocol_args["server_cert_validation"] = "validate"
            else:
                # The private adapter uses NTLM message encryption even though
                # its isolated provisioning endpoint does not terminate TLS.
                protocol_args["message_encryption"] = "always"
            p = Protocol(f"{scheme}://{host}:{port}/wsman", **protocol_args)
            sh = p.open_shell()
            cid = p.run_command(sh, "powershell",
                                ["-NoProfile", "-NonInteractive", "-EncodedCommand", enc])
            out, err, code = p.get_command_output(sh, cid)
            p.cleanup_command(sh, cid)
            p.close_shell(sh)
            return code, _decode(out), _decode(err)
        except Exception as exc:  # noqa: BLE001 - try next transport
            last = exc
    raise SystemExit(f"WinRM connection failed: {last}")


# Windows command lines cap at ~8 KB, so scripts are UPLOADED as a base64 text
# file in chunks (each WinRM call stays tiny) and executed from disk, not inlined.
_REMOTE_DIR = r"C:\dsw"
_REMOTE_WSL_DIR = "/mnt/c/dsw"


def _safe_windows_staging_path(path: str) -> str:
    if not re.fullmatch(r"C:\\dsw\\[A-Za-z0-9._-]{1,180}", path or "", re.IGNORECASE):
        raise SystemExit("remote Windows path must be a simple file under C:\\dsw")
    return path


def _safe_wsl_path(path: str) -> str:
    if (
        not re.fullmatch(r"/[A-Za-z0-9._/-]{1,500}", path or "")
        or ".." in Path(path).parts
    ):
        raise SystemExit("remote WSL path must be an absolute path without traversal")
    return path


def _safe_distro(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", value or ""):
        raise SystemExit("invalid WSL distribution name")
    return value


def _upload_b64(host, user, pw, data: bytes, win_b64_path: str, *,
                use_https: bool, allow_http: bool) -> None:
    b64 = base64.b64encode(data).decode()
    # Each chunk is embedded in PowerShell and then UTF-16/base64 encoded again by
    # _run_ps. Keep the resulting command below the tighter 8191-character path
    # encountered by WinRM-hosted Windows PowerShell, not only CreateProcess's
    # theoretical 32767-character ceiling.
    # For files larger than ~1 MB, bypass this entirely: serve the file from a local
    # HTTP server and have Windows download it with Invoke-WebRequest (see SKILL.md
    # Step 3 — LAN transfer is orders of magnitude faster than chunked WinRM).
    chunk = CHUNK_BYTES
    _run_ps(host, user, pw,
            f"New-Item -ItemType Directory -Force -Path '{_REMOTE_DIR}' | Out-Null; "
            f"Remove-Item -Force -ErrorAction SilentlyContinue '{win_b64_path}'",
            UPLOAD_VERIFY_TIMEOUT_S, use_https=use_https, allow_http=allow_http)
    for i in range(0, len(b64), chunk):
        rc, _, err = _run_ps(host, user, pw,
                             f"Add-Content -NoNewline -Encoding ascii -Path "
                             f"'{win_b64_path}' -Value '{b64[i:i + chunk]}'",
                             UPLOAD_VERIFY_TIMEOUT_S,
                             use_https=use_https, allow_http=allow_http)
        if rc != 0:
            raise SystemExit(f"upload chunk failed: {err.strip()[:200]}")
    # verify the uploaded base64 length matches — catches any silent truncation.
    _, out, _ = _run_ps(host, user, pw,
                        f"(Get-Item '{win_b64_path}').Length", UPLOAD_VERIFY_TIMEOUT_S,
                        use_https=use_https, allow_http=allow_http)
    got = "".join(ch for ch in out if ch.isdigit())
    if got != str(len(b64)):
        raise SystemExit(f"upload size mismatch: wrote {got}, expected {len(b64)}")


def upload_binary(host, user, pw, local_path: str, win_path: str, *,
                  use_https: bool, allow_http: bool) -> None:
    """Upload a local file to a Windows path (binary-exact, via chunked base64)."""
    _safe_windows_staging_path(win_path)
    with open(local_path, "rb") as fh:
        data = fh.read()
    b64p = win_path + ".b64"
    _upload_b64(host, user, pw, data, b64p,
                use_https=use_https, allow_http=allow_http)
    _run_ps(host, user, pw,
            f"[IO.File]::WriteAllBytes('{win_path}',"
            f"[Convert]::FromBase64String([IO.File]::ReadAllText('{b64p}'))); "
            f"Remove-Item -Force '{b64p}'", UPLOAD_DECODE_TIMEOUT_S,
            use_https=use_https, allow_http=allow_http)


_DOWNLOAD_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB hard limit for downloads


def download_wsl(host, user, pw, distro: str, wsl_path: str, local_path: str,
                 timeout: int = 600, *, use_https: bool, allow_http: bool) -> None:
    """Download a file from inside the WSL distro (binary-exact, via base64)."""
    _safe_distro(distro)
    _safe_wsl_path(wsl_path)
    # Validate local_path stays within the expected temp directory to prevent path traversal.
    import pathlib
    local_abs = pathlib.Path(local_path).resolve()
    allowed_root = pathlib.Path("/tmp").resolve()  # nosec B108 - skill-controlled temp dir
    if not local_abs.is_relative_to(allowed_root):
        raise SystemExit(f"download_wsl: local_path {local_path!r} is outside /tmp — refusing")
    # Log the download operation for auditability.
    print(f"[INFO] Downloading {wsl_path} from WSL ({distro}) on {host} -> {local_abs}",
          file=sys.stderr)
    size_ps = f'wsl -d {distro} -u root -- bash -lc "stat -c%s {wsl_path}"'
    rc, out, err = _run_ps(
        host, user, pw, size_ps, timeout,
        use_https=use_https, allow_http=allow_http)
    if rc != 0:
        raise SystemExit(f"download failed for {wsl_path}: {err.strip()[:200]}")
    try:
        remote_size = int(out.strip())
    except ValueError as exc:
        raise SystemExit(f"download failed for {wsl_path}: invalid remote size") from exc
    if remote_size > _DOWNLOAD_MAX_BYTES:
        raise SystemExit(
            f"download_wsl: {wsl_path} is {remote_size} bytes — exceeds 2 GB limit"
        )
    ps = f'wsl -d {distro} -u root -- bash -lc "base64 -w0 {wsl_path}"'
    rc, out, err = _run_ps(
        host, user, pw, ps, timeout,
        use_https=use_https, allow_http=allow_http)
    if rc != 0:
        raise SystemExit(f"download failed for {wsl_path}: {err.strip()[:200]}")
    data = base64.b64decode("".join(out.split()))
    if len(data) > _DOWNLOAD_MAX_BYTES:
        raise SystemExit(f"download_wsl: {wsl_path} is {len(data)} bytes — exceeds 2 GB limit")
    with open(local_abs, "wb") as fh:
        fh.write(data)


def cmd_push(args) -> int:
    pw = resolve_connection(args)
    upload_binary(
        args.host, args.user, pw, args.local, args.remote,
        use_https=args.use_https, allow_http=args.allow_http)
    print(f"PUSHED {args.local} -> {args.remote}")
    return 0


def cmd_pull(args) -> int:
    pw = resolve_connection(args)
    download_wsl(
        args.host, args.user, pw, args.distro,
        args.remote_wsl, args.local, args.timeout,
        use_https=args.use_https, allow_http=args.allow_http)
    print(f"PULLED {args.remote_wsl} -> {args.local}")
    return 0


def cmd_run_ps(args) -> int:
    pw = resolve_connection(args)
    if args.script:
        with open(args.script, "rb") as fh:
            data = fh.read()
        b64p = _REMOTE_DIR + r"\script.ps1.b64"
        _upload_b64(
            args.host, args.user, pw, data, b64p,
            use_https=args.use_https, allow_http=args.allow_http)
        ps1 = _REMOTE_DIR + r"\script.ps1"
        ps = (f"$b=[IO.File]::ReadAllText('{b64p}'); "
              f"[IO.File]::WriteAllBytes('{ps1}',[Convert]::FromBase64String($b)); "
              f"& powershell -NoProfile -NonInteractive -File '{ps1}'")
    else:
        ps = args.command
    if getattr(args, "dry_run", False):
        print(f"[DRY-RUN] Would execute PowerShell on {args.host}: {ps[:200]}{'...' if len(ps)>200 else ''}")
        return 0
    rc, out, err = _run_ps(args.host, args.user, pw, ps, args.timeout,
                            use_https=getattr(args, "use_https", False),
                            allow_http=getattr(args, "allow_http", False))
    sys.stdout.write(out)
    if err.strip():
        sys.stderr.write(err)
    return rc


def cmd_run_wsl(args) -> int:
    pw = resolve_connection(args)
    _safe_distro(args.distro)
    env = _env_prefix(getattr(args, "env", None))  # exported → inherited by the run bash
    if args.script:
        with open(args.script, "rb") as fh:
            data = fh.read()
        suffix = Path(args.script).suffix
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix or ""):
            suffix = ".sh"
        name = f"script-{secrets.token_hex(8)}{suffix}"
        b64p = f"{_REMOTE_DIR}\\{name}.b64"
        wsl_b64 = f"{_REMOTE_WSL_DIR}/{name}.b64"
        _upload_b64(
            args.host, args.user, pw, data, b64p,
            use_https=args.use_https, allow_http=args.allow_http)
        # decode in WSL (strips CR from the base64 text) and run from /tmp
        ps = (f'wsl -d {args.distro} -u root -- bash -lc '
              f'"{env}tr -d \'\\r\' < {wsl_b64} | base64 -d > /tmp/{name} && bash /tmp/{name}"')
    else:
        b64 = base64.b64encode(args.command.encode()).decode()
        ps = (f'wsl -d {args.distro} -u root -- bash -lc '
              f'"{env}echo {b64} | base64 -d | bash"')
    if getattr(args, "dry_run", False):
        decoded = args.command if args.command else f"<script: {args.script}>"
        print(f"[DRY-RUN] Would execute in WSL ({args.distro}) on {args.host}: {decoded[:200]}")
        return 0
    print(f"[INFO] Executing in WSL ({args.distro}) on {args.host} as root: "
          f"{(args.command or args.script or '')[:100]}", file=sys.stderr)
    rc, out, err = _run_ps(args.host, args.user, pw, ps, args.timeout,
                            use_https=getattr(args, "use_https", False),
                            allow_http=getattr(args, "allow_http", False))
    sys.stdout.write(out)
    if err.strip():
        sys.stderr.write(err)
    return rc


def cmd_wait(args) -> int:
    pw = resolve_connection(args)
    start = time.time()
    while time.time() - start < args.timeout:
        try:
            rc, out, _ = _run_ps(args.host, args.user, pw, "$env:COMPUTERNAME",
                                 WINRM_WAIT_PROBE_TIMEOUT_S, transports=("ntlm",),
                                 use_https=args.use_https, allow_http=args.allow_http)
            if rc == 0:
                print(f"WINRM_READY=true ({out.strip()})")
                return 0
        except SystemExit:
            pass
        time.sleep(args.interval)
    print("WINRM_READY=false")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robust WinRM executor (pywinrm + EncodedCommand)")
    p.add_argument("--target-profile",
                   help="private target profile created by configure_target.py")
    p.add_argument("--host")
    p.add_argument("--user", default=os.environ.get("WINRM_USER"),
                   help="Windows local-admin account, or set $WINRM_USER")
    p.add_argument("--pass-file", default=DEFAULT_PASS_FILE)
    p.add_argument("--use-https", action="store_true",
                   help="Connect via HTTPS port 5986 (recommended for external/untrusted networks).")
    p.set_defaults(allow_http=False)
    sub = p.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run-ps")
    g = rp.add_mutually_exclusive_group(required=True)
    g.add_argument("--command")
    g.add_argument("--script")
    rp.add_argument("--timeout", type=int, default=RUN_PS_DEFAULT_TIMEOUT_S)
    rp.add_argument("--dry-run", action="store_true",
                    help="Print the command that would be executed without running it.")
    rp.set_defaults(func=cmd_run_ps)

    rw = sub.add_parser("run-wsl")
    rw.add_argument("--distro", default="Ubuntu-24.04")
    g2 = rw.add_mutually_exclusive_group(required=True)
    g2.add_argument("--command")
    g2.add_argument("--script")
    rw.add_argument("--env", action="append", metavar="KEY=VALUE",
                    help="export an env var into the WSL command (repeatable), "
                         "e.g. --env DS_IMAGE=nvcr.io/... --env NVIDIA_CTK_VERSION=1.18.0-1")
    rw.add_argument("--timeout", type=int, default=900)
    rw.add_argument("--dry-run", action="store_true",
                    help="Print the WSL command that would be executed without running it.")
    rw.set_defaults(func=cmd_run_wsl)

    w = sub.add_parser("wait")
    w.add_argument("--timeout", type=int, default=600)
    w.add_argument("--interval", type=int, default=10)
    w.set_defaults(func=cmd_wait)

    pu = sub.add_parser("push")
    pu.add_argument("--local", required=True)
    pu.add_argument("--remote", required=True, help="Windows path, e.g. C:\\dsw\\x")
    pu.set_defaults(func=cmd_push)

    pl = sub.add_parser("pull")
    pl.add_argument("--distro", default="Ubuntu-24.04")
    pl.add_argument("--remote-wsl", required=True, help="path inside the WSL distro")
    pl.add_argument("--local", required=True)
    pl.add_argument("--timeout", type=int, default=600)
    pl.set_defaults(func=cmd_pull)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
