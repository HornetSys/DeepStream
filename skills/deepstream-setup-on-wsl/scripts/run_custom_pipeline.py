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

"""Run and verify a USER-DEFINED pipeline inside the DeepStream container in WSL.

For skill Step 6: the user gives a command (a gst-launch / deepstream-app / app
invocation), optional input files to stage, and pass criteria (regex(es) that must
appear, and/or a minimum match count). This driver bundles the command + inputs,
runs them in the DeepStream container (inside Docker inside WSL) on the remote
Windows box, pulls the output back to a local run dir, checks the criteria, and
prints + logs a PASS/FAIL verdict.

Example:
  python scripts/run_custom_pipeline.py --target-profile ~/.deepstream-wsl/targets/my-box/target.json \\
    --ds-image nvcr.io/nvidia/deepstream:9.1-triton-multiarch --name my-check \\
    --input ./in.mp4 \\
    --cmd 'gst-launch-1.0 -e filesrc location=/work/in.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! fakesink' \\
    --expect-regex 'Got EOS'

Usage:
  python scripts/run_custom_pipeline.py --target-profile PROFILE --ds-image IMAGE \\
    --name check-name --cmd 'gst-launch-1.0 ...' --expect-regex 'Got EOS' --yes

Arguments:
  --target-profile P  Private target profile created by configure_target.py
  --ds-image IMAGE    DeepStream container image to use
  --cmd CMD           Pipeline command to run inside the container
  --name NAME         Check name (used in log filename)
  --yes               Confirm this exact command may run (required)
  --expect-regex PAT  Regex that must appear in output for PASS
  --min-count N       Minimum match count (with --count-regex)

Output:
  Prints PASS/FAIL verdict; writes custom.<name>.log to run dir.

Exit codes:
  0  Pipeline command passed all criteria
  1  Pipeline failed or criteria not met
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import secrets
import shlex
import stat
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import winrm_exec as W  # noqa: E402
import run_verification as RV  # noqa: E402  (reuse the hardened tar extractor)

DOWNLOAD_TIMEOUT_S = 600    # max seconds to pull results bundle from WSL
DEFAULT_TIMEOUT_S = 1800    # --timeout default
CONTAINER_TIMEOUT_S = 1500  # hard kill for the container run (s) — matches --container-timeout default
DOCKERD_WAIT_ITERS = 25     # max iterations waiting for dockerd to become ready

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_NAME = os.path.basename(SKILL_DIR)
DEFAULT_RUNS_DIR = os.path.join("/tmp", SKILL_NAME, "runs")  # nosec B108


def _validate_ref(value: str, kind: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._:/@-]{1,255}", value or ""):
        sys.exit(f"invalid {kind} {value!r}: allowed chars are [A-Za-z0-9._:/@-]")
    return value


_ALLOWED_EXECUTABLES = (
    "gst-launch-1.0", "deepstream-app", "deepstream-test",
    "python3", "python",
    "ffmpeg", "nvidia-smi", "nvcc",
)
_SHELL_CONTROL = re.compile(r"[\r\n;&|<>`]|\$\(")


def _validate_cmd(cmd: str) -> list[str]:
    """Parse one executable invocation; reject shell syntax and chaining."""
    if _SHELL_CONTROL.search(cmd):
        raise SystemExit("--cmd rejected: shell operators, redirection, and chaining are not allowed")
    try:
        argv = shlex.split(cmd, posix=True)
    except ValueError as exc:
        raise SystemExit(f"--cmd rejected: invalid quoting ({exc})") from exc
    if not argv:
        raise SystemExit("--cmd is empty")
    executable = argv[0]
    if executable not in _ALLOWED_EXECUTABLES:
        raise SystemExit(
            "--cmd rejected: executable is not allowed. "
            f"Allowed names: {', '.join(_ALLOWED_EXECUTABLES)}."
        )
    if executable in ("python", "python3") and (
        len(argv) < 2 or argv[1].startswith("-")
    ):
        raise SystemExit("Python commands must name a script; -c/-m/stdin execution is not allowed")
    return argv


def _redact_cmd(cmd: str) -> str:
    return re.sub(
        r"([A-Za-z][A-Za-z0-9+.-]*://)([^/@:\s]+):([^/@\s]+)@",
        r"\1<redacted>:<redacted>@",
        cmd,
    )


def _make_bundle(cmd: str, inputs: list[str]) -> bytes:
    """tgz with _run.sh (the user command) + any input files at the top level."""
    argv = _validate_cmd(cmd)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        script = (
            "#!/usr/bin/env bash\n"
            "# Confirmed single executable invocation; cwd=/work.\n"
            "set -euo pipefail\nexec " + shlex.join(argv) + "\n"
        )
        info = tarfile.TarInfo("_run.sh")
        data = script.encode()
        info.size = len(data)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(data))
        names = set()
        for f in inputs or []:
            try:
                mode = os.stat(f, follow_symlinks=False).st_mode
            except OSError:
                sys.exit(f"--input not found: {f}")
            if not stat.S_ISREG(mode):
                sys.exit(f"--input must be a regular non-symlink file: {f}")
            name = os.path.basename(f)
            if name in names:
                sys.exit(f"--input basenames must be unique: {name}")
            names.add(name)
            tar.add(f, arcname=name, recursive=False)
    return buf.getvalue()


def _remote_job(ds_image: str, timeout: int, token: str, bundle_wsl: str,
                allow_network: bool) -> tuple[str, str]:
    # Live-tested: nvv4l2decoder/nvv4l2h264enc work with --gpus all alone on WSL2
    # (no additional security-opt or host networking needed). Verified 2026-07-28 on RTX 3080.
    _work = f"/tmp/dscustom-{token}"   # bounded WSL /tmp workspace  # nosec B108
    _out_tgz = f"/tmp/dscustom-out-{token}.tgz"  # nosec B108
    network_flag = "--network bridge" if allow_network else "--network none"
    job = f"""set -e
rm -f {_out_tgz}
rm -rf {_work} && mkdir -p {_work}/out {_work}/work
tar xzf {bundle_wsl} -C {_work}/work
# Note: dockerd may persist after this job exits. Idempotent: re-uses existing daemon if already running.
started_dockerd=0
if ! sudo docker info >/dev/null 2>&1; then
  sudo nohup dockerd >/var/log/dockerd.log 2>&1 </dev/null &
  started_dockerd=1
fi
for _ in $(seq 1 {DOCKERD_WAIT_ITERS}); do sudo docker info >/dev/null 2>&1 && break; sleep 1; done
codec_mounts=()
for lib in /usr/lib/wsl/lib/libnvcuvid.so.1 /usr/lib/wsl/lib/libnvidia-encode.so.1; do
  if [[ -f "$lib" ]]; then
    base=$(basename "$lib")
    codec_mounts+=(--mount "type=bind,src=$lib,dst=/usr/lib/x86_64-linux-gnu/$base,readonly")
  fi
done
cidfile={_work}/container.cid
set +e
timeout --signal=KILL {timeout} \
sudo docker run --rm --cidfile "$cidfile" --gpus all \
  --entrypoint /bin/bash \
  --security-opt=no-new-privileges --cap-drop=ALL \
  {network_flag} \
  "${{codec_mounts[@]}}" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  -v {_work}/work:/work -w /work \
  {ds_image} bash /work/_run.sh > {_work}/out/custom.log 2>&1
echo "DOCKER_RC=$?" >> {_work}/out/custom.log
if [[ -f "$cidfile" ]]; then
  sudo docker kill "$(cat "$cidfile")" >/dev/null 2>&1 || true
fi
# dockerd is stopped after this job by default (STOP_DOCKERD=1). Set STOP_DOCKERD=0 to
# leave it running for performance when making multiple consecutive skill invocations.
if [[ "${{STOP_DOCKERD:-1}}" == "1" && "$started_dockerd" == "1" ]]; then sudo pkill dockerd 2>/dev/null || true; fi
set -e
tar czf {_out_tgz} -C {_work} out
echo "RESULTS_BUNDLED={_out_tgz}"
"""
    return job, _out_tgz


def _evaluate(log: str, args) -> tuple[str, list[str]]:
    """Return (PASS|FAIL, reasons)."""
    reasons = []
    m = re.search(r"DOCKER_RC=(\d+)", log)
    rc = int(m.group(1)) if m else 1
    ok = True
    if not args.allow_nonzero and rc != 0:
        ok = False
        reasons.append(f"exit code {rc} (expected 0)")
    elif rc != 0:
        reasons.append(f"exit code {rc} (allowed)")
    for pat in args.expect_regex or []:
        if re.search(pat, log):
            reasons.append(f"found /{pat}/")
        else:
            ok = False
            reasons.append(f"MISSING /{pat}/")
    if args.min_count is not None:
        cr = args.count_regex or r"\n"
        n = len(re.findall(cr, log))
        if n >= args.min_count:
            reasons.append(f"count /{cr}/ = {n} ≥ {args.min_count}")
        else:
            ok = False
            reasons.append(f"count /{cr}/ = {n} < {args.min_count}")
    if not (args.expect_regex or args.min_count is not None):
        reasons.append("no output criteria given — PASS on exit code only")
    return ("PASS" if ok else "FAIL"), reasons


def run(args) -> int:
    _validate_ref(args.ds_image, "--ds-image")
    if "@sha256:" not in args.ds_image:
        print(f"[WARNING] --ds-image {args.ds_image!r} uses a mutable tag (no digest). "
              "For production, pin by digest: nvcr.io/nvidia/deepstream:<tag>@sha256:<digest>",
              file=sys.stderr)
    _validate_ref(args.distro, "--distro")
    _validate_cmd(args.cmd)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.name):
        raise SystemExit("--name contains unsafe characters")
    if not args.yes:
        raise SystemExit(
            "explicit confirmation required before preparing or uploading a custom command; "
            "show the redacted command, ask the user, then pass --yes"
        )
    pw = W.resolve_connection(args)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RV._private_run_dir(
        args.run_dir, f"{stamp}-custom-{args.name}-")
    print(f"RUN_DIR={run_dir}")

    bundle = _make_bundle(args.cmd, args.input)
    local_bundle = os.path.join(run_dir, "bundle.tgz")
    with open(local_bundle, "wb") as fh:
        fh.write(bundle)
    token = secrets.token_hex(8)
    remote_bundle = rf"C:\dsw\custom-{token}-bundle.tgz"
    remote_job = rf"C:\dsw\custom-{token}-runjob.sh"
    remote_bundle_wsl = f"/mnt/c/dsw/custom-{token}-bundle.tgz"
    remote_job_wsl = f"/mnt/c/dsw/custom-{token}-runjob.sh"
    print("uploading command + inputs …")
    W.upload_binary(
        args.host, args.user, pw, local_bundle, remote_bundle,
        use_https=args.use_https, allow_http=args.allow_http)

    job, remote_out_tgz = _remote_job(
        args.ds_image, args.container_timeout, token,
        remote_bundle_wsl, args.allow_network)
    local_job = os.path.join(run_dir, "runjob.sh")
    with open(local_job, "w", newline="\n") as fh:
        fh.write(job)
    W.upload_binary(
        args.host, args.user, pw, local_job, remote_job,
        use_https=args.use_https, allow_http=args.allow_http)
    print(f"running custom pipeline '{args.name}' in {args.ds_image} "
          f"(Docker inside WSL) …")
    print(f"[WARNING] Remote privileged execution: running user-supplied command on {args.host}")
    print(f"  Host: {args.host}  User: {args.user}  Image: {args.ds_image}")
    safe_cmd = _redact_cmd(args.cmd)
    print(f"  Command: {safe_cmd}")
    print(f"  Network: {'enabled by explicit confirmation' if args.allow_network else 'disabled'}")
    print("  Docker flags: --gpus all, no-new-privileges, cap-drop=ALL")
    # Log the generated remote job script for auditability before execution
    with open(os.path.join(run_dir, "runjob_audit.sh"), "w") as fh:
        fh.write(job)
    print(f"  Audit log: {os.path.join(run_dir, 'runjob_audit.sh')}")
    _, out, err = W._run_ps(
        args.host, args.user, pw,
        f"wsl -d {args.distro} -u root -- bash {remote_job_wsl}",
        args.timeout,
        use_https=getattr(args, "use_https", False),
        allow_http=getattr(args, "allow_http", False))
    with open(os.path.join(run_dir, "orchestration.log"), "w") as fh:
        fh.write(out + "\n---stderr---\n" + err)

    out_tgz = os.path.join(run_dir, "out.tgz")
    try:
        W.download_wsl(args.host, args.user, pw, args.distro,
                       remote_out_tgz, out_tgz, timeout=DOWNLOAD_TIMEOUT_S,  # nosec B108
                       use_https=args.use_https, allow_http=args.allow_http)
        with tarfile.open(out_tgz) as tar:
            RV._safe_extract(tar, run_dir)
    except (SystemExit, Exception) as exc:
        print(f"VERDICT: ❌ FAIL — no output bundle from remote job ({exc}). "
              f"See {os.path.join(run_dir, 'orchestration.log')}")
        return 1

    log_path = os.path.join(run_dir, "out", "custom.log")
    log = ""
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as fh:
            log = fh.read()
    verdict, reasons = _evaluate(log, args)

    lines = ["═" * 64, f" Custom pipeline — {args.name}", "═" * 64,
             f" host: {args.host}   image: {args.ds_image}",
             f" cmd : {safe_cmd[:200]}", "─" * 64]
    for r in reasons:
        lines.append(f"   • {r}")
    lines += ["─" * 64,
              f" VERDICT: {'✅' if verdict == 'PASS' else '❌'} {verdict}",
              f" logs: {run_dir}", "═" * 64]
    term = "\n".join(lines)
    with open(os.path.join(run_dir, "summary.txt"), "w") as fh:
        fh.write(term + "\n")
    with open(os.path.join(run_dir, "result.json"), "w") as fh:
        json.dump({"name": args.name, "host": args.host, "ds_image": args.ds_image,
                   "cmd": safe_cmd, "verdict": verdict, "reasons": reasons,
                   "timestamp": stamp}, fh, indent=2)
    print(term)
    print(f"\nSUMMARY_FILE={os.path.join(run_dir, 'summary.txt')}")
    return 0 if verdict == "PASS" else 1


def build_parser():
    p = argparse.ArgumentParser(description="Run+verify a custom DeepStream pipeline in WSL")
    p.add_argument("--target-profile",
                   help="private target profile created by configure_target.py")
    p.add_argument("--host")
    p.add_argument("--user", default=os.environ.get("WINRM_USER"),
                   help="Windows account or $WINRM_USER")
    p.add_argument("--pass-file", default=W.DEFAULT_PASS_FILE,
                   help="chmod-600 file holding the Windows password (one line); "
                        "or set $WINRM_PASS_FILE")
    p.add_argument("--distro", default="Ubuntu-24.04")
    p.add_argument("--ds-image", default="nvcr.io/nvidia/deepstream:9.1-triton-multiarch")
    p.add_argument("--name", default="custom", help="short name for the check / run dir")
    p.add_argument("--cmd", required=True,
                   help="pipeline command to run in the container (cwd=/work)")
    p.add_argument("--input", action="append", default=[],
                   help="local file to stage into the container /work (repeatable)")
    p.add_argument("--expect-regex", action="append", default=[],
                   help="pattern that must appear in the output (repeatable; all required)")
    p.add_argument("--count-regex", default=None,
                   help="pattern to count for --min-count (e.g. a per-frame marker)")
    p.add_argument("--min-count", type=int, default=None,
                   help="require at least N matches of --count-regex in the output")
    p.add_argument("--allow-nonzero", action="store_true",
                   help="do not require exit code 0")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--yes", action="store_true",
                   help="confirm remote privileged execution without interactive prompt")
    p.add_argument("--allow-network", action="store_true",
                   help="enable container network after separate user confirmation")
    p.add_argument("--use-https", action="store_true",
                   help="WinRM over HTTPS port 5986 (legacy explicit metadata mode)")
    p.set_defaults(allow_http=False)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="WinRM operation timeout (s)")
    p.add_argument("--container-timeout", type=int, default=CONTAINER_TIMEOUT_S,
                   help="hard kill for the container run (s)")
    p.set_defaults(func=run)
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    raise SystemExit(a.func(a))
