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

"""End-to-end DeepStream WSL verification driver (runs from the LOCAL machine).

Bundles the runner + pipelines, uploads to the remote Windows box, runs every
pipeline in the DeepStream container (inside Docker, inside WSL), pulls all
results + logs back into a local run directory, then writes a structured
results.json + summary.md and prints a readable PASS/FAIL summary.

Add new verifications by dropping a file in pipelines/ (see pipelines/README.md).

Example:
  python scripts/run_verification.py --target-profile <PRIVATE_PROFILE> \\
    --ds-image nvcr.io/nvidia/deepstream:9.1-triton-multiarch --label ds-9.1

Usage:
  python scripts/run_verification.py --host HOST --user USER \\
    --ds-image nvcr.io/nvidia/deepstream:9.1-triton-multiarch --label ds-9.1

Arguments:
  --host HOST              Windows host IP
  --user USER              Windows local-admin username
  --ds-image DS_IMAGE      DeepStream container image to verify
  --label LABEL            Tag for this run (e.g. ds-9.1)
  --timeout TIMEOUT        Max seconds to wait for pipeline completion
  --run-dir RUN_DIR        Local dir for logs (default: /tmp/<skill>/runs/<ts>)

Output:
  Prints PASS/FAIL summary; writes summary.txt + results.json to --run-dir.

Exit codes:
  0  All pipelines PASS
  1  One or more pipelines FAIL or no results bundle produced
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import secrets
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import winrm_exec as W  # noqa: E402

# Named constants — avoids magic numbers scattered through the script
CONTAINER_TIMEOUT_S = 1500   # max seconds for the DS container to run all pipelines
DOWNLOAD_TIMEOUT_S = 600     # max seconds to pull the results bundle from WSL
DEFAULT_SCRIPT_TIMEOUT_S = 1800  # --timeout default
PIPELINE_TIMEOUT_S = 180     # per-pipeline timeout inside the container (env: PIPELINE_TIMEOUT)
DOCKERD_WAIT_ITERS = 25      # max iterations waiting for dockerd to become ready

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_NAME = os.path.basename(SKILL_DIR)
# Local run logs live under /tmp/<skill-name>/runs/ (out of the repo tree).
DEFAULT_RUNS_DIR = os.path.join("/tmp", SKILL_NAME, "runs")  # nosec B108


def _private_run_dir(requested: str | None, prefix: str) -> str:
    base = Path(DEFAULT_RUNS_DIR)
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(base, 0o700)
    if requested:
        raw_path = Path(requested).expanduser()
        if raw_path.exists() and raw_path.is_symlink():
            raise SystemExit("--run-dir must not be a symlink")
        path = raw_path.resolve()
        if not path.is_relative_to(Path("/tmp").resolve()):
            raise SystemExit("--run-dir must be under /tmp")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
        return str(path)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def _make_bundle(pipelines_dir: str, runner_path: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(runner_path, arcname="verify_runner.sh")
        tar.add(pipelines_dir, arcname="pipelines")
    return buf.getvalue()


def _remote_job(ds_image: str, out_tgz: str, token: str, bundle_wsl: str) -> str:
    # runs inside WSL as root: extract bundle, ensure dockerd, run container, tar results
    _work = f"/tmp/dsverify-{token}"  # bounded, unique WSL workspace  # nosec B108
    _out_tgz = out_tgz  # per-run unique path (randomised by caller)
    return f"""set -e
rm -f {_out_tgz}   # drop any stale bundle so a failed run can't return old results
rm -rf {_work} && mkdir -p {_work}/out/logs
tar xzf {bundle_wsl} -C {_work}
if ! sudo docker info >/dev/null 2>&1; then sudo nohup dockerd >/var/log/dockerd.log 2>&1 </dev/null & fi
for _ in $(seq 1 {DOCKERD_WAIT_ITERS}); do sudo docker info >/dev/null 2>&1 && break; sleep 1; done
codec_mounts=()
for lib in /usr/lib/wsl/lib/libnvcuvid.so.1 /usr/lib/wsl/lib/libnvidia-encode.so.1; do
  if [[ -f "$lib" ]]; then
    base=$(basename "$lib")
    codec_mounts+=(--mount "type=bind,src=$lib,dst=/usr/lib/x86_64-linux-gnu/$base,readonly")
  fi
done
# Kill only a stale container from this randomized run.
stale=$(sudo docker ps -q --filter label=dsverify={token}); if [ -n "$stale" ]; then sudo docker kill $stale >/dev/null 2>&1; fi
# Bound the whole container so an unkillable element can't hang the run forever.
# Non-fatal (set +e) so results are always collected even on failure/timeout.
set +e
timeout --signal=KILL {CONTAINER_TIMEOUT_S} \
sudo docker run --rm --label dsverify={token} --gpus all --network none \
  --security-opt no-new-privileges --cap-drop=ALL \
  "${{codec_mounts[@]}}" \
  -e PIPELINE_TIMEOUT={PIPELINE_TIMEOUT_S} \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  -e PIPELINES_DIR=/verify/pipelines -e OUT=/out \
  -v {_work}:/verify -v {_work}/out:/out \
  {ds_image} bash /verify/verify_runner.sh > {_work}/out/logs/container.stdout 2>&1
echo "DOCKER_RC=$?"
stale=$(sudo docker ps -q --filter label=dsverify={token}); if [ -n "$stale" ]; then sudo docker kill $stale >/dev/null 2>&1; fi
set -e
tar czf {_out_tgz} -C {_work} out
echo "RESULTS_BUNDLED={_out_tgz}"
"""


def _safe_extract(tar: tarfile.TarFile, dest: str) -> None:
    # The tar comes from the remote verification host, so guard against absolute
    # paths, ../ traversal, and links. Python 3.11.4+ has the hardened 'data'
    # filter (PEP 706); use it when present, else apply the same checks manually
    # so older 3.9-3.11.3 interpreters (where filter= raises TypeError) still work.
    if sys.version_info >= (3, 11, 4):
        tar.extractall(dest, filter="data")
        return
    dest_abs = os.path.abspath(dest)
    for m in tar.getmembers():
        target = os.path.abspath(os.path.join(dest, m.name))
        if target != dest_abs and not target.startswith(dest_abs + os.sep):
            sys.exit(f"unsafe path in results archive: {m.name!r}")
        if m.islnk() or m.issym():
            sys.exit(f"unsafe link in results archive: {m.name!r}")
        if not (m.isfile() or m.isdir()):
            sys.exit(f"non-regular member in results archive: {m.name!r} (type={m.type})")
    tar.extractall(dest)  # noqa: S202  # nosec B202 - members validated safe above


def _validate_ref(value: str, kind: str) -> str:
    # ds-image / distro are interpolated into a root shell command — restrict to a
    # safe charset so a crafted value can't inject shell (no spaces/;/&/$/backticks).
    if not re.fullmatch(r"[A-Za-z0-9._:/@-]{1,255}", value or ""):
        sys.exit(f"invalid {kind} {value!r}: allowed chars are [A-Za-z0-9._:/@-]")
    return value


def run(args) -> int:
    _validate_ref(args.ds_image, "--ds-image")
    if "@sha256:" not in args.ds_image:
        print(f"[WARNING] --ds-image {args.ds_image!r} uses a mutable tag (no digest). "
              "For production, pin the image by digest.", file=sys.stderr)
    _validate_ref(args.distro, "--distro")
    if args.label and not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", args.label):
        raise SystemExit("--label contains unsafe characters")
    pw = W.resolve_connection(args)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = _private_run_dir(
        args.run_dir,
        stamp + (f"-{args.label}" if args.label else "") + "-",
    )
    print(f"RUN_DIR={run_dir}")

    # 1) bundle + upload
    bundle = _make_bundle(args.pipelines_dir,
                          os.path.join(SKILL_DIR, "scripts", "verify_runner.sh"))
    local_bundle = os.path.join(run_dir, "bundle.tgz")
    with open(local_bundle, "wb") as fh:
        fh.write(bundle)
    token = secrets.token_hex(8)
    remote_bundle = rf"C:\dsw\verify-{token}-bundle.tgz"
    remote_job = rf"C:\dsw\verify-{token}-runjob.sh"
    remote_bundle_wsl = f"/mnt/c/dsw/verify-{token}-bundle.tgz"
    remote_job_wsl = f"/mnt/c/dsw/verify-{token}-runjob.sh"
    print("uploading bundle …")
    W.upload_binary(
        args.host, args.user, pw, local_bundle, remote_bundle,
        use_https=args.use_https, allow_http=args.allow_http)

    # 2) run job in WSL (upload the job script, execute it)
    remote_out_tgz = f"/tmp/dsverify_out_{secrets.token_hex(8)}.tgz"  # nosec B108 - randomized per run
    job = _remote_job(args.ds_image, remote_out_tgz, token, remote_bundle_wsl)
    local_job = os.path.join(run_dir, "runjob.sh")
    with open(local_job, "w", newline="\n") as fh:
        fh.write(job)
    W.upload_binary(
        args.host, args.user, pw, local_job, remote_job,
        use_https=args.use_https, allow_http=args.allow_http)
    print(f"running {len(os.listdir(args.pipelines_dir))} pipeline files in "
          f"{args.ds_image} (inside Docker inside WSL) …")
    print(f"[WARNING] Executing remotely on {args.host} as {args.user}: "
          f"WSL root shell via WinRM, then Docker container with GPU access.")
    print(f"  Image: {args.ds_image}")
    print(f"  Distro: {args.distro}  Timeout: {args.timeout}s")
    rc, out, err = W._run_ps(
        args.host, args.user, pw,
        f'wsl -d {args.distro} -u root -- bash {remote_job_wsl}',
        args.timeout,
        use_https=getattr(args, "use_https", False),
        allow_http=getattr(args, "allow_http", False))
    with open(os.path.join(run_dir, "orchestration.log"), "w") as fh:
        fh.write(out + "\n---stderr---\n" + err)

    # 3) pull results bundle back and extract. If the remote job never produced
    # the bundle (e.g. dockerd never came up), don't crash with no output — the
    # orchestration.log is already saved locally; report a clear failure.
    out_tgz = os.path.join(run_dir, "out.tgz")
    try:
        W.download_wsl(args.host, args.user, pw, args.distro,
                       remote_out_tgz, out_tgz, timeout=DOWNLOAD_TIMEOUT_S,
                       use_https=args.use_https, allow_http=args.allow_http)
        with tarfile.open(out_tgz) as tar:
            _safe_extract(tar, run_dir)
    except (Exception, SystemExit) as exc:
        print(f"OVERALL: ❌ FAIL — no results bundle from remote job ({exc}). "
              f"See {os.path.join(run_dir, 'orchestration.log')}")
        return 1
    out_dir = os.path.join(run_dir, "out")

    # 4) parse results + environment
    results = []
    rf = os.path.join(out_dir, "results.jsonl")
    if os.path.exists(rf):
        with open(rf) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    # attach the exact command each pipeline recorded (<id>.cmd), for transparency
    for r in results:
        cmd_path = os.path.join(out_dir, "logs", f"{r.get('id', '')}.cmd")
        if os.path.exists(cmd_path):
            with open(cmd_path) as fh:
                r["command"] = fh.read().strip()
    env = {}
    ef = os.path.join(out_dir, "environment.txt")
    if os.path.exists(ef):
        with open(ef) as fh:
            for line in fh:
                if ":" in line:
                    k, v = line.split(":", 1)
                    env[k.strip()] = v.strip()

    summary = _render(args, results, env, run_dir)
    with open(os.path.join(run_dir, "summary.md"), "w") as fh:
        fh.write(summary["md"])
    # plain-text summary — always displayable/cat-able so it's shown to the user
    with open(os.path.join(run_dir, "summary.txt"), "w") as fh:
        fh.write(summary["term"] + "\n")
    with open(os.path.join(run_dir, "results.json"), "w") as fh:
        json.dump({"host": args.host, "ds_image": args.ds_image, "label": args.label,
                   "timestamp": stamp, "environment": env, "results": results,
                   "overall": summary["overall"]}, fh, indent=2)
    print(summary["term"])
    print(f"\nSUMMARY_FILE={os.path.join(run_dir, 'summary.txt')}")
    return 0 if summary["overall"] == "PASS" else 1


def _render(args, results, env, run_dir):
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_skip = sum(1 for r in results if r.get("status") == "SKIP")
    # PASS requires at least one real PASS and zero FAILs — an all-SKIP (or empty)
    # run verified nothing, so it must not report green.
    overall = "PASS" if (n_fail == 0 and n_pass > 0) else "FAIL"
    mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⚪"}

    lines = []
    lines.append("═" * 64)
    lines.append(f" DeepStream WSL verification — {args.label or args.ds_image}")
    lines.append("═" * 64)
    if env:
        for k in ("gpu", "deepstream", "cuda"):
            if env.get(k):
                lines.append(f" {k:<11}: {env[k]}")
    lines.append(f" host       : {args.host}   image: {args.ds_image}")
    lines.append("─" * 64)
    for r in results:
        st = r.get("status", "?")
        lines.append("─" * 64)
        lines.append(f" {mark.get(st,'?')} {st:<4} {r.get('id','')}")
        lines.append(f"    detail : {r.get('detail','')}")
        if r.get("command"):
            lines.append(f"    command: {r['command']}")
    lines.append("─" * 64)
    lines.append(f" TOTAL: {len(results)}   ✅ {n_pass} PASS   ❌ {n_fail} FAIL   ⚪ {n_skip} SKIP")
    lines.append(f" OVERALL: {mark.get(overall)} {overall}")
    lines.append(f" logs: {run_dir}")
    lines.append("═" * 64)
    term = "\n".join(lines)

    md = [f"# DeepStream WSL verification — {args.label or args.ds_image}", ""]
    md.append(f"- **Overall:** {overall}  ({n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP)")
    md.append(f"- **Host:** `{args.host}`  ·  **Image:** `{args.ds_image}`")
    for k, v in env.items():
        md.append(f"- **{k}:** {v}")
    md.append("")
    md.append("| Test | Result | Detail |")
    md.append("| --- | --- | --- |")
    for r in results:
        md.append(f"| `{r.get('id','')}` | {r.get('status','')} | {r.get('detail','')} |")
    md.append("")
    md.append("## Exact commands run (inside the DeepStream container)")
    for r in results:
        if r.get("command"):
            md.append(f"\n**{r.get('id','')}**\n```\n{r['command']}\n```")
    md.append("")
    md.append("Per-test logs under `out/logs/` in this run directory.")
    return {"term": term, "md": "\n".join(md), "overall": overall}


def build_parser():
    p = argparse.ArgumentParser(description="DeepStream WSL verification driver")
    p.add_argument("--target-profile",
                   help="private target profile created by configure_target.py")
    p.add_argument("--host")
    p.add_argument("--user", default=os.environ.get("WINRM_USER"),
                   help="Windows local-admin account or $WINRM_USER")
    p.add_argument("--pass-file", default=W.DEFAULT_PASS_FILE,
                   help="chmod-600 file holding the Windows password (one line); "
                        "or set $WINRM_PASS_FILE")
    p.add_argument("--distro", default="Ubuntu-24.04")
    p.add_argument("--ds-image", default="nvcr.io/nvidia/deepstream:9.1-triton-multiarch")
    p.add_argument("--pipelines-dir", default=os.path.join(SKILL_DIR, "pipelines"))
    p.add_argument("--run-dir", default=None,
                   help="local dir for logs (default /tmp/<skill-name>/runs/<ts>)")
    p.add_argument("--label", default="", help="tag for the run, e.g. ds-9.1")
    p.add_argument("--timeout", type=int, default=DEFAULT_SCRIPT_TIMEOUT_S)
    p.add_argument("--use-https", action="store_true",
                   help="WinRM over HTTPS port 5986 (legacy explicit metadata mode)")
    p.set_defaults(allow_http=False)
    p.set_defaults(func=run)
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    raise SystemExit(a.func(a))
