#!/usr/bin/env bash
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

# Run a command or payload inside an Ubuntu WSL distro over SSH.
# Use winrm_exec.py for WinRM so credentials never appear on the command line.
#
# Usage:
#   remote_wsl_exec.sh --host <ssh_alias> --cmd '<command>' --yes
#   remote_wsl_exec.sh --host <ssh_alias> --script <local_payload.sh> --yes [-- arg...]
#
# Exit code is the remote command's exit code.

set -euo pipefail

HOST=""
CMD=""
SCRIPT=""
CONFIRMED=false
declare -a PAYLOAD_ARGS=()

die() { echo "ERROR: $*" >&2; exit 2; }

# --- Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)   HOST="${2:-}"; shift 2 ;;
    --cmd)    CMD="${2:-}"; shift 2 ;;
    --script) SCRIPT="${2:-}"; shift 2 ;;
    --yes)    CONFIRMED=true; shift ;;
    --)       shift; PAYLOAD_ARGS=("$@"); break ;;
    *)        die "unknown argument: $1 (WinRM is handled by winrm_exec.py)" ;;
  esac
done

# --- Validation -------------------------------------------------------------
validate_token() {
  # allow alnum, dot, underscore, hyphen, @, colon — reject shell metachars
  [[ "$1" =~ ^[A-Za-z0-9._@:-]{1,255}$ ]] || die "invalid value: $1"
}

[[ -n "$HOST" ]] || die "--host <ssh_alias> is required"
validate_token "$HOST"
[[ -n "$CMD" || -n "$SCRIPT" ]] || die "provide --cmd or --script"
[[ -z "$CMD" || -z "$SCRIPT" ]] || die "provide only one of --cmd / --script"
[[ -z "$SCRIPT" || -f "$SCRIPT" ]] || die "payload script not found: $SCRIPT"
[[ "$CONFIRMED" == true ]] ||
  die "explicit confirmation required before remote execution; review the command or payload, then pass --yes"

# Require a host key already verified through a trusted channel and stored in
# known_hosts. Never auto-trust a first connection.
SSH_CONNECT_TIMEOUT=15   # seconds; increase for high-latency links
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=${SSH_CONNECT_TIMEOUT} -o StrictHostKeyChecking=yes)

if [[ -n "$CMD" ]]; then
  printf 'REMOTE_EXECUTION=command HOST=%s\n' "$HOST" >&2
  printf 'REMOTE_COMMAND=%q\n' "$CMD" >&2
  exec ssh "${ssh_opts[@]}" "$HOST" "bash -lc $(printf '%q' "$CMD")"
fi

# Payload-script path: use mktemp on the remote to avoid filename injection and races.
printf 'REMOTE_EXECUTION=payload HOST=%s\n' "$HOST" >&2
printf 'REMOTE_PAYLOAD=%q\n' "$SCRIPT" >&2
remote_tmp=$(ssh "${ssh_opts[@]}" "$HOST" 'mktemp /tmp/dswsl.XXXXXX') || die "remote mktemp failed"
scp "${ssh_opts[@]}" -q "$SCRIPT" "$HOST:$remote_tmp" || die "scp of payload failed"
q_args=""
if [[ ${#PAYLOAD_ARGS[@]} -gt 0 ]]; then
  for a in "${PAYLOAD_ARGS[@]}"; do q_args+=" $(printf '%q' "$a")"; done
fi
q_remote_tmp="$(printf '%q' "$remote_tmp")"
rc=0
ssh "${ssh_opts[@]}" "$HOST" "bash $q_remote_tmp$q_args" || rc=$?
ssh "${ssh_opts[@]}" "$HOST" "rm -f $q_remote_tmp" 2>/dev/null || true
exit $rc
