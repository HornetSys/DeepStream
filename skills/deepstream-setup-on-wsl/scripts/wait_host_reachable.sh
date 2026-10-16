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

# Poll a host and port until it accepts TCP connections or the timeout expires.
#
# Usage:
#   wait_host_reachable.sh --host <ip_or_alias> [--port 5986] \
#     [--timeout 900] [--interval 15]
#
# Output: REACHABLE=true (exit 0) or REACHABLE=false (exit 1).

set -uo pipefail

HOST=""
PORT="5986"
TIMEOUT=900
INTERVAL=15

die() { echo "ERROR: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)     HOST="${2:-}"; shift 2 ;;
    --port)     PORT="${2:-}"; shift 2 ;;
    --timeout)  TIMEOUT="${2:-}"; shift 2 ;;
    --interval) INTERVAL="${2:-}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$HOST" ]] || die "--host is required"
[[ "$HOST" =~ ^[A-Za-z0-9._@:-]{1,255}$ ]] || die "invalid host: $HOST"
[[ "$PORT" =~ ^[0-9]{1,5}$ ]] || die "invalid port: $PORT"
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || die "invalid timeout"
# interval must be >= 1, else `elapsed` never advances and the loop never times out
[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || die "invalid interval (must be >= 1)"

# Single TCP-connect probe with a short per-attempt timeout. Prefer nc; fall
# back to bash's /dev/tcp when nc is unavailable.
probe() {
  # Use nc with a hard 5-second deadline so elapsed time tracks real time.
  # /dev/tcp has no per-connect timeout and can block indefinitely on dead hosts.
  if command -v nc >/dev/null 2>&1; then
    nc -z -w 5 "$HOST" "$PORT" >/dev/null 2>&1
  else
    timeout 5 bash -c 'exec 3<>"/dev/tcp/$1/$2"' _ "$HOST" "$PORT" >/dev/null 2>&1
  fi
}

elapsed=0
while true; do
  if probe; then
    echo "REACHABLE=true"
    exit 0
  fi
  if [[ "$elapsed" -ge "$TIMEOUT" ]]; then
    echo "REACHABLE=false"
    exit 1
  fi
  echo "waiting for $HOST:$PORT … ${elapsed}s/${TIMEOUT}s" >&2
  sleep "$INTERVAL"
  elapsed=$((elapsed + INTERVAL))
done
