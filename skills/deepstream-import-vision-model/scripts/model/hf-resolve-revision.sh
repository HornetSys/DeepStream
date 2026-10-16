#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Resolve a Hugging Face branch/tag/revision to an immutable commit SHA.
# Usage: hf-resolve-revision.sh <HF_ORG> <MODEL_NAME> [revision]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hf-curl.sh"

HF_ORG="${1:-}"
MODEL_NAME="${2:-}"
REQUESTED="${3:-${HF_REVISION:-main}}"

if [[ -z "$HF_ORG" || -z "$MODEL_NAME" || -z "$REQUESTED" ]]; then
    echo "Usage: $0 <HF_ORG> <MODEL_NAME> [revision]" >&2
    exit 2
fi
for arg_name in HF_ORG MODEL_NAME REQUESTED; do
    value="${!arg_name}"
    if ! [[ "$value" =~ ^[A-Za-z0-9._/-]+$ ]]; then
        echo "ERROR: $arg_name contains invalid characters: $value" >&2
        exit 2
    fi
done

URL="https://huggingface.co/api/models/${HF_ORG}/${MODEL_NAME}/revision/${REQUESTED}"
CURL_OPTS=(-fsSL --proto '=https' --tlsv1.2 --max-time 30)
RESPONSE=$(hf_curl "${CURL_OPTS[@]}" "$URL") || {
    echo "ERROR: unable to resolve Hugging Face revision $REQUESTED" >&2
    exit 1
}

python3 - "$RESPONSE" <<'PYEOF'
import json
import re
import sys

try:
    data = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:
    print(f"ERROR: invalid Hugging Face revision response: {exc}", file=sys.stderr)
    raise SystemExit(1)
sha = data.get("sha") if isinstance(data, dict) else None
if not isinstance(sha, str) or re.fullmatch(r"[0-9a-fA-F]{40}", sha) is None:
    print("ERROR: Hugging Face response did not contain a 40-character commit SHA", file=sys.stderr)
    raise SystemExit(1)
print(sha.lower())
PYEOF
