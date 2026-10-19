#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Download one Hugging Face file from an immutable revision.
# Usage: hf-download-file.sh ORG MODEL REMOTE_PATH DEST_PATH COMMIT_SHA [--optional]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hf-curl.sh"

HF_ORG="${1:-}"
MODEL_NAME="${2:-}"
REMOTE_PATH="${3:-}"
DEST="${4:-}"
REVISION="${5:-${HF_REVISION:-}}"
OPTIONAL="${6:-}"

if [[ -z "$HF_ORG" || -z "$MODEL_NAME" || -z "$REMOTE_PATH" || -z "$DEST" ]]; then
    echo "Usage: $0 <HF_ORG> <MODEL_NAME> <REMOTE_PATH> <DEST_PATH> <commit-sha> [--optional]" >&2
    exit 1
fi
for arg_name in HF_ORG MODEL_NAME; do
    value="${!arg_name}"
    if ! [[ "$value" =~ ^[A-Za-z0-9._/-]+$ ]]; then
        echo "ERROR: $arg_name contains invalid characters: $value" >&2
        exit 1
    fi
done
if [[ "$REMOTE_PATH" = /* || "$REMOTE_PATH" == *..* ]] ||
   ! [[ "$REMOTE_PATH" =~ ^[A-Za-z0-9._/-]+$ ]]; then
    echo "ERROR: REMOTE_PATH must be a safe relative repository path" >&2
    exit 1
fi
case "$DEST" in
    /*)
        echo "ERROR: DEST_PATH must be relative: $DEST" >&2
        exit 1
        ;;
    *..*)
        echo "ERROR: DEST_PATH contains '..': $DEST" >&2
        exit 1
        ;;
esac
if [[ "$OPTIONAL" != "" && "$OPTIONAL" != "--optional" ]]; then
    echo "ERROR: sixth argument must be --optional" >&2
    exit 1
fi
if ! [[ "$REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: REVISION must be an immutable 40-character commit SHA" >&2
    exit 1
fi

URL="https://huggingface.co/${HF_ORG}/${MODEL_NAME}/resolve/${REVISION}/${REMOTE_PATH}"
mkdir -p "$(dirname "$DEST")"
PART="${DEST}.part.$$"
trap 'rm -f -- "$PART"' EXIT
CURL_OPTS=(-fsSL --proto '=https' --tlsv1.2 --max-time 300 -o "$PART")
if ! hf_curl "${CURL_OPTS[@]}" "$URL"; then
    if [[ "$OPTIONAL" == "--optional" ]]; then
        echo "Optional file unavailable: $REMOTE_PATH" >&2
        exit 0
    fi
    echo "ERROR: download failed: ${HF_ORG}/${MODEL_NAME}/${REMOTE_PATH}@${REVISION}" >&2
    exit 1
fi
if [[ ! -s "$PART" ]]; then
    echo "ERROR: downloaded file is empty: $REMOTE_PATH" >&2
    exit 1
fi
mv -- "$PART" "$DEST"
trap - EXIT
echo "Downloaded: $DEST"
