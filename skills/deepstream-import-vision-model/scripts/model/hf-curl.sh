#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Shared HF transport. Credential values go through stdin, never child argv.
hf_curl() {
    set +x
    if [[ -n "${HF_TOKEN:-}" ]]; then
        if ! [[ "$HF_TOKEN" =~ ^[A-Za-z0-9._~-]+$ ]]; then
            echo "ERROR: HF_TOKEN contains unsupported header characters" >&2
            return 2
        fi
        # curl strips Authorization on cross-host redirects by default. Never
        # use --location-trusted; allow only HTTPS redirect destinations.
        printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" |
            curl --config - --proto-redir '=https' "$@"
    else
        curl --proto-redir '=https' "$@"
    fi
}
