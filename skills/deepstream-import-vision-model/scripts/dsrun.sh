#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved. Apache-2.0.
#
# Convenience wrapper: run a command in the named persistent campaign container.
#
#   bash .claude/skills/deepstream-import-vision-model/scripts/dsrun.sh '<command to run in-container>'
#   e.g. bash .claude/skills/deepstream-import-vision-model/scripts/dsrun.sh 'build/.venv_optimum/bin/python -c "import torch;print(torch.__version__)"'
#        bash .claude/skills/deepstream-import-vision-model/scripts/dsrun.sh 'make -C models/yolov8n/parser && ls models/yolov8n/parser/*.so'
#
# Required: DS_CONTAINER from the digest-pinned campaign setup in SKILL.md.
set -euo pipefail
if [ "$#" -eq 0 ]; then echo "usage: bash .claude/skills/deepstream-import-vision-model/scripts/dsrun.sh '<in-container command>'" >&2; exit 2; fi
if [ -z "${DS_CONTAINER:-}" ]; then
  echo "BLOCKED: set DS_CONTAINER to the named campaign container from SKILL.md" >&2
  exit 4
fi
if [ "$(docker inspect --format='{{.State.Running}}' "$DS_CONTAINER" 2>/dev/null)" != "true" ]; then
  echo "BLOCKED: campaign container is unavailable or stopped: $DS_CONTAINER" >&2
  exit 4
fi
exec docker exec "$DS_CONTAINER" bash -lc "$*"
