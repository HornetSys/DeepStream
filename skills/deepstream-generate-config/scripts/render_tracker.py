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

"""Render a custom tracker YAML (NvDCF / NvSORT / IOU) from spec overrides.

Strategy: load the stock-equivalent template line-by-line, substitute values
for a small set of overridable keys, preserve comments and the
`%YAML:1.0` directive that NvDCF's OpenCV parser requires.

Library module imported by render.py; also runnable standalone.

Usage:

    python3 render_tracker.py --algorithm NvDCF \\
        --max-targets-per-stream 200 --probation-age 3 \\
        --out my_tracker.yml

Arguments:
    ``--algorithm``, supported tracker override flags, and required ``--out``.

Output:
    Writes one tracker YAML file and reports its path on standard error.

Exit codes:
    0 on success, 1 for a handled write failure, and 2 for invalid
    command-line arguments.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent.parent / "assets" / "templates"


# Keys allowed to be overridden via spec.tracker.tracker_config.
# Anything else must be done by hand-editing the generated file.
OVERRIDABLE_KEYS: set[str] = {
    "minDetectorConfidence",
    "minTrackerConfidence",
    "maxTargetsPerStream",
    "minIouDiff4NewTarget",
    "probationAge",
    "maxShadowTrackingAge",
    "earlyTerminationAge",
    "useUniqueID",
    "enableBboxUnClipping",
    "preserveStreamUpdateOrder",
    "checkClassMatch",
    "useColorNames",
    "useHog",
    "filterLr",
    "gaussianSigma",
    "stateEstimatorType",
    "associationMatcherType",
    "visualTrackerType",
}


def _canonicalise(key: str) -> str | None:
    """Match a user-supplied key (snake_case or camelCase, any case) to the
    canonical camelCase key from OVERRIDABLE_KEYS. Returns None if unknown.

    Handles acronyms like `use_unique_id` -> `useUniqueID` and
    `min_iou_diff4_new_target` -> `minIouDiff4NewTarget` by collapsing both
    sides to alnum-lowercase before comparing.
    """
    norm = "".join(c.lower() for c in key if c.isalnum())
    for canonical in OVERRIDABLE_KEYS:
        if "".join(c.lower() for c in canonical if c.isalnum()) == norm:
            return canonical
    return None


def build_tracker(algorithm: str, overrides: dict) -> str:
    """Returns the rendered YAML text."""
    if algorithm not in ("NvDCF", "NvSORT", "IOU"):
        sys.stderr.write(f"[render_tracker] unknown algorithm {algorithm!r}; defaulting to NvDCF\n")
        algorithm = "NvDCF"
    template_path = TEMPLATES / f"tracker_{algorithm}.yml"
    text = template_path.read_text(encoding="utf-8")

    # Normalise override keys: accept both snake_case and camelCase, ignore acronym case.
    norm: dict[str, object] = {}
    for k, v in (overrides or {}).items():
        canonical = _canonicalise(k)
        if canonical is None:
            sys.stderr.write(f"[render_tracker] ignoring unknown override key {k!r}\n")
            continue
        norm[canonical] = v

    # Line-level substitution. Pattern matches:  "<indent><key>: <value><opt comment>"
    out_lines: list[str] = []
    for line in text.splitlines():
        substituted = False
        for key, new_val in norm.items():
            m = re.match(rf"^(\s+){re.escape(key)}:\s*([^#\s]+)(\s*#.*)?$", line)
            if m:
                indent, _, trailing = m.groups()
                trailing = trailing or ""
                out_lines.append(f"{indent}{key}: {new_val}{trailing}")
                substituted = True
                break
        if not substituted:
            out_lines.append(line)
    return "\n".join(out_lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Render a custom tracker YAML.")
    ap.add_argument("--algorithm", default="NvDCF", choices=["NvDCF", "NvSORT", "IOU"])
    for k in sorted(OVERRIDABLE_KEYS):
        cli_flag = re.sub(r"([A-Z])", r"-\1", k).lower().lstrip("-")
        ap.add_argument(f"--{cli_flag}", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    argument_values = vars(args)
    overrides: dict[str, object] = {}
    for k in OVERRIDABLE_KEYS:
        cli_name = re.sub(r"([A-Z])", r"_\1", k).lower().lstrip("_")
        v = argument_values.get(cli_name)
        if v is not None:
            overrides[k] = v
    text = build_tracker(args.algorithm, overrides)
    out_path = Path(args.out).expanduser()
    try:
        out_path.write_text(text, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        sys.stderr.write(f"[render_tracker] unable to write {out_path}: {exc}\n")
        return 1
    sys.stderr.write(f"[render_tracker] wrote {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
