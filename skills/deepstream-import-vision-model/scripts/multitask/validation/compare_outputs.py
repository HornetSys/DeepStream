#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Compare deterministic `.npy`/`.npz` reference and candidate outputs."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np


def _arrays(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return {"output": loaded}
    try:
        return {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()


def _source_hash(manifest_path: Path, manifest: dict[str, object]) -> str:
    declared = str(manifest.get("source_hash", "")).strip().lower()
    if len(declared) == 64 and all(
        character in "0123456789abcdef" for character in declared
    ):
        return declared
    source_value = str(manifest.get("source", "")).strip()
    if not source_value:
        raise ValueError("manifest must provide source or a SHA-256 source_hash")
    source = Path(source_value)
    if not source.is_absolute():
        source = manifest_path.parent / source
    if not source.is_file():
        raise ValueError(
            "manifest source is unavailable for backend-parity evidence hashing"
        )
    return hashlib.sha256(source.read_bytes()).hexdigest()


def compare(reference: Path, candidate: Path, *, atol: float, rtol: float) -> dict[str, object]:
    left, right = _arrays(reference), _arrays(candidate)
    keys = sorted(set(left) | set(right))
    outputs: dict[str, object] = {}
    passed = True
    for key in keys:
        if key not in left or key not in right:
            outputs[key] = {"status": "FAIL", "detail": "missing output key"}
            passed = False
            continue
        a, b = np.asarray(left[key]), np.asarray(right[key])
        if a.shape != b.shape:
            outputs[key] = {"status": "FAIL", "detail": f"shape {a.shape} != {b.shape}"}
            passed = False
            continue
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            outputs[key] = {"status": "FAIL", "detail": "non-finite output"}
            passed = False
            continue
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        scale = np.maximum(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64)))
        allowed = atol + rtol * scale
        max_abs = float(delta.max()) if delta.size else 0.0
        denom = np.maximum(scale, atol if atol > 0 else np.finfo(np.float64).tiny)
        max_rel = float((delta / denom).max()) if delta.size else 0.0
        ok = bool(np.all(delta <= allowed))
        outputs[key] = {"status": "PASS" if ok else "FAIL", "shape": list(a.shape), "max_abs": max_abs, "max_rel": max_rel}
        passed = passed and ok
    return {
        "status": "PASS" if passed else "FAIL",
        "reference": str(reference),
        "candidate": str(candidate),
        "thresholds": {"atol": atol, "rtol": rtol},
        "outputs": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5, help="FP32 default; pass 1e-3 for the documented FP16 default")
    parser.add_argument("--rtol", type=float, default=1e-4, help="FP32 default; pass 1e-2 for the documented FP16 default")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--input-fixture", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--reference-backend", default="reference")
    parser.add_argument("--candidate-backend", default="candidate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be non-negative")
    try:
        result = compare(args.reference, args.candidate, atol=args.atol, rtol=args.rtol)
        readiness_values = (args.manifest, args.input_fixture, args.campaign_id)
        if any(value not in (None, "") for value in readiness_values):
            if any(value in (None, "") for value in readiness_values):
                raise ValueError(
                    "--manifest, --input-fixture, and --campaign-id are required together"
                )
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("--manifest must contain a JSON object")
            source_hash = _source_hash(args.manifest, manifest)
            input_hash = hashlib.sha256(args.input_fixture.read_bytes()).hexdigest()
            finite = all(
                item.get("detail") != "non-finite output"
                for item in result["outputs"].values()
            )
            result.update(
                {
                    "schema_version": 1,
                    "campaign_id": args.campaign_id,
                    "producer": "compare_outputs.py",
                    "task": str(manifest.get("task", "")),
                    "source_hash": source_hash,
                    "input_fixture_sha256": input_hash,
                    "reference_backend": args.reference_backend,
                    "candidate_backend": args.candidate_backend,
                    "metric": {
                        "name": "raw_tensor_allclose",
                        "passed": result["status"] == "PASS",
                        "finite": finite,
                    },
                }
            )
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "reference": str(args.reference), "candidate": str(args.candidate), "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
