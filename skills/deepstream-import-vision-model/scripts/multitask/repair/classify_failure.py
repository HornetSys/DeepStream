#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Classify common import/runtime failures into bounded repair actions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RULES = [
    (r"No module named ['\"]?onnx['\"]?|onnx is not installed", "missing-onnx", "install ONNX in the shared conversion environment; do not create a per-model environment"),
    (r"No module named ['\"]?(transformers|optimum)['\"]?|optimum-cli: (command )?not found", "missing-exporter", "use the explicit torch.onnx.export wrapper, or install and import-check a compatible Optimum/Transformers pair in a caller-owned route environment"),
    (r"ForeignNode|Unsupported operator|UNSUPPORTED_NODE", "unsupported-operator", "simplify/export with an explicit opset and inspect the offending node before retrying TensorRT"),
    (r"does not match network definition|optimization.?profile.*(?:mismatch|invalid)|profile\s+\d+.*dimension", "profile-mismatch", "compare the graph-derived input names, rank, dtype, and dimensions with min/opt/max; correct the profile contract and rebuild once"),
    (r"dynamic.*(shape|dimension)|setDimensions", "unresolved-shape", "provide explicit NCHW input shape or build a TensorRT optimization profile"),
    (r"output-tensor-meta|nvdspostprocess|custom.*parser", "postprocess-contract", "verify output-tensor-meta=1, tensor names, and the task-specific parser/library contract"),
    (r"CUDA (error|failure|driver)|cudaError|no CUDA-capable device|CUDA driver version is insufficient|failed to initialize (CUDA|the CUDA driver)", "runtime-capability", "stop and report the GPU/driver/DeepStream capability mismatch; do not silently fall back to CPU"),
    (r"404 Not Found|repository root|README|catalog page|no exact artifact|SHA-256 mismatch", "artifact-resolution", "resolve one exact downloadable artifact and verify its SHA-256; otherwise report BLOCKED"),
    (r"per-channel std|channel-specific scaling", "preprocess-adapter", "generate the adapter-owned normalized input tensor, validate it against the reference preprocessor, and keep ADAPTER_REQUIRED until runtime and visual evidence pass"),
    (r"out of memory|CUDA_ERROR_OUT_OF_MEMORY|Workspace.*insufficient", "memory-capability", "try one smaller valid profile or a higher-memory build host; otherwise report BLOCKED"),
    (r"FP16.*(non-finite|parity|drift)|NaN|Inf", "precision-parity", "preserve the FP16 failure, retry with TF32 disabled, then FP32, and report the passing precision"),
    (r"not-negotiated|YUV444|unsupported.*(codec|profile|pixel format)|decoder.*failed", "media-capability", "preserve the source and create one hashed H.264/YUV420 derivative for the same semantic fixture"),
    (r"ground.?truth|semantic oracle|oracle.*missing", "semantic-untested", "report semantic UNTESTED and obtain a fixed reviewed oracle before claiming PASS"),
    (r"zero detections|empty output|accuracy", "parity-failure", "compare fixed fixtures source→ONNX→TensorRT→DeepStream and correct preprocessing before changing thresholds"),
]


def classify(text: str) -> dict[str, str]:
    """Classify the first causal error in the log, not the first rule that matches anywhere."""

    for line in text.splitlines():
        for pattern, code, action in RULES:
            if re.search(pattern, line, flags=re.IGNORECASE):
                return {"code": code, "action": action, "evidence": line.strip()[:300]}
    return {"code": "unknown", "action": "capture the smallest reproducible log and inspect the first causal error before retrying", "evidence": ""}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="log file")
    args = parser.parse_args()
    result = classify(args.log.read_text(encoding="utf-8", errors="replace"))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
