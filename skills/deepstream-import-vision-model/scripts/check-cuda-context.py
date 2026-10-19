#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exit nonzero if this environment cannot initialize a CUDA device context."""

from __future__ import annotations

import ctypes
import sys


def check() -> tuple[bool, str]:
    try:
        runtime = ctypes.CDLL("libcudart.so")
    except OSError as exc:
        return False, f"CUDA runtime library unavailable: {exc}"
    runtime.cudaFree.argtypes = [ctypes.c_void_p]
    runtime.cudaFree.restype = ctypes.c_int
    runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
    runtime.cudaGetErrorString.restype = ctypes.c_char_p
    code = runtime.cudaFree(None)
    if code:
        message = runtime.cudaGetErrorString(code)
        detail = message.decode("utf-8", errors="replace") if message else "unknown CUDA error"
        return False, f"CUDA context initialization failed ({code}): {detail}"
    return True, "CUDA context initialized"


if __name__ == "__main__":
    ok, message = check()
    print(message, file=sys.stdout if ok else sys.stderr)
    raise SystemExit(0 if ok else 1)
