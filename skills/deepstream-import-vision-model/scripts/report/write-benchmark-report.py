#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for evidence-backed Step 8 reporting.

Usage: python3 write-benchmark-report.py models/<case>/case-evidence.json

Delegate to the shared manifest-driven generator rather than maintaining a
second template with guessed precision, dimensions, or hard-coded PASS values.
"""
import importlib.util
from pathlib import Path


def main():
    path = Path(__file__).with_name("generate-model-report.py")
    spec = importlib.util.spec_from_file_location("model_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
