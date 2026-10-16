# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight must test CUDA initialization, not just trtexec's presence."""

from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check-cuda-context.py"
SPEC = importlib.util.spec_from_file_location("check_cuda_context", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class FakeFunction:
    def __init__(self, result):
        self.result = result
        self.argtypes = None
        self.restype = None

    def __call__(self, *_args):
        return self.result


class TestCudaContextPreflight(unittest.TestCase):
    def test_skill_selects_one_gpu_for_container_and_host_preflight(self):
        skill_root = SCRIPT.parents[1]
        workflow = (skill_root / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("(references/runtime-environment.md)", workflow)
        workflow += (skill_root / "references/runtime-environment.md").read_text()
        preflight = (skill_root / "scripts" / "preflight.sh").read_text(encoding="utf-8")
        self.assertIn('--gpus "device=$DS_GPU_DEVICE"', workflow)
        self.assertIn('--gpus "device=$DS_GPU_DEVICE"', preflight)
        self.assertNotIn("--gpus all", workflow)
        self.assertNotIn("--gpus all", preflight)

    def test_unknown_preflight_profile_is_rejected_before_runtime_checks(self):
        script = SCRIPT.parents[1] / "scripts" / "preflight.sh"
        result = subprocess.run(
            ["bash", str(script)],
            env={"DS_PREFLIGHT_PROFILE": "skip-everything"},
            capture_output=True, text=True, check=False, timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid DS_PREFLIGHT_PROFILE", result.stderr)

    def test_context_success(self):
        runtime = type("Runtime", (), {
            "cudaFree": FakeFunction(0),
            "cudaGetErrorString": FakeFunction(b"success"),
        })()
        with patch.object(PROBE.ctypes, "CDLL", return_value=runtime):
            self.assertEqual(PROBE.check(), (True, "CUDA context initialized"))

    def test_incompatible_driver_is_reported(self):
        runtime = type("Runtime", (), {
            "cudaFree": FakeFunction(35),
            "cudaGetErrorString": FakeFunction(b"CUDA driver version is insufficient"),
        })()
        with patch.object(PROBE.ctypes, "CDLL", return_value=runtime):
            ok, reason = PROBE.check()
        self.assertFalse(ok)
        self.assertIn("(35)", reason)
        self.assertIn("insufficient", reason)

    def test_missing_runtime_is_reported(self):
        with patch.object(PROBE.ctypes, "CDLL", side_effect=OSError("libcudart.so absent")):
            ok, reason = PROBE.check()
        self.assertFalse(ok)
        self.assertIn("unavailable", reason)


if __name__ == "__main__":
    unittest.main()
