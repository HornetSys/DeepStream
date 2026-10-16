# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Keep classification input geometry and bounded encoder choice explicit."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/multitask/deepstream/run_type_adapter_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_type_adapter_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class _Factory:
    @staticmethod
    def find(name: str):
        return object() if name in {"nvv4l2h264enc", "theoraenc", "oggmux"} else None


class _Gst:
    ElementFactory = _Factory


class TestClassifierRunnerOptions(unittest.TestCase):
    def test_small_square_input_preserves_geometry(self):
        description = RUNNER._pipeline_description("theora", mux_width=96, mux_height=96)
        self.assertIn("width=96 height=96", description)
        self.assertIn("theoraenc ! oggmux", description)
        self.assertNotIn("width=1280 height=720", description)

    def test_default_keeps_legacy_geometry(self):
        self.assertIn("width=1280 height=720", RUNNER._pipeline_description("h264"))

    def test_explicit_software_fallback_changes_sink_only(self):
        self.assertEqual(RUNNER._choose_encoder(_Gst, "auto")[0], "h264")
        self.assertEqual(RUNNER._choose_encoder(_Gst, "theora"), ("theora", "rendered-video.ogv"))
        self.assertIn("nvinfer name=infer", RUNNER._pipeline_description("theora"))


if __name__ == "__main__":
    unittest.main()
