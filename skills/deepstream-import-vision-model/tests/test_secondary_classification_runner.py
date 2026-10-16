# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""The reusable ROI cascade fails closed when its Python runtime is incomplete."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/multitask/deepstream/run_secondary_classification.py"
SPEC = importlib.util.spec_from_file_location("run_secondary_classification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TestSecondaryClassificationRunner(unittest.TestCase):
    def test_missing_empty_and_nonempty_video_match_payload_and_exit(self):
        for video_bytes, expected in ((None, "FAIL"), (b"", "FAIL"), (b"rendered", "PASS")):
            with self.subTest(video_bytes=video_bytes), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                labels = root / "labels.txt"
                labels.write_text("vehicle\n")
                output = root / "output"

                class Pipeline:
                    def __init__(self, *args):
                        pass

                    def attach(self, name, operator):
                        self.operator = operator
                        return self

                    def start(self):
                        return self

                    def wait(self, video_bytes=video_bytes, output=output):
                        self.operator.frames = 2
                        self.operator.records = [{"label": "vehicle"}]
                        if video_bytes is not None:
                            (output / "rendered-output.ogv").write_bytes(video_bytes)

                runtime = SimpleNamespace(BatchMetadataOperator=object, Pipeline=Pipeline,
                                          Probe=lambda name, operator: operator, osd=SimpleNamespace())
                args = SimpleNamespace(output_dir=str(output))
                stdout = io.StringIO()
                with patch.dict(sys.modules, {"pyservicemaker": runtime,
                     "yaml": SimpleNamespace(safe_dump=lambda *args, **kwargs: None)}), \
                     patch.object(RUNNER, "_validated_paths", return_value=(labels,) * 5), \
                     patch.object(RUNNER, "_property", return_value="2"), \
                     contextlib.redirect_stdout(stdout):
                    code = RUNNER._run_worker(args)
                self.assertEqual(code, 0 if expected == "PASS" else 1)
                self.assertEqual(json.loads(stdout.getvalue())["status"], expected)
                payload = json.loads((output / "structured-output.json").read_text())
                self.assertEqual(payload["status"], expected)

    def test_classifier_label_is_inside_roi_below_detector_title(self):
        self.assertEqual(
            RUNNER._classifier_text_position({"left": 50.8, "top": 80.4}),
            (52, 82),
        )
        self.assertEqual(
            RUNNER._classifier_text_position({"left": -5.0, "top": -1.0}),
            (0, 1),
        )

    def test_missing_dependency_returns_structured_blocked(self):
        argv = [str(SCRIPT), "--video", "video.mp4", "--primary-config", "primary.txt",
                "--secondary-config", "secondary.txt", "--detector-labels", "detector.txt",
                "--classifier-labels", "classifier.txt", "--output-dir", "evidence", "--worker"]
        stderr = io.StringIO()
        with patch.object(sys, "argv", argv), patch.object(RUNNER, "_validated_paths"), \
             patch.object(RUNNER, "_run_worker", side_effect=ImportError("No module named yaml")), \
             contextlib.redirect_stderr(stderr):
            self.assertEqual(RUNNER.main(), 4)
        self.assertEqual(json.loads(stderr.getvalue())["status"], "BLOCKED")
        self.assertIn("yaml", json.loads(stderr.getvalue())["reason"])


if __name__ == "__main__":
    unittest.main()
