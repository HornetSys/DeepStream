# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify stream-capacity tables from logs, not caller-provided FPS estimates."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report/benchmark-sections.py"
SPEC = importlib.util.spec_from_file_location("benchmark_sections", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class BenchmarkSectionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = self.root / "final.engine"
        def write(name, text):
            p = self.root / name
            p.write_text(text)
            return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        self.write = write
        engine = write("final.engine", "engine")
        def trt(batch):
            text = (f"--loadEngine={self.engine}\nData transfers: Disabled\n"
                    f"Input inference shape : input={batch}x3x8x8\n"
                    "Throughput: 100 qps\nGPU Compute Time: mean = 1 ms, percentile(99%) = 2 ms\n"
                    "&&&& PASSED TensorRT.trtexec\n")
            return {"batch": batch, "log": write(f"trt{batch}.log", text)}
        def ds(key, fps):
            return {"streams": 2, "batch": 2, "exit_code": 0,
                    "config": write(key+".ini", f"[property]\nmodel-engine-file={self.engine}\nbatch-size=2\n"),
                    "log": write(key+".log", (f"**PERF: 62.0 (60.0) {fps} ({fps})\n")*5)}
        self.manifest = {"case_id": "example", "benchmark": {"engine": engine, "input_unit": "frame",
            "scope": "decode-mux-infer-parser-fakesink", "trtexec": {"bs1": trt(1), "bsmax": trt(2)},
            "deepstream": {"run1": ds("run1", 24), "run2": ds("run2", 32)}}}

    def test_real_logs_are_summed_and_every_source_is_checked(self):
        data = REPORT.collect(self.manifest, self.root)
        self.assertEqual(data["deepstream"]["run1"]["total_fps"], 86)
        self.assertEqual(data["deepstream"]["run1"]["worst_fps"], 24)
        self.assertEqual(data["deepstream"]["run1"]["status"], "FAIL")
        self.assertEqual(data["deepstream"]["run2"]["status"], "PASS")
        text = REPORT.sections(data)
        self.assertIn("Calibration | 2 | 43.00 | 24.00 | 86.00 | FAIL", text[8])
        self.assertIn("47.00%", text[9])
        self.assertIn("not a proven absolute maximum", text[8])

    def test_roi_throughput_never_becomes_camera_capacity(self):
        self.manifest["benchmark"]["input_unit"] = "roi"
        text = REPORT.sections(REPORT.collect(self.manifest, self.root))
        self.assertIn("N/A", text[6])
        self.assertIn("ROI/s differs from frames/s", text[9])

    def test_tampered_missing_and_wrong_batch_evidence_is_rejected(self):
        for key, value in [("batch", 1), ("exit_code", 1), ("streams", 3)]:
            with self.subTest(key=key):
                original = self.manifest["benchmark"]["deepstream"]["run2"][key]
                self.manifest["benchmark"]["deepstream"]["run2"][key] = value
                with self.assertRaises(ValueError): REPORT.collect(self.manifest, self.root)
                self.manifest["benchmark"]["deepstream"]["run2"][key] = original
        self.engine.write_text("changed engine")
        with self.assertRaisesRegex(ValueError, "digest"): REPORT.collect(self.manifest, self.root)

    def test_incomplete_windows_do_not_become_a_capacity_result(self):
        self.manifest["benchmark"]["deepstream"]["run2"]["log"] = self.write("short.log", "**PERF: 40 (40) 40 (40)\n")
        with self.assertRaisesRegex(ValueError, "insufficient"): REPORT.collect(self.manifest, self.root)

    def test_no_evidence_is_explicitly_untested(self):
        self.assertTrue(all("UNTESTED" in v for v in REPORT.sections(None).values()))

    def test_transfers_wrong_engine_and_wrong_input_batch_are_rejected(self):
        item = self.manifest["benchmark"]["trtexec"]["bsmax"]
        text = Path(item["log"]["path"]).read_text()
        for bad in (text.replace("Disabled", "Enabled"), text.replace(str(self.engine), "other.engine"),
                    text.replace("input=2x", "input=4x")):
            item["log"] = self.write("bad.log", bad)
            with self.assertRaises(ValueError): REPORT.collect(self.manifest, self.root)

    def test_config_engine_mismatch_is_rejected(self):
        self.manifest["benchmark"]["deepstream"]["run2"]["config"] = self.write(
            "wrong.ini", "[property]\nmodel-engine-file=other.engine\nbatch-size=2\n")
        with self.assertRaisesRegex(ValueError, "config engine"): REPORT.collect(self.manifest, self.root)

    def test_unsupported_scope_is_not_a_capacity_measurement(self):
        self.manifest["benchmark"]["scope"] = "two-independent-rendered-pipelines"
        with self.assertRaisesRegex(ValueError, "scope"): REPORT.collect(self.manifest, self.root)

    def test_replacement_preserves_frames_and_gate_status(self):
        path = self.root / "report.md"
        path.write_text("Result FAIL\n" + "\n".join(f"## {n}. Old\n" + ("![frame](source.png)\n" if n == 7 else
            "### Separate task-capacity diagnostic\nRendered FPS: 28\n" if n == 8 else "old\n") for n in range(1, 13)))
        with mock.patch.object(REPORT.subprocess, "run") as charts:
            def generate(*args, **kwargs):
                d = self.root / "charts"
                d.mkdir()
                for name in ("trtexec_bs1_vs_bsmax", "trtexec_throughput", "ds_streams_vs_fps", "trt_vs_ds", "efficiency"):
                    (d / f"chart_{name}.png").write_bytes(b"png")
            charts.side_effect = generate
            REPORT.amend(path, self.manifest, self.root)
        text = path.read_text()
        self.assertIn("Result FAIL", text)
        self.assertIn("![frame](source.png)", text)
        self.assertIn("## 9. trtexec vs DeepStream Comparison", text)
        self.assertIn("Rendered FPS: 28", text)
        self.assertEqual(text.count("!["), 6)


if __name__ == "__main__":
    unittest.main()
