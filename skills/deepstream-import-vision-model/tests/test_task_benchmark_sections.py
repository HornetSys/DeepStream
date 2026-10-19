# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Task-capacity report evidence and frame-versus-ROI unit contracts, CPU only."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


# Reuse the sibling verified engine/trtexec fixture in this self-contained skill.
SPEC = importlib.util.spec_from_file_location(
    "task_benchmark_legacy_fixture", Path(__file__).with_name("test_benchmark_sections.py")
)
assert SPEC is not None and SPEC.loader is not None
LEGACY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEGACY)
REPORT = LEGACY.REPORT


class TaskBenchmarkSectionsTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = LEGACY.BenchmarkSectionsTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.write = fixture.write
        runner = LEGACY.SCRIPT.parent.parent / "multitask/deepstream/run_task_capacity.py"
        # Capture once for this fixture attempt. Rebinding result/log hashes
        # after a mutation must not relabel the historical runner snapshot.
        self.runner_sha256 = hashlib.sha256(runner.read_bytes()).hexdigest()
        self.manifest = copy.deepcopy(fixture.manifest)
        self.manifest.update(task="segmentation", campaign_id="reviewed-roi-campaign")
        benchmark = self.manifest["benchmark"]
        benchmark.update(runner="task-capacity", input_unit="roi", scope="task-application-fakesink")
        primary_engine = self.write("primary.engine", "caller-owned primary fixture; never deserialized")
        primary_config = self.write("primary.ini", (
            "[property]\ngie-unique-id=1\nprocess-mode=1\nnetwork-type=0\n"
            f"model-engine-file={primary_engine['path']}\nbatch-size=2\ninterval=0\n"
        ))
        secondary_config = self.write("task.ini", (
            "[property]\ngie-unique-id=2\nprocess-mode=2\nnetwork-type=100\n"
            "operate-on-gie-id=1\n"
            f"model-engine-file={benchmark['engine']['path']}\nbatch-size=2\ninterval=0\n"
        ))
        decoder = self.write("reviewed_decoder.py", "def decode(values, context):\n    return {'fixture': True}\n")
        stages = [
            {"config": primary_config["path"], "config_sha256": primary_config["sha256"],
             "engine": primary_engine["path"], "engine_sha256": primary_engine["sha256"],
             "component_id": 1, "inference_batch": 2, "batch_unit": "frames"},
            {"config": secondary_config["path"], "config_sha256": secondary_config["sha256"],
             "engine": benchmark["engine"]["path"], "engine_sha256": benchmark["engine"]["sha256"],
             "component_id": 2, "inference_batch": 2, "batch_unit": "ROIs"},
        ]
        self.results = {}
        benchmark["deepstream"] = {}
        for key, fps in (("run1", [12.0, 4.0]), ("run2", [12.0, 8.0])):
            windows = [{"duration_seconds": 2, "fps": [9.0, 7.0], "task_records": [36, 28]}
                       for _ in range(25)]
            windows.extend({"duration_seconds": 2, "fps": fps.copy(),
                            "task_records": [int(value * 4) for value in fps]} for _ in range(5))
            counts = [int(sum(window["fps"][i] * 2 for window in windows)) for i in range(2)]
            result = {
                "schema_version": 1, "producer": "deepstream-task-capacity-runner",
                "runner_sha256": self.runner_sha256,
                "task": self.manifest["task"], "campaign_id": self.manifest["campaign_id"],
                "startup": True, "errors": [], "streams": 2, "mux_batch": 2,
                "stages": copy.deepcopy(stages), "decoder": copy.deepcopy(decoder),
                "windows": windows, "duration_seconds": 60,
                "frames_per_source": counts, "task_records_per_source": [count * 2 for count in counts],
                "scope": "decode-mux-complete-inference-cascade-task-postprocessing-fakesink",
                "rendering_included": False, "encoding_included": False,
                "target_fps_per_stream": 5, "status": "FAIL" if key == "run1" else "PASS",
            }
            self.results[key] = result
            benchmark["deepstream"][key] = {"streams": 2, "batch": 2,
                                            "exit_code": 1 if key == "run1" else 0}
            self.bind_result(key)
            self.bind_log(key)

    def bind_result(self, key: str) -> None:
        self.manifest["benchmark"]["deepstream"][key]["result"] = self.write(
            key + "-result.json", json.dumps(self.results[key], allow_nan=False)
        )

    def bind_log(self, key: str, windows: list | None = None) -> None:
        windows = self.results[key]["windows"] if windows is None else windows
        self.manifest["benchmark"]["deepstream"][key]["log"] = self.write(
            key + "-task.log", "SOURCE_LINKED 0\nSOURCE_LINKED 1\n" + "".join(
                "TASK_CAPACITY " + json.dumps(window, allow_nan=False) + "\n" for window in windows
            )
        )

    def collect(self) -> dict:
        return REPORT.collect(self.manifest, self.root)

    def test_verified_two_stage_roi_table_separates_stream_mux_and_task_batch_units(self) -> None:
        data = self.collect()
        self.assertEqual(data["runner"], "task-capacity")
        self.assertEqual(data["input_unit"], "roi")
        self.assertEqual(data["deepstream"]["run2"]["streams"], 2)
        self.assertEqual(data["deepstream"]["run2"]["mux_batch"], 2)
        self.assertEqual([stage["inference_batch"] for stage in data["deepstream"]["run2"]["stages"]], [2, 2])
        self.assertEqual([stage["batch_unit"] for stage in data["deepstream"]["run2"]["stages"]], ["frames", "ROIs"])
        text = REPORT.sections(data)
        self.assertIn("| Streams | Mux batch | Task batch / unit |", text[8])
        self.assertIn("Calibration | 2 | 2 | 2 ROIs | 8.00 | 4.00 | 16.00 | FAIL | FAIL", text[8])
        self.assertIn("Validation | 2 | 2 | 2 ROIs | 10.00 | 8.00 | 20.00 | FAIL | PASS", text[8])
        self.assertIn("ROI batches are not camera counts", text[8])
        self.assertIn("5 FPS does not alter the 30-FPS result", text[8])
        self.assertIn("N/A", text[6])
        self.assertIn("ROI/s differs from frames/s", text[9])
        self.assertIn("ROI/s differs from frames/s", text[10])
        self.assertEqual(data["deepstream"]["run2"]["status"], "FAIL")
        self.assertEqual(data["deepstream"]["run2"]["total_fps"], 20)

    def test_amend_preserves_section_eight_title_once_and_failed_gates(self) -> None:
        markdown = self.root / "task-report.md"
        markdown.write_text("Semantic FAIL; visual UNTESTED\n" + "\n".join(
            f"## {number}. Previous title\nprevious body\n" for number in range(1, 13)
        ), encoding="utf-8")

        def fake_charts(*_args, **_kwargs):
            charts = self.root / "charts"
            charts.mkdir()
            for name in ("trtexec_bs1_vs_bsmax", "trtexec_throughput", "ds_streams_vs_fps", "trt_vs_ds", "efficiency"):
                (charts / f"chart_{name}.png").write_bytes(b"CPU-only chart fixture")

        with patch.object(REPORT.subprocess, "run", side_effect=fake_charts) as chart_call:
            REPORT.amend(markdown, self.manifest, self.root)
        chart_call.assert_called_once()
        text = markdown.read_text(encoding="utf-8")
        self.assertEqual(text.count("## 8. DeepStream Benchmark Results"), 1)
        self.assertIn("## 9. trtexec vs DeepStream Comparison", text)
        self.assertIn("## 10. Efficiency Analysis", text)
        self.assertIn("Semantic FAIL; visual UNTESTED", text)
        self.assertIn("| Streams | Mux batch | Task batch / unit |", text)
        self.assertIn("20.00 | FAIL | PASS", text)
        self.assertEqual(text.count("!["), 5)

    def test_missing_source_value_in_sustained_window_is_rejected(self) -> None:
        self.results["run2"]["windows"][-1]["fps"].pop()
        self.bind_result("run2")
        self.bind_log("run2")
        with self.assertRaisesRegex(ValueError, "incomplete.*source/window"):
            self.collect()

    def test_historical_log_mismatch_is_rejected_even_with_valid_final_windows(self) -> None:
        windows = copy.deepcopy(self.results["run2"]["windows"])
        windows[0]["fps"][0] += 1
        self.bind_log("run2", windows)
        with self.assertRaisesRegex(ValueError, "windows differ"):
            self.collect()

    def test_stage_engine_hash_mismatch_is_rejected(self) -> None:
        self.results["run2"]["stages"][0]["engine_sha256"] = "0" * 64
        self.bind_result("run2")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.collect()

    def test_verified_identical_engine_copy_is_accepted_without_rewriting_logs(self) -> None:
        stage = self.results['run2']['stages'][-1]
        original = Path(stage['engine'])
        relocated = self.write('relocated.engine', original.read_text())
        config = Path(stage['config'])
        relocated_config = self.write('relocated.ini', config.read_text().replace(str(original), relocated['path']))
        stage.update(engine=relocated['path'], engine_sha256=relocated['sha256'],
                     config=relocated_config['path'], config_sha256=relocated_config['sha256'])
        self.bind_result('run2')
        self.assertEqual(self.collect()['deepstream']['run2']['streams'], 2)

    def test_static_input_binding_dimensions_are_verified_not_assumed(self) -> None:
        benchmark = self.manifest['benchmark']
        for key in ('bs1', 'bsmax'):
            item = benchmark['trtexec'][key]
            text = Path(item['log']['path']).read_text()
            shape = f"Input inference shape : input={item['batch']}x3x8x8"
            binding = f"Input binding for input with dimensions {item['batch']}x3x8x8 and type fp32 is created."
            item['log'] = self.write(key+'-static.log', text.replace(shape, binding))
        self.assertEqual(self.collect()['trtexec']['bs1']['batch'], 1)
        item = benchmark['trtexec']['bsmax']
        item['log'] = self.write('wrong-static.log', Path(item['log']['path']).read_text().replace('dimensions 2x', 'dimensions 3x'))
        with self.assertRaisesRegex(ValueError, 'batch differs'):
            self.collect()

    def test_tensor_task_without_decoder_provenance_is_rejected(self) -> None:
        self.results["run2"].pop("decoder")
        self.bind_result("run2")
        with self.assertRaises(ValueError):
            self.collect()

    def test_recorded_eos_errors_are_rejected(self) -> None:
        self.results["run2"]["errors"] = ["unexpected EOS before the bounded capacity interval completed"]
        self.bind_result("run2")
        with self.assertRaisesRegex(ValueError, "valid complete graph"):
            self.collect()

    def test_recorded_stage_batch_must_match_verified_config_batch(self) -> None:
        self.results["run2"]["stages"][-1]["inference_batch"] = 3
        self.bind_result("run2")
        with self.assertRaisesRegex(ValueError, "engine/batch differs"):
            self.collect()

    def test_manifest_source_count_must_match_result_and_log(self) -> None:
        self.manifest["benchmark"]["deepstream"]["run2"]["streams"] = 3
        with self.assertRaises(ValueError):
            self.collect()

    def test_one_stream_is_not_multistream_capacity(self) -> None:
        self.results["run1"]["streams"] = 1
        self.manifest["benchmark"]["deepstream"]["run1"]["streams"] = 1
        self.bind_result("run1")
        self.assertIsNone(self.collect())
        self.assertIn("UNTESTED", REPORT.sections(self.collect())[8])

    def test_missing_per_source_task_execution_cannot_become_capacity(self) -> None:
        self.results["run2"]["task_records_per_source"][1] = 0
        self.bind_result("run2")
        with self.assertRaisesRegex(ValueError, "per-source execution/task evidence"):
            self.collect()

    def test_each_final_window_requires_positive_integer_task_records_for_every_source(self) -> None:
        original = copy.deepcopy(self.results["run2"])
        for invalid in (None, [80], [80, 0], [80, -1], [80, True], [80, "2"]):
            with self.subTest(task_records=invalid):
                self.results["run2"] = copy.deepcopy(original)
                for window in self.results["run2"]["windows"][-5:]:
                    window["fps"] = [40.0, 40.0]
                    window["task_records"] = [160, 160]
                window = self.results["run2"]["windows"][-1]
                if invalid is None:
                    window.pop("task_records")
                else:
                    window["task_records"] = invalid
                self.results["run2"]["status"] = "FAIL"
                self.bind_result("run2")
                self.bind_log("run2")
                with self.assertRaises(ValueError):
                    self.collect()

    def test_legacy_windows_without_producer_evidence_are_untested_or_require_rerun(self) -> None:
        for window in self.results["run2"]["windows"]:
            window.pop("task_records")
            window["fps"] = [40.0, 40.0]
        self.bind_result("run2")
        self.bind_log("run2")
        try:
            data = self.collect()
        except ValueError:
            # Rejecting unverifiable producer evidence requires a fresh run;
            # retaining decode-only measurements as UNTESTED is also truthful.
            return
        self.assertEqual(data["deepstream"]["run2"]["status"], "UNTESTED")
        self.assertIn("UNTESTED", REPORT.sections(data)[8])

    def test_result_identity_and_producer_are_verified(self) -> None:
        original = copy.deepcopy(self.results["run2"])
        for field, wrong in (("producer", "unreviewed-producer"), ("task", "depth"),
                             ("campaign_id", "other-campaign")):
            with self.subTest(field=field):
                self.results["run2"] = {**copy.deepcopy(original), field: wrong}
                self.bind_result("run2")
                with self.assertRaises(ValueError):
                    self.collect()


if __name__ == "__main__":
    unittest.main()
