# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only runner contract tests using the real canonical PERF parser."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "scripts/deepstream/ds-perf-run.sh"


class CapacityRunnerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="capacity-runner-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.config = self.root / "input config.txt"
        self.config.write_text("[property]\nbatch-size=1\n", encoding="utf-8")
        self.video = self.root / "input video.mp4"
        self.video.write_text("CPU-only fixture; never decoded", encoding="utf-8")
        self.log = self.root / "results" / "ds_s2_run1.log"
        self.result_path = Path(str(self.log) + ".json")
        self.started = self.root / "started"
        self.captured_config = self.root / "captured-config.txt"
        self.env = {
            "PATH": str(self.bin) + os.pathsep + os.defpath,
            "PYTHONDONTWRITEBYTECODE": "1",
            "FAKE_STARTED": str(self.started),
            "FAKE_CONFIG": str(self.captured_config),
            "FAKE_PERF": self.perf(40, 30),
        }
        self.executable("deepstream-app", f"#!{sys.executable}\n" +
                        "import os, pathlib, sys\n"
                        "pathlib.Path(os.environ['FAKE_STARTED']).touch()\n"
                        "pathlib.Path(os.environ['FAKE_CONFIG']).write_text(\n"
                        "    pathlib.Path(sys.argv[2]).read_text())\n"
                        "if os.environ.get('FAKE_JSON_COLLISION'):\n"
                        "    pathlib.Path(os.environ['FAKE_JSON_COLLISION']).write_text('racing evidence')\n"
                        "print(os.environ['FAKE_PERF'], end='')\n"
                        "sys.exit(int(os.environ.get('FAKE_APP_EXIT', '0')))\n")
        # Execute only the mock app; do not wait for the production 60-second cap.
        self.executable("timeout", "#!/usr/bin/env bash\n"
                        "shift\n\"$@\"\nstatus=$?\n"
                        "[ \"$status\" -eq 0 ] || exit \"$status\"\n"
                        "exit \"${FAKE_TIMEOUT_EXIT:-0}\"\n")

    @staticmethod
    def perf(*fps: float, windows: int = 5) -> str:
        header = "**PERF: " + " ".join(f"FPS {i} (Avg)" for i in range(len(fps)))
        row = "**PERF: " + " ".join(f"{value:.1f} (99.0)" for value in fps)
        return header + "\n" + (row + "\n") * windows

    def executable(self, name: str, source: str) -> None:
        path = self.bin / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o700)

    def run_runner(self, streams: str = "2", **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(RUNNER), str(self.config), streams, str(self.log), str(self.video)],
            cwd=self.root, env={**self.env, **env}, capture_output=True, text=True, timeout=10, check=False,
        )

    def result(self) -> dict:
        return json.loads(self.result_path.read_text(encoding="utf-8"))

    def test_uneven_sources_sum_measured_fps_and_fail_slow_source(self) -> None:
        run = self.run_runner(FAKE_PERF=self.perf(40, 20))
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        result = self.result()
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["aggregate_average_fps"], 60)
        self.assertEqual([item["minimum_fps"] for item in result["per_stream"]], [40, 20])

    def test_pass_writes_canonical_json_defaults_and_matching_batch_sizes(self) -> None:
        run = self.run_runner()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        result = self.result()
        self.assertEqual(result["producer"], "deepstream-per-source-perf-parser")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["streams"], 2)
        self.assertEqual(result["target_fps_per_stream"], 30)
        self.assertEqual(result["sustained_windows"], 5)
        self.assertEqual(result["task"], "detection")
        self.assertEqual(result["campaign_id"], "")
        self.assertEqual(result["log_sha256"], hashlib.sha256(self.log.read_bytes()).hexdigest())
        self.assertIn('"aggregate_average_fps": 70.0', run.stdout)
        config = self.captured_config.read_text(encoding="utf-8")
        self.assertEqual(config.count("batch-size=2\n"), 2)
        self.assertIn("num-sources=2\n", config)

    def test_target_task_and_campaign_are_literal_parser_arguments(self) -> None:
        campaign = "campaign with spaces; $(touch should-not-exist)"
        run = self.run_runner(FAKE_PERF=self.perf(25, 24), DS_TARGET_FPS="24",
                              DS_TASK="classification", DS_CAMPAIGN_ID=campaign)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        result = self.result()
        self.assertEqual(result["target_fps_per_stream"], 24)
        self.assertEqual(result["task"], "classification")
        self.assertEqual(result["campaign_id"], campaign)
        self.assertFalse((self.root / "should-not-exist").exists())
        self.log = self.log.with_name("higher-target.log")
        self.result_path = Path(str(self.log) + ".json")
        run = self.run_runner(DS_TARGET_FPS="41")
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertEqual(self.result()["status"], "FAIL")

    def test_fewer_than_five_windows_fail_with_evidence(self) -> None:
        run = self.run_runner(FAKE_PERF=self.perf(40, 40, windows=4))
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        result = self.result()
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["window_count"], 4)
        self.assertEqual(result["sustained_windows"], 5)

    def test_final_five_windows_ignore_warmup_but_reject_one_recent_dip(self) -> None:
        for recent, status in ((self.perf(40, 40), "PASS"),
                               (self.perf(40, 20, windows=1) + self.perf(40, 40, windows=4), "FAIL")):
            with self.subTest(status=status):
                self.log = self.root / f"{status}.log"
                self.result_path = Path(str(self.log) + ".json")
                run = self.run_runner(FAKE_PERF=self.perf(0, 0, windows=1) + recent)
                self.assertEqual(run.returncode, 0 if status == "PASS" else 1)
                self.assertEqual(self.result()["status"], status)

    def test_missing_sources_fail_and_no_perf_retains_fail_evidence(self) -> None:
        for perf, writes_json in ((self.perf(40), False), ("no measurements\n", True)):
            with self.subTest(perf=perf):
                self.log = self.root / ("missing-source.log" if "**PERF" in perf else "empty.log")
                self.result_path = Path(str(self.log) + ".json")
                run = self.run_runner(FAKE_PERF=perf)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(self.result_path.exists(), writes_json)
                if writes_json:
                    self.assertEqual(self.result()["status"], "FAIL")

    def test_invalid_stream_count_stops_before_execution_or_outputs(self) -> None:
        for streams in ("0", "-1", "1.5", "abc", "2; touch injected", " 2", "2\n", "00", "+2"):
            with self.subTest(streams=streams):
                run = self.run_runner(streams)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("positive integer", run.stdout + run.stderr)
                self.assertFalse(self.started.exists())
                self.assertFalse(self.log.exists())
                self.assertFalse(self.result_path.exists())

    def test_preexisting_log_or_json_is_preserved_before_execution(self) -> None:
        self.log.parent.mkdir()
        for path in (self.log, self.result_path):
            with self.subTest(path=path):
                path.write_text("existing evidence", encoding="utf-8")
                run = self.run_runner()
                self.assertNotEqual(run.returncode, 0)
                self.assertFalse(self.started.exists())
                self.assertEqual(path.read_text(encoding="utf-8"), "existing evidence")
                self.assertFalse((self.result_path if path == self.log else self.log).exists())
                path.unlink()

    def test_output_symlinks_are_rejected_without_touching_targets(self) -> None:
        self.log.parent.mkdir()
        for path in (self.log, self.result_path):
            for existing in (False, True):
                with self.subTest(path=path, existing=existing):
                    target = self.root / f"target-{path.suffix}-{existing}"
                    if existing:
                        target.write_text("keep target", encoding="utf-8")
                    path.symlink_to(target)
                    run = self.run_runner()
                    self.assertNotEqual(run.returncode, 0)
                    self.assertFalse(self.started.exists())
                    self.assertTrue(path.is_symlink())
                    if existing:
                        self.assertEqual(target.read_text(encoding="utf-8"), "keep target")
                    else:
                        self.assertFalse(target.exists())
                    path.unlink()

    def test_json_created_during_run_is_not_overwritten(self) -> None:
        run = self.run_runner(FAKE_JSON_COLLISION=str(self.result_path))
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(self.result_path.read_text(encoding="utf-8"), "racing evidence")

    def test_log_created_after_precheck_is_not_overwritten(self) -> None:
        # Inject a competing output exactly between the precheck and log open.
        self.executable("mkdir", f"#!{sys.executable}\n" +
                        "import os, pathlib, sys\n"
                        "os.makedirs(sys.argv[-1], exist_ok=True)\n"
                        "log = pathlib.Path(os.environ['FAKE_LOG_COLLISION'])\n"
                        "if os.environ.get('FAKE_SYMLINK_TARGET'):\n"
                        "    log.symlink_to(os.environ['FAKE_SYMLINK_TARGET'])\n"
                        "else:\n"
                        "    log.write_text('racing evidence')\n")
        target = self.root / "absent-target"
        for symlink in (False, True):
            with self.subTest(symlink=symlink):
                run = self.run_runner(FAKE_LOG_COLLISION=str(self.log),
                                      FAKE_SYMLINK_TARGET=str(target) if symlink else "")
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("exclusively create", run.stderr)
                self.assertFalse(self.started.exists())
                self.assertFalse(self.result_path.exists())
                if symlink:
                    self.assertTrue(self.log.is_symlink())
                    self.assertFalse(target.exists())
                else:
                    self.assertEqual(self.log.read_text(encoding="utf-8"), "racing evidence")
                self.log.unlink()

    def test_expected_timeout_parses_but_app_failure_does_not(self) -> None:
        run = self.run_runner(FAKE_TIMEOUT_EXIT="124")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(self.result()["status"], "PASS")
        self.log = self.root / "failure.log"
        self.result_path = Path(str(self.log) + ".json")
        run = self.run_runner(FAKE_APP_EXIT="7")
        self.assertEqual(run.returncode, 7, run.stdout + run.stderr)
        self.assertTrue(self.log.exists())
        self.assertFalse(self.result_path.exists())

    def test_log_writer_failure_cannot_publish_capacity(self) -> None:
        self.executable("tee", "#!/usr/bin/env bash\nexit 9\n")
        run = self.run_runner()
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(self.result_path.exists())


if __name__ == "__main__":
    unittest.main()
