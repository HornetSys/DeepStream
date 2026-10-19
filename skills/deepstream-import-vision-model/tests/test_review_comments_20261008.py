#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for review fixes and credential-safe report helpers."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SKILL = Path(__file__).resolve().parents[1]

class ReviewFixTests(unittest.TestCase):
    def _benchmark_fixture(self, folder, name, throughput="50"):
        model = Path(folder) / "models" / name
        for subdir in ("benchmarks/engines", "benchmarks/b1", "benchmarks/b2", "benchmarks/ds"):
            (model / subdir).mkdir(parents=True)
        (model / "benchmarks/engines/model_dynamic_b2.engine").write_text("unit-fixture-only")
        for batch in (1, 2):
            (model / f"benchmarks/b{batch}/trtexec_b{batch}.log").write_text(f"Throughput: {throughput} qps\nGPU Compute Time: mean = 1.0 ms, percentile(99%) = 2.0 ms\n")
        for run in (1, 2):
            (model / f"benchmarks/ds/ds_s2_run{run}.log").write_text("**PERF: 40.0 (40.0) 20.0 (20.0)\n" * 5)

    def test_collector_checks_every_source_and_clears_stale_input_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            self._benchmark_fixture(folder, "first")
            self._benchmark_fixture(folder, "second")
            script = 'source "$1" models/first || exit 9; echo "MEASURED:$TOTAL_FPS_RUN2:$WORST_FPS_RUN2:$RT_LABEL_RUN2"; unset MODEL_INPUT_NAME MODEL_INPUT_H MODEL_INPUT_W; source "$1" models/second; echo "SECOND:$?"'
            env = {"PATH": os.defpath, "SKILL_ROOT": str(SKILL), "MODEL_INPUT_NAME": "input_first", "MODEL_INPUT_H": "224", "MODEL_INPUT_W": "224"}
            result = subprocess.run(["bash", "-c", script, "bash", str(SKILL / "scripts/report/collect-report-vars.sh")], cwd=folder, env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MEASURED:60.0:20.0:NO", result.stdout)
        self.assertIn("SECOND:1", result.stdout)
        self.assertIn("missing numeric model input contract", result.stderr)

    def test_collector_rejects_zero_or_missing_throughput(self):
        for throughput in ("0", "malformed"):
            with self.subTest(throughput=throughput), tempfile.TemporaryDirectory() as folder:
                self._benchmark_fixture(folder, "model", throughput)
                env = {"PATH": os.defpath, "SKILL_ROOT": str(SKILL), "MODEL_INPUT_NAME": "input", "MODEL_INPUT_H": "224", "MODEL_INPUT_W": "224"}
                result = subprocess.run(["bash", "-c", 'source "$1" models/model; echo "STATUS:$?"', "bash", str(SKILL / "scripts/report/collect-report-vars.sh")], cwd=folder, env=env, capture_output=True, text=True, timeout=15)
                self.assertIn("STATUS:1", result.stdout)

    def _host_preflight(self, folder, image, media):
        fake = Path(folder) / "docker"
        fake.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("MOCK_DOCKER")\nif "--entrypoint" in sys.argv and sys.argv[sys.argv.index("--entrypoint")+1]=="test":\n    sys.exit(0 if os.path.isfile(sys.argv[-1]) else 1)\n')
        fake.chmod(0o700)
        # Simulate host mode even when the unit-test host itself is a container.
        source = (SKILL / "scripts/preflight.sh").read_text().replace('if [ -f /.dockerenv ]; then', 'if false; then')
        return subprocess.run(["bash", "-c", source], cwd=folder, env={"PATH": folder + os.pathsep + os.defpath, "DS_IMAGE": image, "DS_VIDEO": str(media), "DS_PREFLIGHT_PROFILE": "native-pipeline"}, capture_output=True, text=True, timeout=15)

    def test_invalid_image_pin_stops_before_any_docker_call(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self._host_preflight(folder, "nvcr.io/nvidia/deepstream:latest", Path(folder) / "missing.mp4")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("MOCK_DOCKER", result.stdout)

    def test_explicit_media_is_checked_instead_of_default(self):
        image = "nvcr.io/nvidia/deepstream@sha256:" + "a" * 64
        with tempfile.TemporaryDirectory() as folder:
            media = Path(folder) / "explicit.mp4"
            media.write_text("unit-fixture-only")
            valid = self._host_preflight(folder, image, media)
            missing = self._host_preflight(folder, image, Path(folder) / "missing.mp4")
        self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
        self.assertEqual(missing.returncode, 1)
        self.assertIn("selected input video missing", missing.stdout)

    def test_sourced_collector_does_not_exit_parent_or_mask_missing_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(["bash", "-c", 'source "$1" models/missing; result=$?; echo "PARENT_ALIVE:$result"', "bash", str(SKILL / "scripts/report/collect-report-vars.sh")], cwd=folder, env={"PATH": os.defpath, "SKILL_ROOT": str(SKILL)}, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0)
        self.assertIn("PARENT_ALIVE:1", result.stdout)
        self.assertIn("No engine found", result.stderr)

    def test_nonfinite_benchmark_values_and_unsafe_model_path(self):
        spec = importlib.util.spec_from_file_location("writer", SKILL / "scripts/report/write-benchmark-data.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for value in ("nan", "inf", "-inf"):
            self.assertIsNone(module.to_num(value))
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(["python3", str(SKILL / "scripts/report/write-benchmark-data.py")], cwd=folder, env={"PATH": os.defpath, "MODEL_NAME": "../outside"}, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("safe model directory", result.stderr)

    def test_hf_token_never_enters_curl_argv_or_trace(self):
        synthetic = "synthetic_review_only_20261008"
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / "curl"
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps({"argv":sys.argv[1:],"stdin":sys.stdin.read()}))\n')
            fake.chmod(0o700)
            result = subprocess.run(["bash", "-x", "-c", 'source "$1"; hf_curl -sS --max-time 5 https://huggingface.co/', "bash", str(SKILL / "scripts/model/hf-curl.sh")], env={"PATH": folder + os.pathsep + os.defpath, "HF_TOKEN": synthetic}, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertNotIn(synthetic, " ".join(record["argv"]))
        self.assertNotIn(synthetic, result.stderr)
        self.assertIn("Authorization: Bearer " + synthetic, record["stdin"])

    def test_encoder_contract_requires_theora_even_with_nvenc(self):
        source = (SKILL / "scripts/preflight.sh").read_text()
        self.assertNotIn('gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1 ||', source)
        self.assertIn("theoraenc+oggmux required", source)
        self.assertIn("gst-inspect-1.0 oggmux", source)

    def test_report_child_errors_are_bounded_and_license_is_present(self):
        source = (SKILL / "scripts/report/generate-model-report.py").read_text()
        self.assertIn("subprocess.SubprocessError", source)
        self.assertIn("SPDX-License-Identifier: Apache-2.0", (SKILL / "BENCHMARK.md").read_text())

    def test_called_process_error_returns_report_blocked_without_traceback(self):
        spec = importlib.util.spec_from_file_location("report_review", SKILL / "scripts/report/generate-model-report.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        stderr = io.StringIO()
        with mock.patch.object(module, "generate", side_effect=subprocess.CalledProcessError(1, ["benchmark-sections.py"])), mock.patch("sys.argv", ["report", "case-evidence.json"]), contextlib.redirect_stderr(stderr):
            status = module.main()
        self.assertEqual(status, 4)
        self.assertIn("REPORT BLOCKED", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_failed_oracle_validator_cannot_publish_pass(self):
        spec = importlib.util.spec_from_file_location("prepare_review", SKILL / "scripts/report/prepare-campaign.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for status in (0, 1):
            with self.subTest(exit_code=status), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for name in ("model.onnx", "input.mp4", "oracle.json"):
                    (root/name).write_text("unit-fixture-only")
                campaign = root/"input.json"
                campaign.write_text(json.dumps({"schema_version":1,"cases":[{
                    "case_id":"case","task":"classification","source":str(root/"model.onnx"),
                    "inputs":[{"path":str(root/"input.mp4")}],"oracle":str(root/"oracle.json")
                }]}))
                process = subprocess.CompletedProcess([], status, '{"status":"PASS","errors":[]}', "failure detail" if status else "")
                with mock.patch.object(module.subprocess, "run", return_value=process):
                    module.prepare(campaign, root/"models")
                manifest = json.loads((root/"models/case/case-evidence.json").read_text())
                validation = json.loads((root/"models/case/oracle-validation.json").read_text())
                self.assertEqual(manifest["oracle_status"], "PASS" if status == 0 else "BLOCKED")
                self.assertEqual(validation["validator_exit_code"], status)
                if status:
                    self.assertEqual(validation["reported_result"]["status"], "PASS")
                    self.assertIn("oracle validator exited 1", validation["errors"][-1])
                    self.assertIn("reviewed semantic oracle is BLOCKED", manifest["certification_limitations"])

    def test_legacy_writer_no_longer_uses_environment_model_path(self):
        source = (SKILL/"scripts/report/write-benchmark-report.py").read_text()
        self.assertNotIn('env("MODEL_NAME"', source)
        self.assertIn('generate-model-report.py', source)
        reporter = (SKILL/"scripts/report/generate-model-report.py").read_text()
        self.assertIn('SAFE_NAME.fullmatch(case_id)', reporter)
        self.assertIn('case_id != case_dir.name', reporter)

if __name__ == "__main__":
    unittest.main()
