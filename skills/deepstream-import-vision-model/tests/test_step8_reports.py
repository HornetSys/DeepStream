# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 8 reporting must remain evidence-backed and keep legacy filenames."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT = ROOT_DIR / "scripts" / "report"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), REPORT / name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reporter = load_script("generate-model-report.py")
indexer = load_script("generate-campaign-index.py")
preparer = load_script("prepare-campaign.py")


def write(path: Path, data: bytes = b"evidence") -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def png_bytes(width: int = 64, height: int = 48) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + kind + body +
                struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))

    rows = b"".join(b"\x00" + bytes((30, 110, 210)) * width for _ in range(height))
    header = struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def passing_adapter_case(case: Path) -> dict:
    case.mkdir()
    source = write(case / "model.onnx", b"model")
    fixture = write(case / "input.mp4", b"fixture")
    source_hash = source["sha256"]
    fixture_hash = fixture["sha256"]
    oracle_data = {"status": "PASS", "samples": [{"frame": 1, "class_id": 1}]}
    oracle_path = case / "oracle.json"
    oracle_path.write_text(json.dumps(oracle_data), encoding="utf-8")
    oracle_hash = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    oracle_validation = write(case / "oracle-validation.json",
                              json.dumps({"status": "PASS", "oracle_sha256": oracle_hash}).encode())
    structured = {"schema_version": 1, "campaign_id": "campaign-1",
                  "producer": "deepstream-type-adapter-runner", "task": "classification",
                  "source_hash": source_hash, "input_fixture_sha256": fixture_hash,
                  "results": [{"class_id": 1}]}
    structured_record = write(case / "structured.json", json.dumps(structured).encode())
    structured_hash = structured_record["sha256"]
    output_record = write(case / "rendered-output.png", png_bytes())
    output_hash = output_record["sha256"]
    source_frame = write(case / "source.png", png_bytes())
    shared = {"schema_version": 1, "campaign_id": "campaign-1", "task": "classification",
              "source_hash": source_hash, "input_fixture_sha256": fixture_hash}
    records = {
        "backend_parity": {**shared, "producer": "compare_outputs.py", "status": "PASS"},
        "deepstream_runtime": {**shared, "producer": "deepstream-type-adapter-runner",
                               "status": "PASS", "startup": True, "buffer_count": 2,
                               "eos": True, "exit_code": 0,
                               "structured_output_sha256": structured_hash,
                               "rendered_output_sha256": output_hash},
        "semantic_validation": {**shared, "producer": "adapter-semantic-validator",
                                 "status": "PASS", "oracle_sha256": oracle_hash,
                                 "structured_output_sha256": structured_hash},
        "overlay_evidence": {**shared, "producer": "adapter-renderer",
                             "structured_output_sha256": structured_hash,
                             "source_frame_sha256": source_frame["sha256"]},
        "visual_validation": {**shared, "producer": "validate_rendered_output.py",
                              "status": "RENDERED_PASS", "pipeline_visual_result": "PASS",
                              "structured_output_sha256": structured_hash,
                              "rendered_output_sha256": output_hash},
    }
    evidence = {"input_fixture": fixture, "rendered_output": output_record,
                "structured_output": structured_record}
    for name, payload in records.items():
        evidence[name] = write(case / f"{name}.json", json.dumps(payload).encode())
    status = {"schema_version": 1, "task": "classification", "status": "PASS",
              "implementation_status": "tested", "source_hash": source_hash,
              "evidence": evidence}
    status_path = case / "adapter-status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    rendered_frame = output_record
    video = write(case / "rendered.ogv", b"video")
    log = write(case / "pipeline.log", b"pipeline passed")
    manifest = {
        "schema_version": 1, "case_id": case.name, "task": "classification",
        "revision": f"sha256:{source_hash}", "source": source, "inputs": [fixture],
        "oracle": {"path": str(oracle_path), "sha256": oracle_hash}, "oracle_status": "PASS",
        "commands": [{"phase": "pipeline", "exit_code": 0, "log": log}],
        "gates": {name: {"status": "PASS"} for name in reporter.GATES},
        "capacity": {"status": "UNTESTED"},
        "evidence": {"source_frame": source_frame, "rendered_frame": rendered_frame,
                     "rendered_video": video, "structured_output": structured_record,
                     "adapter_status": write(status_path, status_path.read_bytes()),
                     "oracle_validation": oracle_validation},
    }
    return manifest


def passing_detection_case(case: Path) -> dict:
    case.mkdir()
    reports = case / "reports"
    assets = reports / "assets"
    charts = reports / "charts"
    assets.mkdir(parents=True)
    charts.mkdir()
    source = write(case / "model.onnx", b"detector")
    fixture = write(case / "fixture.mp4", b"video")
    source_frame = write(case / "source.png", png_bytes())
    rendered_frame = write(case / "rendered.png", png_bytes())
    shutil.copyfile(source_frame["path"], assets / "source_frame.png")
    shutil.copyfile(rendered_frame["path"], assets / "rendered_frame.png")
    rendered_video = write(case / "rendered.ogv", b"rendered video")
    structured = write(case / "detections.json", b'{"objects":[{"class_id":1}]}' )
    kitti = write(case / "kitti.txt", b"Car 0 0 0 1 2 30 40 1 1 1 1 1 1 0 0.9\n")
    benchmark_data = write(reports / "benchmark_data.json", b'{"measured":true}')
    chart_names = (
        "chart_trtexec_bs1_vs_bsmax.png", "chart_trtexec_throughput.png",
        "chart_ds_streams_vs_fps.png", "chart_trt_vs_ds.png", "chart_efficiency.png",
    )
    for name in chart_names:
        write(charts / name, png_bytes(24, 18))
    sections = [f"## {index}. Section {index}" for index in range(1, 13)]
    chart_refs = [f"![{name}](charts/{name})" for name in chart_names]
    report_text = "# Detection benchmark\n\n" + "\n\n".join(sections + chart_refs)
    report_text += "\n\n![input](assets/source_frame.png)\n\n![output](assets/rendered_frame.png)\n"
    benchmark_report = write(reports / "benchmark_report.md", report_text.encode())
    oracle_data = {"status": "PASS", "annotations": [{"frame": 1, "class_id": 1}]}
    oracle_path = case / "oracle.json"
    oracle_path.write_text(json.dumps(oracle_data), encoding="utf-8")
    oracle_hash = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    oracle_validation = write(case / "oracle-validation.json",
                              json.dumps({"status": "PASS", "oracle_sha256": oracle_hash}).encode())
    source_hash, fixture_hash = source["sha256"], fixture["sha256"]
    common = {"schema_version": 1, "campaign_id": "detect-1", "task": "detection",
              "source_hash": source_hash, "input_fixture_sha256": fixture_hash}
    records = {
        "backend_parity": {**common, "producer": "compare_outputs.py", "status": "PASS"},
        "deepstream_runtime": {**common, "producer": "deepstream-runner", "status": "PASS",
                               "startup": True, "buffer_count": 2, "eos": True, "exit_code": 0,
                               "structured_output_sha256": structured["sha256"]},
        "semantic_validation": {**common, "producer": "semantic-validator", "status": "PASS",
                                 "oracle_status": "PASS", "oracle_sha256": oracle_hash,
                                 "structured_output_sha256": structured["sha256"]},
        "visual_validation": {**common, "producer": "frame-validator", "status": "RENDERED_PASS",
                              "pipeline_visual_result": "PASS",
                              "structured_output_sha256": structured["sha256"],
                              "source_frame_sha256": source_frame["sha256"],
                              "rendered_frame_sha256": rendered_frame["sha256"]},
    }
    evidence = {
        "source_frame": source_frame, "rendered_frame": rendered_frame,
        "rendered_video": rendered_video, "structured_output": structured,
        "semantic_oracle": {"path": str(oracle_path), "sha256": oracle_hash},
        "kitti_output": kitti, "benchmark_data": benchmark_data,
        "benchmark_report": benchmark_report,
    }
    for name, payload in records.items():
        evidence[name] = write(case / f"{name}.json", json.dumps(payload).encode())
    log = write(case / "deepstream.log", b"startup buffers eos")
    return {
        "schema_version": 1, "case_id": case.name, "task": "detection",
        "revision": f"sha256:{source_hash}", "source": source, "inputs": [fixture],
        "oracle": {"path": str(oracle_path), "sha256": oracle_hash}, "oracle_status": "PASS",
        "commands": [{"phase": "deepstream", "exit_code": 0, "log": log}],
        "gates": {name: {"status": "PASS"} for name in reporter.GATES},
        "capacity": {"status": "UNTESTED"}, "evidence": {**evidence, "oracle_validation": oracle_validation},
    }


class Step8ReportTests(unittest.TestCase):
    def test_legacy_converter_names_and_validation_kind(self):
        fake_markdown = types.SimpleNamespace(markdown=lambda value, extensions: "<p>ok</p>")
        with mock.patch.dict(sys.modules, {"markdown": fake_markdown}):
            converter = load_script("md-to-html-pdf.py")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            md = root / "source.md"
            md.write_text("# Test\n", encoding="utf-8")
            css = root / "style.css"
            css.write_text("body {}", encoding="utf-8")

            def fake_run(argv, **_kwargs):
                Path(argv[-1]).write_bytes(b"%PDF-1.4\ntest")
                return types.SimpleNamespace(returncode=0, stderr="")

            with mock.patch.object(converter, "find_wkhtmltopdf", return_value="/usr/bin/wkhtmltopdf"), mock.patch.object(converter.subprocess, "run", side_effect=fake_run):
                converter.main([str(md), str(css), str(root), "model"])
                self.assertTrue((root / "benchmark_report_model.pdf").is_file())
                converter.main([str(md), str(css), str(root), "model", "--report-kind", "validation"])
                self.assertTrue((root / "validation_report_model.pdf").is_file())
                converter.main([str(md), str(css), str(root), "model", "--report-kind", "status"])
                self.assertTrue((root / "status_report_model.pdf").is_file())
            self.assertIn("DeepStream Validation Report", (root / "validation_report.html").read_text())

    def test_converter_embeds_local_image_in_html(self):
        fake_markdown = types.SimpleNamespace(markdown=lambda value, extensions: '<img src="frame.png"/>')
        with mock.patch.dict(sys.modules, {"markdown": fake_markdown}):
            converter = load_script("md-to-html-pdf.py")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root / "frame.png", png_bytes())
            embedded = converter.embed_images('<img src="frame.png"/>', str(root))
            self.assertIn("data:image/png;base64,", embedded)

    def test_full_adapter_evidence_can_reach_functional_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "classification"
            manifest = passing_adapter_case(case)
            status, errors, evidence, capacity = reporter.evaluate(manifest, case)
            self.assertEqual(status, "PASS", errors)
            self.assertEqual(reporter._pipeline_visual_result(manifest, evidence), "PASS")
            self.assertEqual(capacity["status"], "UNTESTED")

    def test_validation_report_preserves_synthetic_scope_and_clean_log_path(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "classification"
            manifest = passing_adapter_case(case)
            manifest["scope_note"] = "Synthetic dog fixture only; not real-world accuracy."
            status, errors, evidence, capacity = reporter.evaluate(manifest, case)
            self.assertEqual(status, "PASS", errors)
            markdown = reporter._write_markdown(manifest, case, status, errors, evidence, capacity, "validation")
            body = markdown.read_text(encoding="utf-8")
            self.assertIn("Synthetic dog fixture only; not real-world accuracy.", body)
            self.assertNotIn("{&#x27;path&#x27;", body)

    def test_detection_keeps_legacy_twelve_sections_and_five_charts(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "detector"
            manifest = passing_detection_case(case)
            status, errors, _, capacity = reporter.evaluate(manifest, case)
            self.assertEqual(status, "PASS", errors)
            self.assertEqual(capacity["status"], "UNTESTED")

    def test_upstream_stage_requires_pass_record_bound_to_fixture_and_output(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "vehiclemakenet"
            manifest = passing_adapter_case(case)
            stage = "vehicle_detection_roi"
            output = write(case / "vehicle-rois.json", b'{"rois":[[1,2,3,4]]}')
            record = {
                "schema_version": 1, "stage": stage, "status": "PASS",
                "kind": "inference", "model_sha256": manifest["source"]["sha256"],
                "producer": "detector-roi-stage",
                "input_fixture_sha256s": [manifest["inputs"][0]["sha256"]],
                "output": output,
            }
            stage_record = write(case / "upstream-stage.json", json.dumps(record).encode())
            manifest["upstream_stages"] = [stage]
            manifest["evidence"][f"upstream_{stage}"] = stage_record
            status, errors, _, _ = reporter.evaluate(manifest, case)
            self.assertEqual(status, "PASS", errors)

            record["input_fixture_sha256s"] = ["0" * 64]
            stage_record = write(case / "upstream-stage.json", json.dumps(record).encode())
            manifest["evidence"][f"upstream_{stage}"] = stage_record
            status, errors, _, _ = reporter.evaluate(manifest, case)
            self.assertEqual(status, "BLOCKED")
            self.assertTrue(any("input fixture bindings differ" in error for error in errors))

    def test_detection_visual_frame_must_match_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "detector"
            manifest = passing_detection_case(case)
            visual_path = Path(manifest["evidence"]["visual_validation"]["path"])
            visual = json.loads(visual_path.read_text(encoding="utf-8"))
            visual["rendered_frame_sha256"] = "0" * 64
            visual_record = write(visual_path, json.dumps(visual).encode())
            manifest["evidence"]["visual_validation"] = visual_record
            status, errors, _, _ = reporter.evaluate(manifest, case)
            self.assertEqual(status, "BLOCKED")
            self.assertTrue(any("rendered frame digest differs" in error for error in errors))

    def test_campaign_preparer_hashes_inputs_and_never_marks_gates_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write(root / "model.onnx", b"model")
            media = write(root / "input.mp4", b"media")
            campaign = root / "input.json"
            campaign.write_text(json.dumps({"schema_version": 1, "cases": [{
                "case_id": "case-one", "task": "classification",
                "source": source["path"], "inputs": [{"role": "video", "path": media["path"]}],
            }]}), encoding="utf-8")
            models = root / "models"
            preparer.prepare(campaign, models)
            manifest = json.loads((models / "case-one" / "case-evidence.json").read_text())
            self.assertEqual(manifest["source"]["sha256"], source["sha256"])
            self.assertEqual(manifest["inputs"][0]["sha256"], media["sha256"])
            self.assertTrue(all(gate["status"] == "UNTESTED" for gate in manifest["gates"].values()))
            self.assertEqual(manifest["oracle_status"], "UNTESTED")
            self.assertEqual(manifest["input_blockers"], [])
            self.assertEqual(manifest["certification_limitations"],
                             ["reviewed semantic oracle is UNTESTED"])
            self.assertEqual(manifest["first_failure"]["phase"], "execution")
            self.assertIn("reviewed semantic oracle is UNTESTED", manifest["preflight_findings"])
            with self.assertRaises(FileExistsError):
                preparer.prepare(campaign, models)

    def test_campaign_preparer_blocks_missing_model_before_diagnostic_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            media = write(root / "input.mp4", b"media")
            campaign = root / "input.json"
            campaign.write_text(json.dumps({"schema_version": 1, "cases": [{
                "case_id": "missing-model", "task": "detection", "source": None,
                "inputs": [{"role": "video", "path": media["path"]}],
            }]}), encoding="utf-8")
            models = root / "models"
            preparer.prepare(campaign, models)
            manifest = json.loads((models / "missing-model" / "case-evidence.json").read_text())
            self.assertTrue(manifest["input_blockers"])
            self.assertEqual(manifest["first_failure"]["phase"], "input")
            self.assertIn("model artifact", manifest["first_failure"]["reason"])

    def test_model_media_source_map_keeps_all_application_cases_and_legacy_step8(self):
        source_map = (ROOT_DIR / "references" / "model-zoo-media.md").read_text()
        report_reference = (ROOT_DIR / "references" / "report-generation.md").read_text()
        for name in ("TrafficCamNet", "PeopleNet", "YOLOv8n", "RT-DETR", "LPD-US",
                     "VehicleMakeNet", "VehicleTypeNet", "CitySemSegFormer", "SegFormer",
                     "BodyPose2D", "Depth Anything V2", "FoundationStereo", "Bi3D"):
            with self.subTest(name=name):
                self.assertIn(f"| {name} |", source_map)
        self.assertIn("sample_720p.mp4", source_map)
        self.assertIn("sample_1080p_h264.mp4", source_map)
        self.assertIn("## Detection benchmark contract", report_reference)
        self.assertIn("## Final visual inspection", report_reference)
        self.assertIn("exactly five", report_reference)

    def test_real_pdf_embeds_and_renders_image_when_tools_are_available(self):
        # Earlier tests (including the legacy report tests) may leave a stub
        # Markdown module in sys.modules. Import a real installation in an
        # isolated module-table state so this integration test exercises the
        # actual renderer instead of inheriting another test's fake.
        had_markdown = "markdown" in sys.modules
        previous_markdown = sys.modules.pop("markdown", None)
        try:
            markdown_module = importlib.import_module("markdown")
        except ImportError:
            markdown_module = None
        try:
            if (markdown_module is None or not callable(getattr(markdown_module, "markdown", None))
                    or not shutil.which("wkhtmltopdf")):
                self.skipTest("Python Markdown or wkhtmltopdf is unavailable")
            converter = load_script("md-to-html-pdf.py")
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write(root / "frame.png", png_bytes(320, 180))
                markdown_path = root / "status_report.md"
                markdown_path.write_text("# Status\n\n![frame](frame.png)\n", encoding="utf-8")
                output = root / "reports"
                converter.main([str(markdown_path), str(REPORT / "report-style.css"),
                                str(output), "pdf-smoke", "--report-kind", "status"])
                pdf = output / "status_report_pdf-smoke.pdf"
                self.assertTrue(pdf.is_file() and pdf.stat().st_size > 0)
                self.assertIn("data:image/png;base64,", (output / "status_report.html").read_text())
                if shutil.which("pdfimages"):
                    listing = subprocess.run(["pdfimages", "-list", str(pdf)],
                                             capture_output=True, text=True, check=True).stdout
                    self.assertRegex(listing, r"320\s+180")
                if shutil.which("pdftoppm"):
                    page = root / "page"
                    subprocess.run(["pdftoppm", "-f", "1", "-l", "1", "-png",
                                    "-singlefile", str(pdf), str(page)], check=True)
                    self.assertTrue((root / "page.png").is_file())
        finally:
            if had_markdown:
                sys.modules["markdown"] = previous_markdown
            else:
                sys.modules.pop("markdown", None)

    def test_claimed_pass_without_evidence_becomes_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "citysemsegformer"
            case.mkdir()
            manifest = {"schema_version": 1, "case_id": case.name,
                        "task": "segmentation", "gates": {name: {"status": "PASS"} for name in reporter.GATES}}
            result, errors, _, _ = reporter.evaluate(manifest, case)
            self.assertEqual(result, "BLOCKED")
            self.assertIn("adapter_status: path is missing", errors)

    def test_failed_gate_still_produces_status_with_capacity_unmeasured(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "depth"
            case.mkdir()
            manifest = {"schema_version": 1, "case_id": case.name, "task": "depth",
                        "inputs": "invalid", "gates": {"backend": {"status": "FAIL"}}}
            result, errors, _, capacity = reporter.evaluate(manifest, case)
            self.assertEqual(result, "FAIL")
            self.assertEqual(capacity["status"], "UNTESTED")
            self.assertIn("input fixture list is absent", errors)

    def test_invalid_capacity_claim_does_not_erase_functional_result(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "model"
            case.mkdir()
            manifest = {"schema_version": 1, "case_id": case.name, "task": "classification",
                        "capacity": {"status": "PASS", "streams": 1, "fps_per_stream": 31,
                                     "target_fps": 30, "runner": "single-stream"}}
            capacity_errors = []
            capacity = reporter._capacity(manifest, case, capacity_errors)
            self.assertEqual(capacity["status"], "UNTESTED")
            self.assertTrue(any("multi-stream" in error for error in capacity_errors))

    def test_capacity_pass_requires_hashed_task_compatible_multistream_log(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp)
            log = write(case / "capacity.log", b"two-stream measured run")
            manifest = {"task": "classification", "capacity": {
                "status": "PASS", "streams": 2, "fps_per_stream": 31,
                "target_fps": 30, "task": "classification",
                "runner": "classification-multistream-runner", "log": log,
            }}
            errors = []
            result = reporter._capacity(manifest, case, errors)
            self.assertEqual(result["status"], "PASS", errors)

            manifest["capacity"]["log"] = {"path": log["path"]}
            errors = []
            result = reporter._capacity(manifest, case, errors)
            self.assertEqual(result["status"], "UNTESTED")
            self.assertTrue(any("log SHA-256" in error for error in errors))

    def test_stereo_requires_two_real_inputs_and_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "foundationstereo"
            case.mkdir()
            source = write(case / "model.onnx")
            left = write(case / "left.png")
            manifest = {"schema_version": 1, "case_id": case.name, "task": "stereo-depth",
                        "source": source, "inputs": [{"role": "left", **left}], "gates": {}}
            result, errors, _, _ = reporter.evaluate(manifest, case)
            self.assertEqual(result, "BLOCKED")
            self.assertTrue(any("left and right" in message for message in errors))
            self.assertTrue(any("calibration" in message for message in errors))

    def test_stereo_intake_checks_calibration_file_and_distinct_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write(root / "model.onnx")
            left = write(root / "left.png", png_bytes())
            right = write(root / "right.png", png_bytes(65, 48))
            input_path = root / "input.json"
            case = {"case_id": "foundationstereo", "task": "stereo-depth",
                    "source": source["path"],
                    "inputs": [{"role": "left", "path": left["path"]},
                               {"role": "right", "path": right["path"]}],
                    "calibration": str(root / "missing-calib.txt")}
            input_path.write_text(json.dumps({"schema_version": 1, "cases": [case]}),
                                  encoding="utf-8")
            preparer.prepare(input_path, root / "missing-calibration")
            manifest = json.loads((root / "missing-calibration" / case["case_id"] /
                                   "case-evidence.json").read_text(encoding="utf-8"))
            self.assertTrue(any("calibration file is absent" in item
                                for item in manifest["input_blockers"]))
            _, errors, _, _ = reporter.evaluate(
                manifest, root / "missing-calibration" / case["case_id"])
            self.assertTrue(any("stereo calibration" in item for item in errors))

            calibration = write(root / "calib.txt", b"fx=100\nbaseline=1\n")
            case["calibration"] = calibration["path"]
            case["inputs"][1]["path"] = left["path"]
            input_path.write_text(json.dumps({"schema_version": 1, "cases": [case]}),
                                  encoding="utf-8")
            preparer.prepare(input_path, root / "duplicate-pair")
            duplicate = json.loads((root / "duplicate-pair" / case["case_id"] /
                                    "case-evidence.json").read_text(encoding="utf-8"))
            self.assertTrue(any("same file" in item for item in duplicate["input_blockers"]))
            self.assertEqual(duplicate["calibration"]["sha256"], calibration["sha256"])

    def test_status_report_has_real_failure_and_no_fabricated_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "bodypose2d"
            case.mkdir()
            source = write(case / "model.onnx")
            video = write(case / "people.mp4")
            manifest = {"schema_version": 1, "case_id": case.name, "task": "pose",
                        "source": source, "revision": "pinned", "inputs": [{"role": "video", **video}],
                        "gates": {"backend": {"status": "PASS"}, "pipeline": {"status": "BLOCKED"}},
                        "first_failure": {"phase": "pipeline", "reason": "person ROI runner unavailable"}}
            manifest_path = case / "case-evidence.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            def fake_run(argv, **_kwargs):
                output = Path(argv[4])
                output.joinpath(f"status_report_{case.name}.pdf").write_bytes(b"%PDF-1.4\ntest")
                output.joinpath("status_report.html").write_text("report", encoding="utf-8")
                return types.SimpleNamespace(returncode=0, stderr="")

            with mock.patch.object(reporter.subprocess, "run", side_effect=fake_run):
                result = reporter.generate(manifest_path)
            self.assertEqual(result["functional_status"], "BLOCKED")
            self.assertEqual(result["capacity_status"], "UNTESTED")
            md = (case / "reports" / "status_report.md").read_text()
            self.assertIn("person ROI runner unavailable", md)
            self.assertIn("Result: BLOCKED", md)

    def test_early_stop_status_does_not_require_model_media_or_oracle(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "missing-inputs"
            case.mkdir()
            manifest_path = case / "case-evidence.json"
            manifest_path.write_text(json.dumps({
                "schema_version": 1,
                "case_id": case.name,
                "task": "classification",
                "gates": {name: {"status": "UNTESTED"} for name in reporter.GATES},
                "input_blockers": ["model and fixture absent"],
                "first_failure": {"phase": "input", "reason": "model and fixture absent"},
            }), encoding="utf-8")

            def fake_run(argv, **_kwargs):
                output = Path(argv[4])
                output.joinpath(f"status_report_{case.name}.pdf").write_bytes(b"%PDF-1.4\ntest")
                output.joinpath("status_report.html").write_text("report", encoding="utf-8")
                return types.SimpleNamespace(returncode=0, stderr="")

            with mock.patch.object(reporter.subprocess, "run", side_effect=fake_run):
                result = reporter.generate(manifest_path)
            self.assertEqual(result["functional_status"], "BLOCKED")
            self.assertNotEqual(result["pipeline_visual_result"], "PASS")
            self.assertTrue((case / "reports" / f"status_report_{case.name}.pdf").is_file())
            status = (case / "reports" / "status_report.md").read_text()
            self.assertIn("model and fixture absent", status)
            self.assertIn("Result: BLOCKED", status)

    def test_detection_status_preserves_visual_gate_without_benchmark_noise(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "trafficcamnet"
            case.mkdir()
            source = write(case / "model.onnx")
            video = write(case / "source.mp4")
            input_frame = write(case / "input.png", png_bytes(24, 18))
            output_frame = write(case / "output.png", png_bytes(24, 18))
            output_video = write(case / "output.ogv")
            structured = write(case / "detections.txt")
            common = {"schema_version": 1, "campaign_id": "diagnostic-1", "task": "detection",
                      "source_hash": source["sha256"], "input_fixture_sha256": video["sha256"],
                      "structured_output_sha256": structured["sha256"]}
            runtime = write(case / "runtime.json", json.dumps({
                **common, "producer": "deepstream-runner", "status": "PASS", "startup": True,
                "buffer_count": 2, "eos": True, "exit_code": 0,
            }).encode())
            visual = write(case / "visual.json", json.dumps({
                **common, "producer": "render-validator", "status": "RENDERED_PASS",
                "pipeline_visual_result": "PASS",
                "source_frame_sha256": input_frame["sha256"],
                "rendered_frame_sha256": output_frame["sha256"],
            }).encode())
            manifest = {
                "schema_version": 1, "case_id": case.name, "task": "detection",
                "source": source, "revision": "pinned",
                "inputs": [{"role": "video", **video}], "oracle": None,
                "gates": {"pipeline": {"status": "PASS"},
                          "visual": {"status": "PASS"},
                          "semantic": {"status": "UNTESTED"}},
                "evidence": {"source_frame": input_frame,
                             "rendered_frame": output_frame,
                             "rendered_video": output_video,
                             "structured_output": structured,
                             "deepstream_runtime": runtime,
                             "visual_validation": visual},
            }
            status, errors, evidence, _ = reporter.evaluate(manifest, case)
            self.assertEqual(status, "BLOCKED")
            self.assertIn("rendered_frame", evidence)
            self.assertEqual(reporter._pipeline_visual_result(manifest, evidence), "PASS")
            self.assertTrue(any("oracle is absent" in error for error in errors))
            self.assertFalse(any("chart " in error or "benchmark_data" in error
                                 for error in errors))

            missing_video = {key: value for key, value in evidence.items()
                             if key != "rendered_video"}
            self.assertEqual(reporter._pipeline_visual_result(manifest, missing_video),
                             "UNTESTED")

            forged = dict(evidence)
            forged.pop("visual_validation")
            self.assertEqual(reporter._pipeline_visual_result(manifest, forged), "UNTESTED")
            manifest["gates"]["backend"] = {"status": "FAIL"}
            status, _, failed_evidence, _ = reporter.evaluate(manifest, case)
            self.assertEqual(status, "FAIL")
            self.assertEqual(reporter._pipeline_visual_result(manifest, failed_evidence), "PASS")

    def test_campaign_index_rejects_missing_case(self):
        with tempfile.TemporaryDirectory() as temp:
            models = Path(temp) / "models"
            models.mkdir()
            campaign = models / "campaign.json"
            campaign.write_text(json.dumps({"schema_version": 1, "cases": ["one", "two"]}))
            with self.assertRaisesRegex(ValueError, "report is absent for one"):
                indexer.build_index(campaign)

    def test_vision13_suite_stages_all_cases_without_claiming_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = json.loads((ROOT_DIR / "assets" / "vision13-cases.json").read_text())
            self.assertEqual(len(expected["cases"]), 13)
            campaign = root / "input.json"
            campaign.write_text(json.dumps({"schema_version": 1, "suite": "vision13",
                                            "cases": expected["cases"]}), encoding="utf-8")
            models = root / "models"
            preparer.prepare(campaign, models)
            staged = json.loads((models / "campaign.json").read_text())
            self.assertEqual(staged["suite"], "vision13")
            self.assertEqual(len(staged["cases"]), 13)
            for case in staged["cases"]:
                manifest = json.loads((models / case / "case-evidence.json").read_text())
                self.assertTrue(manifest["input_blockers"])
                self.assertTrue(all(gate["status"] == "UNTESTED"
                                    for gate in manifest["gates"].values()))

            altered = {"schema_version": 1, "suite": "vision13",
                       "cases": expected["cases"][:-1]}
            campaign.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly its 13"):
                preparer.prepare(campaign, root / "another-models")

    def test_vision14_adds_one_primary_imagenet_classifier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry = json.loads((ROOT_DIR / "assets" / "vision13-cases.json").read_text())
            cases = registry["cases"] + [registry["vision14_additional_case"]]
            self.assertEqual(len(cases), 14)
            self.assertEqual(cases[-1], {"case_id": "mobilenetv2-imagenet",
                                         "task": "classification"})
            campaign = root / "input.json"
            campaign.write_text(json.dumps({"schema_version": 1, "suite": "vision14",
                                            "cases": cases}), encoding="utf-8")
            preparer.prepare(campaign, root / "models")
            staged = json.loads((root / "models" / "campaign.json").read_text())
            self.assertEqual(staged["suite"], "vision14")
            self.assertEqual(len(staged["cases"]), 14)

            cases[-1] = {"case_id": "mobilenetv2-imagenet", "task": "detection"}
            campaign.write_text(json.dumps({"schema_version": 1, "suite": "vision14",
                                            "cases": cases}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly its 14"):
                preparer.prepare(campaign, root / "wrong-models")

    def test_vision13_index_requires_all_reports_and_separate_visual_status(self):
        with tempfile.TemporaryDirectory() as temp:
            models = Path(temp) / "models"
            models.mkdir()
            cases = json.loads((ROOT_DIR / "assets" / "vision13-cases.json").read_text())["cases"]
            campaign = models / "campaign.json"
            campaign.write_text(json.dumps({"schema_version": 1, "suite": "vision13",
                                            "cases": [case["case_id"] for case in cases]}),
                                encoding="utf-8")
            for case in cases:
                case_id = case["case_id"]
                reports = models / case_id / "reports"
                reports.mkdir(parents=True)
                pdf = write(reports / f"status_report_{case_id}.pdf", b"%PDF-1.4\nstatus")
                result = {"case_id": case_id, "task": case["task"],
                          "functional_status": "BLOCKED", "capacity_status": "UNTESTED",
                          "pipeline_visual_result": "PASS" if case_id == "lpd-us" else "UNTESTED",
                          "pdf": (str(Path(temp) / "unmounted-container" / "models" /
                                      case_id / "reports" / Path(pdf["path"]).name)
                                  if case_id == "trafficcamnet" else pdf["path"]),
                          "pdf_sha256": pdf["sha256"]}
                (reports / "report-result.json").write_text(json.dumps(result), encoding="utf-8")
            payload = indexer.build_index(campaign)
            self.assertEqual(len(payload["cases"]), 13)
            lpd = next(row for row in payload["cases"] if row["case_id"] == "lpd-us")
            self.assertEqual(lpd["pipeline_visual_result"], "PASS")
            self.assertEqual(lpd["functional_status"], "BLOCKED")
            self.assertIn("Pipeline + visual", (models / "campaign-report-index.md").read_text())
            relocated = next(row for row in payload["cases"] if row["case_id"] == "trafficcamnet")
            self.assertEqual(relocated["pdf"], "trafficcamnet/reports/status_report_trafficcamnet.pdf")
            relocated_result = models / "trafficcamnet" / "reports" / "report-result.json"
            recorded = json.loads(relocated_result.read_text(encoding="utf-8"))
            recorded["pdf_sha256"] = "0" * 64
            relocated_result.write_text(json.dumps(recorded), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "PDF digest differs"):
                indexer.build_index(campaign)
            recorded["pdf_sha256"] = relocated["pdf_sha256"]
            relocated_result.write_text(json.dumps(recorded), encoding="utf-8")

            broken = json.loads(campaign.read_text(encoding="utf-8"))
            broken["cases"] = broken["cases"][:-1]
            campaign.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly its 13"):
                indexer.build_index(campaign)


if __name__ == "__main__":
    unittest.main()
