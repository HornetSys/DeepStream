# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the current review, without GPU or network."""

from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts/multitask/model"))
sys.path.insert(0, str(ROOT / "scripts/multitask/visual"))
sys.path.insert(0, str(ROOT / "scripts/multitask/deepstream"))

import emit_commands
import generate_configs
import model_pipeline
import render_trt_visual
import test_step8_reports as reports
import test_unified_task_routing as routing
import visual_io_probe


class TestLatestReviewRepairs(unittest.TestCase):
    def test_unsupported_task_rejects_before_irrelevant_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            invalid = root / "invalid.json"
            invalid.write_text("not-json")
            for metadata in (invalid, root / "missing.json"):
                result = subprocess.run([sys.executable, str(routing.ROUTER), str(root / "missing.onnx"),
                    "--task", "optical_flow", "--metadata", str(metadata)],
                    capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 3, result.stderr)
                self.assertEqual(json.loads(result.stdout)["capability_status"], "BLOCKED")

    def test_configs_require_artifact_and_emit_graph_plan_for_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = root / "model.engine"
            engine.write_bytes(b"structural-test-only; never deserialized")
            snapshot = root / "environment.json"
            snapshot.write_text("{}")
            metadata = {
                "input_name": "input", "input_shape": [1, 3, 8, 8], "output_names": ["scores"],
                "environment_snapshot": str(snapshot), "environment": {"deepstream": "fixture"},
                "task_contract": {"class_count": 3, "label_map": {"0": "a", "1": "b", "2": "c"},
                                  "score_semantics": "logits", "top_k": 1, "threshold": 0.01},
                "pipeline_graph": {"status": "PLANNED", "nodes": [{"id": "source", "role": "input"},
                                    {"id": "infer", "role": "model"}],
                    "edges": [{"from": "source", "to": "infer", "contract": "frames"}],
                    "input_ownership": "full-frame", "coordinate_transforms": "resize",
                    "decoder": "classification", "renderer": "labels", "structured_output": "scores",
                    "evidence": {"fixture": "caller-media"}},
            }
            manifest = root / "route.json"
            manifest.write_text(json.dumps(model_pipeline.route_model(engine, task="classification", metadata=metadata).to_dict()))
            candidate = root / "candidate"
            generated = subprocess.run([sys.executable, str(routing.ADAPTER_GENERATOR), str(manifest), str(candidate)],
                                       capture_output=True, text=True, check=False)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            candidate_manifest = candidate / "manifest-with-adapter.json"
            script = ROOT / "scripts/multitask/deepstream/generate_configs.py"
            with patch.object(sys, "argv", [str(script), str(candidate_manifest),
                              str(root / "invalid-configs"), "--allow-implemented-adapter"]), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
                generate_configs.main()
            self.assertEqual(rejected.exception.code, 2)
            self.assertFalse((root / "invalid-configs").exists())
            output = root / "configs"
            with patch.object(sys, "argv", [str(script), str(candidate_manifest), str(output),
                              "--engine", str(engine), "--allow-implemented-adapter"]):
                self.assertEqual(generate_configs.main(), 0)
            self.assertTrue((output / "nvinfer.txt").is_file())
            config_text = (output / "nvinfer.txt").read_text()
            self.assertNotIn("Raw network type 100", config_text)
            self.assertNotIn("generated postprocess-contract.json", config_text)
            graph = json.loads((output / "pipeline-plan.json").read_text())
            self.assertEqual(graph, json.loads(candidate_manifest.read_text())["pipeline_graph"])
            self.assertEqual(json.loads((output / "adapter-validation-required.json").read_text())["production_ready"], False)

    def test_underscore_aliases_route_explicit_and_auto(self):
        helper = routing.TestUnifiedTaskRouting()
        for task in ("object_detection", "image_classification", "semantic_segmentation",
                     "2d_pose", "monocular_depth"):
            for requested in (task, "auto"):
                with self.subTest(task=task, requested=requested):
                    result, manifest = helper.route(requested, helper.runnable_metadata(task=task))
                    self.assertIn(result.returncode, (0, 4), result.stderr)
                    self.assertNotEqual(manifest.get("status"), "unsupported")
        result, _ = helper.route("optical_flow", create_source=False)
        self.assertEqual(result.returncode, 3)

    def test_native_classification_default_is_consistent(self):
        manifest = model_pipeline.manifest_from_dict({
            "source": "/data/model.onnx", "source_format": "onnx",
            "task": "image-classification", "model_name": "classifier",
            "input_name": "input", "input_shape": [1, 3, 8, 8], "output_names": ["scores"],
        })
        self.assertEqual(manifest.postprocess, "native")
        self.assertEqual(manifest.capability_status, "READY")

    def test_supplied_adapter_cannot_bypass_readiness_for_native_classifier(self):
        for contract in ({"implementation_status": "implemented", "validation_status": "generated-untested"},
                         {"implementation_status": "tested"}):
            status, reasons = model_pipeline._capability_for(
                "classification", "classification", "image", "native", "", (), False, (), contract,
            )
            self.assertEqual(status, "ADAPTER_REQUIRED")
            self.assertTrue(reasons)

    def test_emitted_paths_are_real_and_quote_installed_roots(self):
        for relative in ("deepstream/generate_adapter.py", "deepstream/generate_configs.py",
                         "validation/validate_artifact.py"):
            command = emit_commands._script(relative)
            self.assertTrue(Path(shlex.split(command)[0]).is_file())
        with patch.object(emit_commands, "__file__", "/fresh system/.claude/skills/vision/scripts/multitask/model/emit_commands.py"):
            self.assertEqual(shlex.split(emit_commands._script("deepstream/generate_configs.py")),
                             ["/fresh system/.claude/skills/vision/scripts/multitask/deepstream/generate_configs.py"])

    def test_unknown_raw_dtype_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported.*dtype"):
            render_trt_visual.raw_dtype("fp8_unknown")
        for token in ("bool", "uint16", "uint32", "uint64", "float64", "fp32", "fp16"):
            self.assertIsInstance(render_trt_visual.raw_dtype(token), np.dtype)

    @staticmethod
    def visual_fixture(root):
        image = root / "case.png"
        Image.new("RGB", (8, 8), "red").save(image)
        montage = root / "visual-montage.png"
        Image.new("RGB", (8, 8), "blue").save(montage)
        report = root / "visual-report.json"
        report.write_text(json.dumps({"entries": [{"id": "case", "case_image": str(image),
            "result": "UNTESTED", "input_visual_status": "UNTESTED", "output_visual_status": "UNTESTED"}]}))
        inventory = root / "inventory.json"
        inventory.write_text(json.dumps({"entries": [{"id": "case", "task": "classification",
            "inspection": {"inputs": [{"name": "input", "shape": [1, 3, 8, 8], "dtype": 1}],
                           "outputs": [{"name": "scores"}]}}]}))
        raw = root / "raw"
        raw.mkdir()
        np.array([0.1, 0.9], dtype=np.float32).tofile(raw / "scores.output.1.2.fp32.raw")
        return report, inventory, raw, (image, montage, report)

    def test_later_invalid_mapping_does_not_change_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            report, inventory, raw, paths = self.visual_fixture(Path(td))
            before = [path.read_bytes() for path in paths]
            with self.assertRaises(ValueError):
                render_trt_visual.attach(report, inventory, [f"case={raw}", "missing=nowhere"], confirm_overwrite=True)
            self.assertEqual(before, [path.read_bytes() for path in paths])

    def test_visual_replacement_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as td:
            report, inventory, raw, paths = self.visual_fixture(Path(td))
            before = [path.read_bytes() for path in paths]
            replace = os.replace
            calls = 0

            def fail_once(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated commit failure")
                replace(source, target)

            with patch.object(render_trt_visual.os, "replace", side_effect=fail_once), self.assertRaises(OSError):
                render_trt_visual.attach(report, inventory, [f"case={raw}"], confirm_overwrite=True)
            self.assertEqual(before, [path.read_bytes() for path in paths])

    def test_visual_success_commits_all_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            report, inventory, raw, paths = self.visual_fixture(Path(td))
            before = [path.read_bytes() for path in paths]
            counts = render_trt_visual.attach(report, inventory, [f"case={raw}"], confirm_overwrite=True)
            self.assertEqual(counts["pass"], 1)
            self.assertTrue(all(path.read_bytes() != old for path, old in zip(paths, before)))

    def test_probe_persists_absolute_case_paths_from_relative_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inventory = root / "inventory.json"
            inventory.write_text('{"entries": []}')
            captured = []
            with patch.object(sys, "argv", ["probe", str(inventory), "--output-dir", "relative-output"]), \
                 patch.dict(sys.modules, {"onnxruntime": SimpleNamespace()}), \
                 patch.object(visual_io_probe, "_make_montage", side_effect=lambda rows, path: captured.append(path)), \
                 redirect_stdout(io.StringIO()):
                previous = Path.cwd()
                try:
                    os.chdir(root)
                    self.assertEqual(visual_io_probe.main(), 1)  # An empty inventory is not a PASS.
                finally:
                    os.chdir(previous)
            self.assertTrue(captured[0].is_absolute())

    def test_index_rejects_nonobject_json_and_invalid_pdf_types(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            campaign = root / "campaign.json"
            for data in ([], None, "bad"):
                campaign.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    reports.indexer.build_index(campaign)
            campaign.write_text('{"schema_version": 1, "cases": ["case"]}')
            result = root / "case/reports/report-result.json"
            result.parent.mkdir(parents=True)
            for data in ([], None, {"functional_status": "BLOCKED", "capacity_status": "UNTESTED", "pdf": None}):
                result.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    reports.indexer.build_index(campaign)

    def test_malformed_optional_report_fields_produce_status_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            case = Path(td) / "case"
            case.mkdir()
            for stages, findings in (([None], None), ("not-an-array", 7)):
                manifest = {"schema_version": 1, "case_id": "case", "task": "classification",
                            "upstream_stages": stages, "preflight_findings": findings}
                outcome, errors, evidence, capacity = reports.reporter.evaluate(manifest, case)
                self.assertNotEqual(outcome, "PASS")
                self.assertIn("upstream stage contract is invalid", errors)
                path = reports.reporter._write_markdown(manifest, case, outcome, errors, evidence, capacity, "status")
                self.assertIn("upstream stage contract is invalid", path.read_text())

    def test_decode_findings_already_propagate_to_preparer_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model.onnx"
            media = root / "input.mp4"
            model.write_bytes(b"model")
            media.write_bytes(b"fixture")
            request = root / "input.json"
            request.write_text(json.dumps({"schema_version": 1, "cases": [{
                "case_id": "case", "task": "classification", "source": str(model),
                "inputs": [{"path": str(media)}]}]}))
            for code, finding in ((1, "source frame decode failed"), (0, "source frame decode produced no image")):
                with patch.object(reports.preparer.subprocess, "run", return_value=SimpleNamespace(returncode=code)):
                    output = root / f"models-{code}"
                    summary = reports.preparer.prepare(request, output, capture_source_frame=True)
                manifest = json.loads((output / "case/case-evidence.json").read_text())
                self.assertIn(finding, summary["cases"][0]["preflight_findings"])
                self.assertEqual(summary["cases"][0]["preflight_findings"], manifest["preflight_findings"])

    def test_glib_signal_dispatch_and_watch_cleanup(self):
        source = (ROOT / "scripts/multitask/deepstream/render_segmentation_logits.cpp").read_text()
        self.assertNotIn("signal(SIG", source)
        self.assertIn("g_unix_signal_add(SIGINT", source)
        self.assertIn("g_unix_signal_add(SIGTERM", source)
        self.assertIn("g_source_remove(interrupt_watch)", source)
        self.assertIn("g_source_remove(terminate_watch)", source)


if __name__ == "__main__":
    unittest.main()
