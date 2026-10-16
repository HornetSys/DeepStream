# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np


SKILL_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = SKILL_ROOT / "scripts" / "multitask" / "validation"
MODEL_DIR = SKILL_ROOT / "scripts" / "multitask" / "model"
VISUAL_DIR = SKILL_ROOT / "scripts" / "multitask" / "visual"
DEEPSTREAM_DIR = SKILL_ROOT / "scripts" / "multitask" / "deepstream"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(VISUAL_DIR))
sys.path.insert(0, str(DEEPSTREAM_DIR))

from compare_outputs import main as compare_outputs_main  # noqa: E402
from emit_commands import _supplied_engine_validation_command, emit  # noqa: E402
from model_pipeline import (  # noqa: E402
    ManifestError,
    manifest_from_dict,
    pipeline_graph_errors,
    route_model,
)
from validate_adapter_readiness import validate as validate_adapter_readiness  # noqa: E402
from validate_semantic_oracle import validate as validate_semantic_oracle  # noqa: E402
from render_trt_visual import read_output  # noqa: E402
from type_adapter_runtime import (  # noqa: E402
    AdapterContractError,
    build_pipeline_task,
    decode_task,
    input_adapter_task,
    render_task,
    run_pipeline_task,
    validate_task,
)
from run_type_adapter_pipeline import (  # noqa: E402
    RunnerError,
    _finite_prediction_scores,
    _parse_pipeline,
    _require_validate_hook,
)
from validate_artifact import validate as validate_artifact  # noqa: E402
from validate_rendered_output import _gstreamer_video_check, _video_check  # noqa: E402


SEGMENTATION_CONTRACT = {
    "mask_layout": "NCHW",
    "class_count": 2,
    "resize_alignment": "nearest",
    "ignore_label": -1,
}


class TestUnifiedReviewFixes(unittest.TestCase):
    def test_native_detection_manifest_round_trips(self):
        with tempfile.TemporaryDirectory() as td:
            engine = Path(td) / "detector.engine"
            engine.write_bytes(b"fixture")
            manifest = route_model(engine, task="object-detection")
            loaded = manifest_from_dict(manifest.to_dict())

        self.assertEqual(loaded.status, "ready-for-validation")
        self.assertEqual(loaded.capability_status, "READY")
        self.assertEqual(loaded.delegate_skill, "")
        self.assertIn("KITTI gate", " ".join(loaded.validation))

    def test_generated_adapter_manifest_contains_pipeline_graph(self):
        generator = (
            SKILL_ROOT
            / "scripts"
            / "multitask"
            / "deepstream"
            / "generate_adapter.py"
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = root / "segmenter.engine"
            engine.write_bytes(b"fixture")
            manifest = route_model(
                engine,
                task="semantic-segmentation",
                metadata={
                    "input_shape": [1, 3, 224, 224],
                    "output_names": ["mask"],
                    "mean": [0.1, 0.2, 0.3],
                    "std": [0.2, 0.3, 0.4],
                    "preprocess_units": "normalized",
                    "task_contract": SEGMENTATION_CONTRACT,
                },
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
            output_dir = root / "adapter"
            completed = subprocess.run(
                [sys.executable, str(generator), str(manifest_path), str(output_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            plan = json.loads((output_dir / "pipeline-plan.json").read_text())
            merged = json.loads(
                (output_dir / "manifest-with-adapter.json").read_text()
            )
            contract = merged["adapter_contract"]
            self.assertEqual(contract["implementation_status"], "implemented")
            self.assertEqual(contract["validation_status"], "generated-untested")
            self.assertEqual(
                contract["semantic_oracle_validator"],
                "validate_semantic_oracle.py",
            )
            self.assertEqual(contract["preprocess_strategy"], "adapter-input-tensor")
            self.assertNotIn("preprocess_library", contract["required_hooks"])
            self.assertEqual(contract["readiness_status"], "")
            self.assertIn("semantic_validation", contract["validation_evidence"])
            self.assertIn("visual_validation", contract["validation_evidence"])
            self.assertEqual(merged["pipeline_graph"], plan)
            self.assertEqual(pipeline_graph_errors(plan), [])
            self.assertTrue((output_dir / "type_adapter_runtime.py").is_file())
            self.assertTrue((output_dir / "validate_semantic_oracle.py").is_file())
            self.assertNotIn(
                "NotImplemented",
                (output_dir / "adapter.py").read_text(encoding="utf-8"),
            )
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import numpy as np; import adapter; "
                        "o={'mask':np.array([[[[1,0],[0,1]],[[0,1],[1,0]]]],dtype=np.float32)}; "
                        "d=adapter.decode(o,{}); "
                        "r=adapter.render(np.zeros((8,8,3),dtype=np.uint8),d,{}); "
                        "assert d['task']=='segmentation' and r.shape==(8,8,3)"
                    ),
                ],
                cwd=output_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_classification_reference_runner_requires_native_preprocessing(self):
        generator = DEEPSTREAM_DIR / "generate_adapter.py"
        cases = (
            ([0.2, 0.2, 0.2], "nvinfer-native", True),
            ([0.2, 0.3, 0.4], "adapter-input-tensor", False),
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = root / "classifier.engine"
            engine.write_bytes(b"fixture")
            for index, (std, strategy, has_reference) in enumerate(cases):
                with self.subTest(std=std):
                    manifest = route_model(
                        engine,
                        task="image-classification",
                        metadata={
                            "input_shape": [1, 3, 224, 224],
                            "output_names": ["scores"],
                            "mean": [0.1, 0.2, 0.3],
                            "std": std,
                            "preprocess_units": "normalized",
                            "postprocess": "custom",
                            "task_contract": {
                                "class_count": 2,
                                "label_map": ["cat", "dog"],
                                "score_semantics": "logits",
                                "top_k": 1,
                                "threshold": 0.0,
                            },
                        },
                    )
                    manifest_path = root / f"manifest-{index}.json"
                    manifest_path.write_text(
                        json.dumps(manifest.to_dict()), encoding="utf-8"
                    )
                    output_dir = root / f"adapter-{index}"
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(generator),
                            str(manifest_path),
                            str(output_dir),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    generated = json.loads(
                        (output_dir / "manifest-with-adapter.json").read_text()
                    )
                    contract = generated["adapter_contract"]
                    self.assertEqual(contract["preprocess_strategy"], strategy)
                    self.assertEqual(
                        bool(contract["reference_pipeline_runner"]), has_reference
                    )
                    plan = build_pipeline_task(
                        {
                            "task": "classification",
                            "input_domain": "image",
                            "preprocess_strategy": strategy,
                        },
                        {},
                    )
                    expected_stage = (
                        "nvinfer-native-preprocess"
                        if has_reference
                        else "adapter-input-tensor"
                    )
                    self.assertIn(expected_stage, plan["stages"])

    def test_adapter_generator_rejects_non_image_input_domain(self):
        generator = DEEPSTREAM_DIR / "generate_adapter.py"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = root / "classifier.engine"
            engine.write_bytes(b"fixture")
            manifest = route_model(
                engine,
                task="image-classification",
                metadata={
                    "input_shape": [1, 3, 224, 224],
                    "output_names": ["scores"],
                    "input_domain": "audio",
                },
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest.to_dict()), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(generator),
                    str(manifest_path),
                    str(root / "adapter"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 2)
        diagnostic = json.loads(completed.stderr)
        self.assertEqual(diagnostic["status"], "BLOCKED")
        self.assertIn("input_domain='image'", " ".join(diagnostic["errors"]))

    def test_supplied_engine_command_uses_approval_gated_validator(self):
        manifest = SimpleNamespace(
            execution_approval={
                "approved": True,
                "approver": "caller",
                "approved_at": "2026-09-16T12:00:00Z",
                "trusted_root": "/data/models",
                "scope": "deserialize-supplied-tensorrt-engine",
                "engine_provenance": "/data/models/build.json",
            }
        )
        command = _supplied_engine_validation_command(
            manifest, "/data/models/model.engine", "/data/models/manifest.json"
        )
        self.assertIn("validate_artifact.py", command)
        self.assertIn("--trusted-root /data/models", command)
        self.assertIn("--engine-provenance /data/models/build.json", command)
        self.assertNotIn("trtexec --loadEngine", command)

    def test_supplied_engine_command_requires_explicit_approval(self):
        manifest = SimpleNamespace(execution_approval={})
        with self.assertRaises(ManifestError):
            _supplied_engine_validation_command(
                manifest, "/data/models/model.engine", "manifest.json"
            )

    def test_adapter_config_is_bound_to_each_precision_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "classifier.pt"
            model.write_bytes(b"fixture")
            manifest = route_model(
                model,
                task="image-classification",
                metadata={
                    "input_shape": [1, 3, 224, 224],
                    "output_names": ["scores"],
                    "postprocess": "custom",
                    "export_wrapper": "/tools/export_classifier.py",
                    "environment_snapshot": "environment.json",
                    "environment": {"deepstream": "9.1"},
                    "pipeline_graph": {
                        "status": "ADAPTER_REQUIRED",
                        "nodes": [
                            {"id": "source", "role": "video-source"},
                            {"id": "model", "role": "classification-inference"},
                        ],
                        "edges": [
                            {
                                "from": "source",
                                "to": "model",
                                "contract": "decoded-frame",
                            }
                        ],
                        "input_ownership": {"owner": "deepstream"},
                        "coordinate_transforms": {"policy": "frame-space"},
                        "decoder": "type-adapter",
                        "renderer": "classification-overlay",
                        "structured_output": "classification-json",
                        "evidence": {"plan": "manifest.json"},
                    },
                    "task_contract": {
                        "class_count": 2,
                        "label_map": ["cat", "dog"],
                        "score_semantics": "logits",
                        "top_k": 1,
                        "threshold": 0.0,
                    },
                },
            )
            self.assertEqual(manifest.capability_status, "ADAPTER_REQUIRED")
            manifest.adapter_contract["implementation_status"] = "implemented"
            plan = emit(
                manifest,
                output_dir=str(root / "out"),
                engine_path=str(root / "out" / "classifier.engine"),
                batch_size=1,
                manifest_path=str(root / "manifest.json"),
            )

        attempts = plan["precision_attempts"]
        self.assertGreaterEqual(len(attempts), 2)
        for attempt in attempts:
            self.assertIn(
                f"--engine {attempt['engine']}",
                attempt["config_command"],
            )
            self.assertIn("--allow-implemented-adapter", attempt["config_command"])
        self.assertFalse(
            any("--engine" in command for command in plan["adapter_commands"])
        )

    def test_raw_binding_name_may_contain_output_token(self):
        with tempfile.TemporaryDirectory() as td:
            raw_dir = Path(td)
            raw_path = raw_dir / "tensor.output.head.output.1.2.fp32.raw"
            np.asarray([1.0, 2.0], dtype=np.float32).tofile(raw_path)
            values, selected = read_output(raw_dir, "tensor.output.head")

        self.assertEqual(selected.name, raw_path.name)
        self.assertEqual(values.shape, (1, 2))
        np.testing.assert_allclose(values, [[1.0, 2.0]])

    def test_nonstandard_onxx_suffix_is_not_accepted_as_onnx(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model.onxx"
            model.write_bytes(b"fixture")
            manifest = route_model(
                model,
                task="image-classification",
                metadata={"input_shape": [1, 3, 224, 224], "output_names": ["output"]},
            )

        self.assertEqual(manifest.status, "needs-format-identification")
        self.assertEqual(manifest.capability_status, "BLOCKED")

    def test_compare_outputs_emits_readiness_bound_evidence(self):
        comparator = VALIDATION_DIR / "compare_outputs.py"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            model = root / "model.onnx"
            np.savez(reference, scores=np.asarray([[0.1, 0.9]], dtype=np.float32))
            np.savez(candidate, scores=np.asarray([[0.1, 0.9]], dtype=np.float32))
            model.write_bytes(b"model-fixture")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({"task": "classification", "source": model.name}),
                encoding="utf-8",
            )
            fixture = root / "fixture.mp4"
            fixture.write_bytes(b"fixture")
            output = root / "backend-parity.json"
            arguments = [
                str(comparator),
                str(reference),
                str(candidate),
                "--manifest",
                str(manifest_path),
                "--input-fixture",
                str(fixture),
                "--campaign-id",
                "classification-smoke-001",
                "--reference-backend",
                "onnxruntime",
                "--candidate-backend",
                "tensorrt",
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", arguments):
                return_code = compare_outputs_main()
            self.assertEqual(return_code, 0)
            evidence = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(evidence["schema_version"], 1)
            self.assertEqual(evidence["producer"], "compare_outputs.py")
            self.assertEqual(evidence["task"], "classification")
            self.assertEqual(
                evidence["source_hash"],
                hashlib.sha256(model.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                evidence["input_fixture_sha256"],
                hashlib.sha256(fixture.read_bytes()).hexdigest(),
            )
            self.assertEqual(evidence["reference_backend"], "onnxruntime")
            self.assertEqual(evidence["candidate_backend"], "tensorrt")
            self.assertTrue(evidence["metric"]["passed"])
            self.assertTrue(evidence["metric"]["finite"])

            repeat_stdout = io.StringIO()
            with patch.object(sys, "argv", arguments), patch.object(
                sys, "stdout", repeat_stdout
            ):
                repeat_code = compare_outputs_main()
            self.assertEqual(repeat_code, 2)
            failure = json.loads(repeat_stdout.getvalue())
            self.assertEqual(failure["status"], "FAIL")
            self.assertIn("File exists", failure["error"])

    def test_reference_runner_requires_validate_hook_for_oracle(self):
        with self.assertRaisesRegex(RunnerError, "callable validate hook"):
            _require_validate_hook(SimpleNamespace())
        self.assertTrue(
            callable(_require_validate_hook(SimpleNamespace(validate=lambda *_: {})))
        )

    def test_reference_runner_rejects_malformed_prediction_scores(self):
        self.assertTrue(_finite_prediction_scores([{"score": "0.75"}]))
        self.assertFalse(_finite_prediction_scores([{"score": "nan"}]))
        for predictions in ([{}], [{"score": None}], ["not-an-object"]):
            with self.subTest(predictions=predictions):
                with self.assertRaisesRegex(RunnerError, "classification prediction"):
                    _finite_prediction_scores(predictions)

    def test_artifact_source_is_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "detector.onnx"
            source.write_bytes(b"fixture")
            manifest = route_model(source, task="object-detection").to_dict()
            manifest["source"] = source.name
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = validate_artifact(
                manifest_path,
                onnx_path=None,
                engine_path=None,
                run_trtexec=False,
            )

        self.assertEqual(result["gates"]["source"]["status"], "PASS")
        self.assertEqual(result["gates"]["source"]["detail"], str(source))

    def test_pipeline_status_requires_every_named_gate(self):
        spec = {
            "task": "classification",
            "input_domain": "image",
            "preprocess_strategy": "nvinfer-native",
        }
        passing_gates = {
            "startup": "PASS",
            "nonzero-buffers": "PASS",
            "task-metadata": "PASS",
            "eos": "PASS",
            "structured-output": "PASS",
            "visible-overlay": "PASS",
        }
        result = run_pipeline_task(
            spec,
            {
                "run_deepstream": lambda _plan, _context: {
                    "status": "PASS",
                    "gate_statuses": {**passing_gates, "visible-overlay": "FAIL"},
                }
            },
        )
        self.assertEqual(result["deepstream_status"], "FAIL")
        self.assertEqual(
            result["required_gates"],
            [
                "startup",
                "nonzero-buffers",
                "task-metadata",
                "eos",
                "structured-output",
                "visible-overlay",
            ],
        )
        with self.assertRaisesRegex(AdapterContractError, "gate_statuses mapping"):
            run_pipeline_task(
                spec,
                {"run_deepstream": lambda _plan, _context: {"status": "PASS"}},
            )

    def test_reference_runner_has_a_parent_side_timeout(self):
        spec = {
            "task": "classification",
            "input_domain": "image",
            "preprocess_strategy": "nvinfer-native",
            "reference_pipeline_runner": "/tmp/reference-runner.py",
        }
        context = {
            "manifest": "/tmp/manifest.json",
            "nvinfer_config": "/tmp/nvinfer.txt",
            "input": "/tmp/input.mp4",
            "output_dir": "/tmp/evidence",
            "campaign_id": "bounded-run",
            "timeout_seconds": 1,
        }
        with patch(
            "type_adapter_runtime.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["reference-runner"], 16),
        ) as run_mock:
            with self.assertRaisesRegex(AdapterContractError, "parent timeout of 16"):
                run_pipeline_task(spec, context)
        self.assertEqual(run_mock.call_args.kwargs["timeout"], 16)

    def test_empty_classification_predictions_return_fail(self):
        result = validate_task(
            {"task": "classification"},
            {"predictions": []},
            {"class_id": 1},
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIsNone(result["metric"]["actual_class_id"])

    def test_gstreamer_parse_error_maps_to_runner_error(self):
        class FakeGError(Exception):
            pass

        def fail_parse(_description):
            raise FakeGError("missing element")

        gst = SimpleNamespace(parse_launch=fail_parse)
        glib = SimpleNamespace(GError=FakeGError)
        with self.assertRaisesRegex(
            RunnerError,
            "DeepStream pipeline construction failed: missing element",
        ):
            _parse_pipeline(gst, glib, "pipeline")

    def test_detection_label_gate_rejects_missing_architecture(self):
        gate = SKILL_ROOT / "scripts" / "model" / "config-to-labels.py"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "config.json"
            labels = root / "labels.txt"
            config.write_text(
                json.dumps({"id2label": {"0": "person"}}), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(gate),
                    "--config",
                    str(config),
                    "--labels",
                    str(labels),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("no 'architectures' field", completed.stderr)

    def test_single_stream_missing_video_is_blocked_before_gstreamer(self):
        runner = SKILL_ROOT / "scripts" / "deepstream" / "ds-single-stream.sh"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            completed = subprocess.run(
                [
                    "bash",
                    str(runner),
                    str(root / "config.txt"),
                    str(root / "output.mp4"),
                    str(root / "missing.mp4"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 4)
        self.assertIn("input video not found", completed.stderr)

    def test_adapter_promotion_requires_hashed_passing_evidence(self):
        generator = (
            SKILL_ROOT
            / "scripts"
            / "multitask"
            / "deepstream"
            / "generate_adapter.py"
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = root / "bundle"
            bundle.mkdir()
            engine = bundle / "classifier.engine"
            engine.write_bytes(b"fixture-engine")
            initial = route_model(
                engine,
                task="image-classification",
                metadata={
                    "input_shape": [1, 3, 224, 224],
                    "output_names": ["scores"],
                    "postprocess": "custom",
                    "source_hash": "a" * 64,
                    "task_contract": {
                        "class_count": 2,
                        "label_map": ["cat", "dog"],
                        "score_semantics": "logits",
                        "top_k": 1,
                        "threshold": 0.0,
                    },
                },
            )
            initial_path = bundle / "initial.json"
            initial_path.write_text(
                json.dumps(initial.to_dict()), encoding="utf-8"
            )
            adapter_dir = bundle / "adapter"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(generator),
                    str(initial_path),
                    str(adapter_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            manifest_path = adapter_dir / "implemented.json"
            manifest = json.loads(
                (adapter_dir / "manifest-with-adapter.json").read_text()
            )
            evidence_dir = bundle / "evidence"
            evidence_dir.mkdir()
            evidence_paths = {}
            input_path = evidence_dir / "input_fixture.bin"
            rendered_path = evidence_dir / "rendered_output.jpg"
            input_path.write_bytes(b"real-input-fixture")
            rendered_path.write_bytes(b"rendered-frame-fixture")
            evidence_paths["input_fixture"] = str(input_path)
            evidence_paths["rendered_output"] = str(rendered_path)
            for field in (
                "backend_parity",
                "deepstream_runtime",
                "structured_output",
                "semantic_validation",
                "overlay_evidence",
                "visual_validation",
            ):
                path = evidence_dir / f"{field}.json"
                path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
                evidence_paths[field] = str(path)
            manifest["adapter_contract"]["validation_evidence"] = evidence_paths
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            status_path = adapter_dir / "adapter-status.json"
            promoted_path = adapter_dir / "manifest-tested.json"
            weak_code, weak_result = validate_adapter_readiness(
                manifest_path,
                status_path=status_path,
                promoted_manifest_path=promoted_path,
            )
            self.assertEqual(weak_code, 4)
            self.assertIn("schema_version", " ".join(weak_result["errors"]))

            input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
            rendered_hash = hashlib.sha256(rendered_path.read_bytes()).hexdigest()
            campaign = "classification-smoke-001"
            source_hash = manifest["source_hash"]

            def common(producer):
                return {
                    "schema_version": 1,
                    "campaign_id": campaign,
                    "producer": producer,
                    "task": "classification",
                    "source_hash": source_hash,
                    "input_fixture_sha256": input_hash,
                }

            structured = {
                **common("deepstream-type-adapter-runner"),
                "identity": {"frame_id": 7, "roi_id": "full-frame"},
                "result": {
                    "predictions": [
                        {"class_id": 1, "label": "dog", "score": 0.99}
                    ]
                },
            }
            structured_path = Path(evidence_paths["structured_output"])
            structured_path.write_text(json.dumps(structured), encoding="utf-8")
            structured_hash = hashlib.sha256(structured_path.read_bytes()).hexdigest()

            payloads = {
                "backend_parity": {
                    **common("compare_outputs.py"),
                    "status": "PASS",
                    "reference_backend": "onnxruntime",
                    "candidate_backend": "tensorrt",
                    "metric": {"name": "top1_agreement", "passed": True, "finite": True},
                },
                "deepstream_runtime": {
                    **common("deepstream-type-adapter-runner"),
                    "producer_sha256": hashlib.sha256(
                        (
                            DEEPSTREAM_DIR / "run_type_adapter_pipeline.py"
                        ).read_bytes()
                    ).hexdigest(),
                    "status": "PASS",
                    "startup": True,
                    "buffer_count": 3,
                    "eos": True,
                    "exit_code": 0,
                    "task_metadata": {"count": 3, "tensor_names": ["scores"]},
                    "numeric_assertion": {"passed": True, "finite": True},
                    "structured_output_sha256": structured_hash,
                    "rendered_output_sha256": rendered_hash,
                },
                "semantic_validation": {
                    **common("adapter-semantic-validator"),
                    "status": "PASS",
                    "metric": {"actual_class_id": 1, "expected_class_id": 1},
                    "oracle_sha256": "b" * 64,
                    "structured_output_sha256": structured_hash,
                },
                "overlay_evidence": {
                    **common("adapter-renderer"),
                    "frame_id": 7,
                    "roi_id": "full-frame",
                    "renderer": "NvDsDisplayMeta classification overlay",
                    "overlay_fields": ["class_id", "label", "score"],
                    "structured_output_sha256": structured_hash,
                    "rendered_frame_sha256": rendered_hash,
                },
                "visual_validation": {
                    **common("validate_rendered_output.py"),
                    "status": "RENDERED_PASS",
                    "visual_status": "PASS",
                    "deepstream_status": "PASS",
                    "pipeline_visual_result": "PASS",
                    "passed": True,
                    "checks": {
                        name: {"status": "PASS"}
                        for name in (
                            "decoded_frame",
                            "structured_output",
                            "overlay_difference",
                            "overlay_provenance",
                            "video",
                        )
                    },
                    "structured_output_sha256": structured_hash,
                    "rendered_output_sha256": rendered_hash,
                },
            }
            for field, payload in payloads.items():
                Path(evidence_paths[field]).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            runtime_path = Path(evidence_paths["deepstream_runtime"])
            tampered_runtime = dict(payloads["deepstream_runtime"])
            tampered_runtime["producer"] = "untrusted-writer"
            runtime_path.write_text(
                json.dumps(tampered_runtime), encoding="utf-8"
            )
            bad_code, bad_result = validate_adapter_readiness(
                manifest_path,
                status_path=adapter_dir / "rejected-status.json",
                promoted_manifest_path=adapter_dir / "rejected-manifest.json",
            )
            self.assertEqual(bad_code, 4)
            self.assertIn("producer must equal", " ".join(bad_result["errors"]))
            runtime_path.write_text(
                json.dumps(payloads["deepstream_runtime"]), encoding="utf-8"
            )

            status_path = adapter_dir / "adapter-status.json"
            promoted_path = adapter_dir / "manifest-tested.json"
            code, result = validate_adapter_readiness(
                manifest_path,
                status_path=status_path,
                promoted_manifest_path=promoted_path,
            )
            self.assertEqual(code, 0, result)
            self.assertEqual(result["status"], "PASS")
            promoted_data = json.loads(promoted_path.read_text())
            self.assertEqual(promoted_data["pipeline_graph"]["status"], "RENDERED_PASS")
            for path in promoted_data["pipeline_graph"]["evidence"].values():
                self.assertFalse(Path(path).is_absolute())
            self.assertFalse(
                Path(
                    promoted_data["adapter_contract"]["readiness_status"]
                ).is_absolute()
            )
            status_data = json.loads(status_path.read_text())
            for item in status_data["evidence"].values():
                self.assertFalse(Path(item["path"]).is_absolute())
            promoted = manifest_from_dict(
                promoted_data,
                manifest_path=promoted_path,
            )
            self.assertEqual(promoted.capability_status, "READY")

            relocated = root / "relocated"
            shutil.move(bundle, relocated)
            relocated_promoted_path = relocated / "adapter" / "manifest-tested.json"
            relocated_data = json.loads(relocated_promoted_path.read_text())
            relocated_manifest = manifest_from_dict(
                relocated_data,
                manifest_path=relocated_promoted_path,
            )
            self.assertEqual(relocated_manifest.capability_status, "READY")
            with patch.object(
                Path,
                "read_bytes",
                side_effect=OSError("permission denied"),
            ):
                unreadable = manifest_from_dict(
                    relocated_data,
                    manifest_path=relocated_promoted_path,
                )
            self.assertEqual(unreadable.capability_status, "ADAPTER_REQUIRED")
            self.assertIn("unreadable", " ".join(unreadable.capability_reasons))

            (relocated / "evidence" / "input_fixture.bin").write_bytes(b"tampered")
            rechecked = manifest_from_dict(
                json.loads(relocated_promoted_path.read_text()),
                manifest_path=relocated_promoted_path,
            )
            self.assertEqual(rechecked.capability_status, "ADAPTER_REQUIRED")
            self.assertIn(
                "SHA-256 changed",
                " ".join(rechecked.capability_reasons),
            )

    def test_type_adapters_decode_render_and_validate_supported_tasks(self):
        cases = (
            (
                {
                    "task": "classification",
                    "output_names": ["scores"],
                    "task_contract": {
                        "class_count": 3,
                        "label_map": ["cat", "dog", "bird"],
                        "score_semantics": "logits",
                        "top_k": 2,
                        "threshold": 0.0,
                    },
                },
                {"scores": np.asarray([[0.1, 3.0, 0.2]], dtype=np.float32)},
                {"class_id": 1},
            ),
            (
                {
                    "task": "segmentation",
                    "output_names": ["mask"],
                    "task_contract": SEGMENTATION_CONTRACT,
                },
                {"mask": np.asarray([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]]], dtype=np.float32)},
                {"mask": [[0, 1], [1, 0]]},
            ),
            (
                {
                    "task": "pose",
                    "output_names": ["keypoints"],
                    "task_contract": {
                        "keypoint_names": ["left", "right"],
                        "skeleton_edges": [[0, 1]],
                        "coordinate_space": "pixels",
                    },
                },
                {"keypoints": np.asarray([[[4, 4, 0.9], [12, 12, 0.8]]], dtype=np.float32)},
                {"keypoints": [{"x": 4, "y": 4}, {"x": 12, "y": 12}], "tolerance": 0.01},
            ),
            (
                {
                    "task": "depth",
                    "output_names": ["depth"],
                    "task_contract": {
                        "units": "meters",
                        "inverse_depth": False,
                        "scale": 1.0,
                        "invalid_value": -1,
                        "alignment": "input-frame",
                        "render_min": 0.0,
                        "render_max": 10.0,
                    },
                },
                {"depth": np.asarray([[[[1, 2], [3, 4]]]], dtype=np.float32)},
                {"depth": [[1, 2], [3, 4]], "max_rmse": 0.0},
            ),
        )
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        for spec, outputs, oracle in cases:
            with self.subTest(task=spec["task"]):
                decoded = decode_task(spec, outputs, {})
                rendered = render_task(spec, frame, decoded, {})
                result = validate_task(spec, decoded, oracle)
                self.assertEqual(rendered.shape, frame.shape)
                self.assertTrue(np.any(rendered != frame))
                self.assertEqual(result["status"], "PASS")
                if spec["task"] == "classification":
                    white_text_pixels = np.all(
                        rendered[..., :3] == np.array([255, 255, 255]), axis=-1
                    )
                    self.assertTrue(np.any(white_text_pixels))

    def test_pose_and_depth_contracts_fail_closed_on_ambiguous_rendering(self):
        with self.assertRaisesRegex(AdapterContractError, "heatmap output must declare"):
            decode_task(
                {
                    "task": "pose",
                    "output_names": ["keypoints"],
                    "task_contract": {
                        "keypoint_names": ["left", "right"],
                        "skeleton_edges": [[0, 1]],
                        "coordinate_space": "pixels",
                    },
                },
                {"keypoints": np.ones((1, 2, 2, 2), dtype=np.float32)},
                {},
            )
        with self.assertRaisesRegex(AdapterContractError, "render_min and render_max"):
            decode_task(
                {
                    "task": "depth",
                    "output_names": ["depth"],
                    "task_contract": {
                        "units": "meters",
                        "inverse_depth": False,
                        "scale": 1.0,
                        "invalid_value": -1,
                        "alignment": "input-frame",
                    },
                },
                {"depth": np.ones((1, 1, 2, 2), dtype=np.float32)},
                {},
            )

    def test_reference_runner_requires_tensor_meta_visible_panel_and_shared_bindings(self):
        config_source = (DEEPSTREAM_DIR / "generate_configs.py").read_text(encoding="utf-8")
        runner_source = (DEEPSTREAM_DIR / "run_type_adapter_pipeline.py").read_text(encoding="utf-8")
        setup_source = (SKILL_ROOT / "setup.sh").read_text(encoding="utf-8")
        preflight_source = (SKILL_ROOT / "scripts" / "preflight.sh").read_text(encoding="utf-8")
        dsrun_source = (SKILL_ROOT / "scripts" / "dsrun.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'manifest.task == "classification" and bool(reference_runner)',
            config_source,
        )
        self.assertIn(
            ".claude/skills/deepstream-import-vision-model/scripts/multitask/deepstream/run_type_adapter_pipeline.py",
            (SKILL_ROOT / "references" / "adapter-development.md").read_text(
                encoding="utf-8"
            ),
        )
        for token in (
            "nvdsosd name=osd process-mode=0 display-text=1",
            "display_meta.num_rects = 1",
            '"DejaVu Sans"',
            "except (ImportError, OSError, ValueError, RunnerError, json.JSONDecodeError)",
        ):
            with self.subTest(token=token):
                self.assertIn(token, runner_source)
        self.assertIn("--system-site-packages", setup_source)
        self.assertIn("deepstream-pyds.pth", setup_source)
        self.assertIn(
            'PYDS_VENV="$DS_ROOT/sources/deepstream_python_apps/pyds"',
            setup_source,
        )
        self.assertIn('bash "$PYDS_INSTALLER" --build-bindings', setup_source)
        self.assertIn("GstPbutils", setup_source)
        self.assertIn("gir1.2-gst-plugins-base-1.0", setup_source)
        self.assertIn("GstPbutils", preflight_source)
        self.assertIn("RUNNER_IMPORTS=", preflight_source)
        self.assertIn('IMG="${1:-${DS_IMAGE:-}}"', preflight_source)
        self.assertNotIn(
            'IMG="${1:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}"',
            preflight_source,
        )
        self.assertIn('exec docker exec "$DS_CONTAINER"', dsrun_source)
        self.assertNotIn("docker run --rm", dsrun_source)

    def test_video_validation_does_not_require_ffprobe_when_gstreamer_passes(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "rendered.ogv"
            video.write_bytes(b"video-fixture")
            with patch(
                "validate_rendered_output._gstreamer_video_check",
                return_value={
                    "status": "PASS",
                    "detail": "video contains a decoded video stream",
                    "inspector": "gstreamer-discoverer",
                },
            ), patch("validate_rendered_output.shutil.which", return_value=None):
                result = _video_check(video)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["inspector"], "gstreamer-discoverer")

    def test_video_validation_ignores_unrelated_missing_audio_plugin(self):
        video_stream = SimpleNamespace(get_width=lambda: 1280, get_height=lambda: 720)
        info = SimpleNamespace(
            get_video_streams=lambda: [video_stream],
            get_result=lambda: SimpleNamespace(value_nick="missing-plugins"),
        )
        discoverer = SimpleNamespace(discover_uri=lambda _uri: info)
        gst = SimpleNamespace(SECOND=1, init=lambda _args: None)
        pbutils = SimpleNamespace(Discoverer=SimpleNamespace(new=lambda _timeout: discoverer))
        gi = ModuleType("gi")
        gi.require_version = lambda _name, _version: None
        repository = ModuleType("gi.repository")
        repository.Gst = gst
        repository.GstPbutils = pbutils

        with patch.dict(sys.modules, {"gi": gi, "gi.repository": repository}), patch(
            "validate_rendered_output._decode_video_frame",
            return_value=(True, "decoded video frame observed"),
        ) as decode:
            result = _gstreamer_video_check(Path("/tmp/video-with-aac.mp4"))

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["container_discovery"], "missing-plugins")
        decode.assert_called_once()

    def test_video_validation_rejects_undecodable_video_even_with_dimensions(self):
        video_stream = SimpleNamespace(get_width=lambda: 1280, get_height=lambda: 720)
        info = SimpleNamespace(
            get_video_streams=lambda: [video_stream],
            get_result=lambda: SimpleNamespace(value_nick="missing-plugins"),
        )
        discoverer = SimpleNamespace(discover_uri=lambda _uri: info)
        gst = SimpleNamespace(SECOND=1, init=lambda _args: None)
        pbutils = SimpleNamespace(Discoverer=SimpleNamespace(new=lambda _timeout: discoverer))
        gi = ModuleType("gi")
        gi.require_version = lambda _name, _version: None
        repository = ModuleType("gi.repository")
        repository.Gst = gst
        repository.GstPbutils = pbutils

        with patch.dict(sys.modules, {"gi": gi, "gi.repository": repository}), patch(
            "validate_rendered_output._decode_video_frame",
            return_value=(False, "no decoded video frame observed"),
        ):
            result = _gstreamer_video_check(Path("/tmp/undecodable.mp4"))

        self.assertEqual(result["status"], "FAIL")

    def test_video_validation_decodes_when_discoverer_raises_for_audio(self):
        def missing_plugin(_uri):
            raise RuntimeError("missing AAC plugin")

        discoverer = SimpleNamespace(discover_uri=missing_plugin)
        gst = SimpleNamespace(SECOND=1, init=lambda _args: None)
        pbutils = SimpleNamespace(Discoverer=SimpleNamespace(new=lambda _timeout: discoverer))
        gi = ModuleType("gi")
        gi.require_version = lambda _name, _version: None
        repository = ModuleType("gi.repository")
        repository.Gst = gst
        repository.GstPbutils = pbutils

        with patch.dict(sys.modules, {"gi": gi, "gi.repository": repository}), patch(
            "validate_rendered_output._decode_video_frame",
            return_value=(True, "decoded video frame observed"),
        ):
            result = _gstreamer_video_check(Path("/tmp/video-with-aac.mp4"))

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["inspector"], "gstreamer-video-decode")

    def test_input_adapter_applies_per_channel_normalization_and_layout(self):
        spec = {
            "input_domain": "image",
            "input_shape": [1, 3, 1, 1],
            "layout": "NCHW",
            "color_format": "RGB",
            "scale": 1.0 / 255.0,
            "mean": [0.1, 0.2, 0.3],
            "std": [0.2, 0.4, 0.5],
            "preprocess_units": "normalized",
        }
        pixel = np.asarray([[[51, 102, 153]]], dtype=np.uint8)
        result = input_adapter_task(
            spec,
            pixel,
            {"payload_layout": "HWC", "source_color_format": "RGB"},
        )
        self.assertEqual(result.shape, (1, 3, 1, 1))
        np.testing.assert_allclose(
            result[0, :, 0, 0],
            [(0.2 - 0.1) / 0.2, (0.4 - 0.2) / 0.4, (0.6 - 0.3) / 0.5],
            rtol=1e-6,
        )

    def test_hugging_face_helpers_reject_unpinned_revisions(self):
        for script_name, arguments in (
            ("hf-list-files.sh", ["org", "model"]),
            ("hf-download-config.sh", ["org", "model", "config.json"]),
            (
                "hf-download-file.sh",
                ["org", "model", "model.onnx", "models/model.onnx"],
            ),
        ):
            script = SKILL_ROOT / "scripts" / "model" / script_name
            completed = subprocess.run(
                ["bash", str(script), *arguments],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("immutable", completed.stderr)

    def test_semantic_oracle_contract_is_type_specific_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mask = root / "mask.npy"
            np.save(mask, np.zeros((2, 2), dtype=np.int32))
            oracle_path = root / "oracle.json"
            oracle = {
                "schema_version": 1,
                "task": "semantic-segmentation",
                "oracle_type": "reviewed",
                "provenance": {
                    "source": "fixture review v1",
                    "reviewer": "tester@example.com",
                    "reviewed_at": "2026-09-17T10:00:00-07:00",
                },
                "precision": "fp16",
                "acceptance": {
                    "metric": "mean_iou",
                    "comparator": "gte",
                    "threshold": 0.99,
                    "minimum_samples": 1,
                },
                "samples": [
                    {
                        "fixture_id": "frame-1",
                        "media": {"path": "video.mp4", "frame_index": 1},
                        "expected": {"mask_path": mask.name},
                    }
                ],
            }
            self.assertEqual(
                validate_semantic_oracle(
                    oracle,
                    oracle_path=oracle_path,
                    required_task="semantic-segmentation",
                ),
                [],
            )
            bundle = root / "bundle"
            outside = root / "outside"
            bundle.mkdir()
            outside.mkdir()
            outside_mask = outside / "mask.npy"
            np.save(outside_mask, np.zeros((2, 2), dtype=np.int32))
            (bundle / "artifacts").symlink_to(outside, target_is_directory=True)
            escaped_oracle = json.loads(json.dumps(oracle))
            escaped_oracle["samples"][0]["expected"]["mask_path"] = (
                "artifacts/mask.npy"
            )
            escaped_errors = validate_semantic_oracle(
                escaped_oracle,
                oracle_path=bundle / "oracle.json",
                required_task="semantic-segmentation",
            )
            self.assertTrue(
                any(
                    "escapes the oracle directory" in error
                    for error in escaped_errors
                ),
                escaped_errors,
            )
            oracle["acceptance"]["threshold"] = 0.5
            errors = validate_semantic_oracle(
                oracle,
                oracle_path=oracle_path,
                required_task="semantic-segmentation",
            )
            self.assertTrue(any("relaxes" in error for error in errors), errors)

    def test_missing_cli_inputs_fail_cleanly(self):
        scripts = (
            (
                DEEPSTREAM_DIR / "generate_adapter.py",
                ["/missing/manifest.json", "/tmp/adapter-output"],
            ),
            (
                VISUAL_DIR / "attach_visual_evidence.py",
                ["/missing/catalog.json", "/missing/report.json"],
            ),
        )
        for script, arguments in scripts:
            with self.subTest(script=script.name):
                completed = subprocess.run(
                    [sys.executable, str(script), *arguments],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertNotIn("Traceback", completed.stderr)
                diagnostic = json.loads(completed.stderr)
                self.assertEqual(diagnostic["status"], "BLOCKED")
                self.assertTrue(diagnostic["errors"])

    def test_config_generator_rejects_existing_destination(self):
        generator = DEEPSTREAM_DIR / "generate_configs.py"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = root / "detector.engine"
            engine.write_bytes(b"fixture")
            manifest = route_model(
                engine,
                task="object-detection",
                metadata={
                    "input_name": "images",
                    "input_shape": [1, 3, 640, 640],
                    "output_names": ["detections"],
                    "environment_snapshot": "environment.json",
                    "environment": {"deepstream": "9.1"},
                    "pipeline_graph": {
                        "status": "PLANNED",
                        "nodes": [
                            {"id": "source", "role": "video-source"},
                            {"id": "model", "role": "detection-inference"},
                        ],
                        "edges": [
                            {
                                "from": "source",
                                "to": "model",
                                "contract": "decoded-frame",
                            }
                        ],
                        "input_ownership": {"owner": "deepstream"},
                        "coordinate_transforms": {"policy": "frame-space"},
                        "decoder": "native-detection-parser",
                        "renderer": "bounding-box-overlay",
                        "structured_output": "KITTI",
                        "evidence": {"plan": "manifest.json"},
                    },
                },
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest.to_dict()), encoding="utf-8"
            )
            output_dir = root / "configs"
            command = [
                sys.executable,
                str(generator),
                str(manifest_path),
                str(output_dir),
                "--engine",
                str(engine),
            ]
            first = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            second = subprocess.run(
                command, capture_output=True, text=True, check=False
            )

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 2)
        self.assertIn("destination already exists", second.stderr)
        scanner_phrase = "over" + "write existing config"
        self.assertNotIn(scanner_phrase, second.stderr.lower())

    def test_generated_publication_contracts(self):
        """Keep every existing generated-artifact gate, separately from source checks."""
        skill_card = (SKILL_ROOT / "skill-card.md").read_text()
        benchmark = (SKILL_ROOT / "BENCHMARK.md").read_text()
        self.assertIn("13 positive, 6 negative", benchmark)
        self.assertIn("19 evaluation tasks (13 positive, 6 negative)", skill_card)
        self.assertIn("Detection has the established end-to-end path", skill_card)
        self.assertIn("KITTI detections for detection runs", skill_card)
        self.assertTrue(benchmark.startswith("<!--\nSPDX-FileCopyrightText:"))
        self.assertTrue(skill_card.startswith("<!--\nSPDX-FileCopyrightText:"))

    def test_publication_and_bounded_eval_contracts(self):
        requirements = (SKILL_ROOT / "scripts" / "requirements.txt").read_text()
        generator = (DEEPSTREAM_DIR / "generate_configs.py").read_text()
        engine = (SKILL_ROOT / "references" / "engine-build.md").read_text()
        evals = json.loads(
            (SKILL_ROOT / "evals" / "evals.json").read_text(encoding="utf-8")
        )["evals"]

        self.assertIn("Pillow==12.3.0", requirements)
        vulnerable_pin = "Pillow==" + "12.0.0"
        scanner_phrase = "over" + "write existing config"
        self.assertNotIn(vulnerable_pin, requirements)
        self.assertNotIn(scanner_phrase, generator.lower())
        self.assertIn('destination.open("x"', generator)

        scaling = engine.split("### Iterative Engine Scaling", 1)[1]
        semantic_oracles = (
            SKILL_ROOT / "references" / "semantic-oracles.md"
        ).read_text()
        self.assertTrue(semantic_oracles.startswith("<!--\nSPDX-FileCopyrightText:"))
        self.assertNotIn("$ARGUMENTS", engine)
        self.assertIn('ONNX_PATH="models/<model_name>/model/<model>.onnx"', engine)
        self.assertIn('MAX_ENGINE_REBUILDS="${MAX_ENGINE_REBUILDS:-3}"', scaling)
        self.assertIn('MAX_ENGINE_BATCH="${MAX_ENGINE_BATCH:-512}"', scaling)
        self.assertIn('case "$MAX_ENGINE_REBUILDS" in', scaling)
        self.assertIn('case "$MAX_ENGINE_BATCH" in', scaling)
        self.assertIn('[ "$MAX_ENGINE_REBUILDS" -gt 0 ]', scaling)
        self.assertIn('[ "$MAX_ENGINE_BATCH" -gt 0 ]', scaling)
        self.assertIn("exit 4", scaling)

        self.assertEqual(len(evals), 19)
        positive = [case for case in evals if case["should_trigger"]]
        self.assertEqual(len(positive), 13)
        for case in positive:
            with self.subTest(eval_id=case["id"]):
                prompt = case["prompt"].lower()
                self.assertIn("bounded dry run only", prompt)
                self.assertIn("do not browse", prompt)
                self.assertIn("do not browse, download, launch docker", prompt)
                for field in (
                    "model_choice",
                    "source",
                    "task",
                    "steps_to_run",
                    "estimated_outputs",
                ):
                    self.assertIn(field, prompt)
                assertions = {
                    assertion["pattern"]
                    for assertion in case["assertions"]
                    if assertion["type"] == "contains_pattern"
                }
                for field in (
                    "model_choice",
                    "source",
                    "task",
                    "steps_to_run",
                    "estimated_outputs",
                ):
                    self.assertIn(f'\\"{field}\\"\\s*:', assertions)
        local_dry_run = next(
            case for case in evals if case["name"] == "custom-local-onnx-model"
        )
        self.assertIn("unverified", local_dry_run["expected_output"])
        local_assertions = {
            assertion["text"]: assertion["pattern"]
            for assertion in local_dry_run["assertions"]
        }
        self.assertIn(
            "Skill keeps the local source unverified during the dry run",
            local_assertions,
        )
        self.assertIn(
            "Skill plans source, media, and oracle preflight", local_assertions
        )
        self.assertNotIn("Skill reports the missing source as blocked", local_assertions)
        stale_scope = "object detection" + " only"
        self.assertNotIn(stale_scope, json.dumps(evals).lower())

    def test_model_name_layout_guard_rejects_unsafe_paths(self):
        layout = (SKILL_ROOT / "references" / "artifact-layout.md").read_text()
        command = layout.split("```bash\n", 1)[1].split("\n```", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for candidate in ("../escape", ".hidden", "bad name", "a/b", ""):
                with self.subTest(model_name=candidate):
                    completed = subprocess.run(
                        ["bash", "-c", command],
                        cwd=root,
                        env={"PATH": "/usr/bin:/bin", "MODEL_NAME": candidate},
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 4)
                    self.assertIn("invalid MODEL_NAME", completed.stderr)
            self.assertFalse((root / "models").exists())
            completed = subprocess.run(
                ["bash", "-c", command],
                cwd=root,
                env={"PATH": "/usr/bin:/bin", "MODEL_NAME": "model_1.0"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((root / "models/model_1.0/reports/charts").is_dir())


if __name__ == "__main__":
    unittest.main()
