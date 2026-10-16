# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
ROUTER = SKILL_ROOT / "scripts" / "model" / "route_task.py"
INNER_ROUTER = (
    SKILL_ROOT / "scripts" / "multitask" / "model" / "route_model.py"
)
ADAPTER_GENERATOR = (
    SKILL_ROOT
    / "scripts"
    / "multitask"
    / "deepstream"
    / "generate_adapter.py"
)


class TestUnifiedTaskRouting(unittest.TestCase):
    @staticmethod
    def runnable_metadata(**extra):
        metadata = {
            "input_shape": [1, 3, 224, 224],
            "output_names": ["output"],
        }
        metadata.update(extra)
        return metadata

    def route(
        self,
        task: str,
        metadata: dict | None = None,
        *,
        create_source: bool = True,
    ):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model.engine"
            if create_source:
                model.write_bytes(b"test-engine-placeholder")
            command = [sys.executable, str(ROUTER), str(model), "--task", task]
            if metadata is not None:
                metadata_path = root / "metadata.json"
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                command.extend(["--metadata", str(metadata_path)])
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            return result, json.loads(result.stdout)

    def test_detection_stays_on_native_pipeline(self):
        result, route = self.route("object-detection")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(route["public_skill"], "deepstream-import-vision-model")
        self.assertEqual(route["category"], "object-detection")
        self.assertEqual(route["execution_family"], "native-object-detection")
        self.assertEqual(route["capability_status"], "READY")
        self.assertEqual(route["delegate_skill"], "")

    def test_native_classification_uses_embedded_runtime(self):
        result, route = self.route("image-classification", self.runnable_metadata())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(route["execution_family"], "multitask-image-classification")
        self.assertEqual(route["capability_status"], "READY")

    def test_adapter_categories_use_embedded_runtime_and_stop(self):
        cases = {
            "semantic-segmentation": "multitask-semantic-segmentation",
            "2d-pose": "multitask-2d-pose",
            "monocular-depth": "multitask-monocular-depth",
        }
        for task, family in cases.items():
            with self.subTest(task=task):
                result, route = self.route(task, self.runnable_metadata())
                self.assertEqual(result.returncode, 4, result.stderr)
                self.assertEqual(route["category"], task)
                self.assertEqual(route["execution_family"], family)
                self.assertEqual(route["capability_status"], "ADAPTER_REQUIRED")
                self.assertEqual(route["delegate_skill"], "")

    def test_secondary_classifier_preserves_task_variant(self):
        result, route = self.route(
            "secondary-classification", self.runnable_metadata()
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(route["category"], "image-classification")
        self.assertEqual(route["task"], "classification")
        self.assertEqual(route["task_variant"], "secondary-classification")

    def test_auto_uses_the_metadata_task(self):
        result, route = self.route(
            "auto",
            self.runnable_metadata(task="semantic-segmentation"),
        )
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertEqual(route["category"], "semantic-segmentation")

    def test_auto_without_declared_task_is_blocked_not_unsupported(self):
        result, route = self.route("auto", self.runnable_metadata())
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertEqual(route["category"], "undetermined")
        self.assertEqual(route["status"], "needs-task-identification")

    def test_deferred_categories_fail_closed(self):
        for task in (
            "audio-classification",
            "instance-segmentation",
            "sam2",
            "stereo-depth",
            "ocr",
        ):
            with self.subTest(task=task):
                result, route = self.route(task)
                self.assertEqual(result.returncode, 3)
                self.assertEqual(route["status"], "unsupported")
                self.assertEqual(route["capability_status"], "BLOCKED")

    def test_supported_category_list_is_exact(self):
        _, route = self.route("object-detection")
        self.assertEqual(
            route["supported_categories"],
            [
                "object-detection",
                "image-classification",
                "semantic-segmentation",
                "2d-pose",
                "monocular-depth",
            ],
        )

    def test_missing_source_is_blocked_and_returns_nonzero(self):
        result, route = self.route(
            "semantic-segmentation",
            self.runnable_metadata(),
            create_source=False,
        )
        self.assertEqual(result.returncode, 4)
        self.assertEqual(route["status"], "blocked-missing-source")
        self.assertEqual(route["capability_status"], "BLOCKED")

    def test_relative_source_is_persisted_absolute_across_adapter_manifest_move(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "inputs" / "model.engine"
            model.parent.mkdir()
            model.write_bytes(b"fixture")
            metadata = root / "task-metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "input_shape": [1, 3, 224, 224],
                        "output_names": ["mask"],
                        "mean": [0.1, 0.2, 0.3],
                        "std": [0.2, 0.3, 0.4],
                        "preprocess_units": "normalized",
                        "task_contract": {
                            "mask_layout": "NCHW",
                            "class_count": 2,
                            "resize_alignment": "nearest",
                            "ignore_label": -1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            route_path = root / "routes" / "route.json"
            routed = subprocess.run(
                [
                    sys.executable,
                    str(ROUTER),
                    "inputs/model.engine",
                    "--task",
                    "semantic-segmentation",
                    "--metadata",
                    str(metadata),
                    "--output",
                    str(route_path),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(routed.returncode, 4, routed.stderr)
            route = json.loads(route_path.read_text(encoding="utf-8"))
            self.assertEqual(route["source"], str(model.resolve()))

            adapter_dir = root / "generated" / "adapter"
            generated = subprocess.run(
                [
                    sys.executable,
                    str(ADAPTER_GENERATOR),
                    str(route_path),
                    str(adapter_dir),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            merged = json.loads(
                (adapter_dir / "manifest-with-adapter.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(merged["source"], str(model.resolve()))

    def test_native_detection_route_has_no_self_delegation_warning(self):
        result, route = self.route("object-detection")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(
            any(
                "use deepstream-import-vision-model" in warning
                for warning in route["warnings"]
            )
        )

    def test_inner_router_uses_fail_closed_exit_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model.engine"
            model.write_bytes(b"fixture")
            audio = subprocess.run(
                [
                    sys.executable,
                    str(INNER_ROUTER),
                    str(model),
                    "--task",
                    "audio-classification",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(audio.returncode, 3, audio.stderr)
            self.assertEqual(json.loads(audio.stdout)["status"], "unsupported")

            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(self.runnable_metadata()),
                encoding="utf-8",
            )
            segmentation = subprocess.run(
                [
                    sys.executable,
                    str(INNER_ROUTER),
                    str(model),
                    "--task",
                    "semantic-segmentation",
                    "--metadata",
                    str(metadata_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(segmentation.returncode, 4, segmentation.stderr)
            self.assertEqual(
                json.loads(segmentation.stdout)["capability_status"],
                "ADAPTER_REQUIRED",
            )

            missing = subprocess.run(
                [
                    sys.executable,
                    str(INNER_ROUTER),
                    str(root / "missing.engine"),
                    "--task",
                    "classification",
                    "--metadata",
                    str(metadata_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(missing.returncode, 4, missing.stderr)
            self.assertEqual(
                json.loads(missing.stdout)["status"],
                "blocked-missing-source",
            )

    def test_inner_router_help_lists_only_accepted_tasks(self):
        completed = subprocess.run(
            [sys.executable, str(INNER_ROUTER), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        help_text = re.sub(r"-\s+", "-", completed.stdout)
        self.assertIn("object-detection", help_text)
        self.assertIn("secondary-classification", help_text)
        self.assertIn("monocular-depth", help_text)
        self.assertNotIn("embedding", help_text)
        self.assertNotIn("generic", help_text)

    def test_embedded_documentation_uses_installed_script_paths(self):
        documents = (
            SKILL_ROOT / "references" / "multitask" / "examples.md",
            SKILL_ROOT / "references" / "multitask" / "validation.md",
            SKILL_ROOT / "scripts" / "multitask" / "visual" / "visual_io_probe.py",
        )
        obsolete = (
            "scripts/model/route_model.py",
            "scripts/model/emit_commands.py",
            "scripts/validation/validate_artifact.py",
            "scripts/validation/compare_outputs.py",
            "scripts/visual/render_trt_visual.py",
            "scripts/visual/visual_io_probe.py",
        )
        combined = "\n".join(path.read_text(encoding="utf-8") for path in documents)
        for token in obsolete:
            with self.subTest(token=token):
                self.assertNotIn(token, combined)

        expected = (
            "scripts/multitask/model/route_model.py",
            "scripts/multitask/model/emit_commands.py",
            "scripts/multitask/validation/validate_artifact.py",
            "scripts/multitask/validation/compare_outputs.py",
            "scripts/multitask/visual/render_trt_visual.py",
            "scripts/multitask/visual/visual_io_probe.py",
        )
        for relative in expected:
            with self.subTest(relative=relative):
                self.assertTrue((SKILL_ROOT / relative).is_file())

    def test_generated_adapter_uses_approved_capture_fallback(self):
        generator = (
            SKILL_ROOT / "scripts" / "multitask" / "deepstream" / "generate_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '["nvv4l2h264enc", "theoraenc+oggmux", "skipped"]',
            generator,
        )
        self.assertNotIn(
            '"encoder": ["platform-hardware-encoder", "x264enc"',
            generator,
        )
        self.assertNotIn("x264enc/theoraenc fallback", generator)

    def test_classification_reference_runner_has_real_deepstream_contract(self):
        runner = (
            SKILL_ROOT
            / "scripts"
            / "multitask"
            / "deepstream"
            / "run_type_adapter_pipeline.py"
        )
        completed = subprocess.run(
            [sys.executable, str(runner), "--describe"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        contract = json.loads(completed.stdout)
        self.assertEqual(contract["producer"], "deepstream-type-adapter-runner")
        self.assertEqual(contract["task"], "classification")
        self.assertEqual(contract["bounded_timeout_seconds"], 300)
        pipeline = " ".join(contract["pipeline"])
        for component in (
            "nvstreammux",
            "nvinfer",
            "NvDsInferTensorMeta",
            "NvDsDisplayMeta",
            "nvdsosd",
        ):
            with self.subTest(component=component):
                self.assertIn(component, pipeline)

if __name__ == "__main__":
    unittest.main()
