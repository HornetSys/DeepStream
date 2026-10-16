# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only plan, measurement, output, and tensor-pointer capacity contracts."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


DEEPSTREAM = Path(__file__).resolve().parents[1] / "scripts/multitask/deepstream"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DEEPSTREAM / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TENSORS = load_module("capacity_test_tensors", "run_type_adapter_pipeline.py")
# The capacity script imports its sibling by name. Keep the alias local to
# this import rather than changing sys.path or other tests' module bindings.
with patch.dict(sys.modules, {"run_type_adapter_pipeline": TENSORS}):
    CAPACITY = load_module("capacity_test_runner", "run_task_capacity.py")


class TaskCapacityPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="task-capacity-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.engine = self.root / "fake engine.plan"
        self.engine.write_bytes(b"unit-test artifact: never deserialized")
        self.video = self.root / "fake video.mp4"
        self.video.write_bytes(b"unit-test media: never decoded")
        self.primary = self.config("primary")
        self.plan = {
            "task": "detection", "streams": 2, "mux_batch": 2,
            "width": 1280, "height": 720, "video": str(self.video),
            "stages": [{"config": str(self.primary)}],
            "decoder_kind": "native-detection",
        }

    def config(self, name: str, overrides: dict | None = None) -> Path:
        properties = {
            "gie-unique-id": "1", "process-mode": "1", "network-type": "0",
            "batch-size": "2", "model-engine-file": self.engine.name,
            "interval": "0",
        }
        properties.update(overrides or {})
        path = self.root / f"{name}.txt"
        path.write_text("[property]\n" + "".join(
            f"{key}={value}\n" for key, value in properties.items()), encoding="utf-8")
        return path

    def secondary(self, name: str, overrides: dict | None = None) -> Path:
        return self.config(name, {
            "gie-unique-id": "2", "process-mode": "2", "network-type": "1",
            "operate-on-gie-id": "1", "batch-size": "8",
            **(overrides or {}),
        })

    def with_stages(self, *configs: Path) -> dict:
        return {**self.plan, "stages": [{"config": str(path)} for path in configs]}

    def test_valid_primary_records_resolved_engine_config_and_hashes(self) -> None:
        media, records = CAPACITY.validate_plan(self.plan)
        self.assertEqual(media, self.video.resolve())
        self.assertEqual(records, [{
            "config": str(self.primary.resolve()),
            "config_sha256": hashlib.sha256(self.primary.read_bytes()).hexdigest(),
            "engine": str(self.engine.resolve()),
            "engine_sha256": hashlib.sha256(self.engine.read_bytes()).hexdigest(),
            "component_id": 1, "inference_batch": 2, "batch_unit": "frames",
        }])
        absolute = self.config("absolute", {"model-engine-file": str(self.engine)})
        _, records = CAPACITY.validate_plan(self.with_stages(absolute))
        self.assertEqual(records[0]["engine"], str(self.engine))

    def test_valid_cascade_preserves_stage_order_and_roi_batch_units(self) -> None:
        classifier = self.secondary("classifier")
        roi_stage = self.secondary("roi-stage", {"gie-unique-id": "3", "batch-size": "16"})
        _, records = CAPACITY.validate_plan({
            **self.with_stages(self.primary, classifier, roi_stage),
            "task": "classification", "decoder_kind": "native-classification",
        })
        self.assertEqual([record["component_id"] for record in records], [1, 2, 3])
        self.assertEqual([record["inference_batch"] for record in records], [2, 8, 16])
        self.assertEqual([record["batch_unit"] for record in records], ["frames", "ROIs", "ROIs"])

    def test_all_five_tasks_and_boundary_values_accept_valid_plans(self) -> None:
        tensor_config = self.config("tensor-primary", {"network-type": "100"})
        for task in ("detection", "classification", "segmentation", "pose", "depth"):
            with self.subTest(task=task):
                media, _ = CAPACITY.validate_plan({
                    **self.with_stages(tensor_config), "task": task, "decoder_kind": "tensor-hook",
                    "decoder": "caller_decoder.py:decode",
                    "streams": 512, "mux_batch": 1, "width": 16, "height": 4096,
                    "duration_seconds": 300, "warmup_seconds": 30, "target_fps": 0.5,
                })
                self.assertEqual(media, self.video)

    def test_invalid_task_dimensions_counts_timing_and_target(self) -> None:
        cases = (
            ("task", "audio"), ("streams", 0), ("streams", True),
            ("streams", "2"), ("streams", 513), ("mux_batch", 0),
            ("mux_batch", 3), ("width", 15), ("width", 4097),
            ("height", 15), ("height", 4097), ("height", 720.0),
            ("duration_seconds", 59), ("duration_seconds", 301),
            ("warmup_seconds", -1), ("warmup_seconds", 31),
            ("target_fps", 0), ("target_fps", -1),
            ("target_fps", float("nan")), ("target_fps", float("inf")),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                CAPACITY.validate_plan({**self.plan, key: value})

    def test_missing_empty_or_directory_media_is_rejected(self) -> None:
        empty = self.root / "empty.mp4"
        empty.touch()
        for media in (self.root / "absent.mp4", empty, self.root):
            with self.subTest(media=media), self.assertRaises((ValueError, OSError)):
                CAPACITY.validate_plan({**self.plan, "video": str(media)})

    def test_missing_configs_and_missing_or_directory_engines_are_rejected(self) -> None:
        missing_config = self.root / "absent-config.txt"
        for config in (
            missing_config,
            self.config("missing-engine", {"model-engine-file": "absent.plan"}),
            self.config("directory-engine", {"model-engine-file": str(self.root)}),
        ):
            with self.subTest(config=config), self.assertRaises((ValueError, OSError)):
                CAPACITY.validate_plan(self.with_stages(config))

    def test_missing_or_nonlist_stages_are_rejected(self) -> None:
        for stages in (None, [], {}, "primary.txt"):
            with self.subTest(stages=stages), self.assertRaisesRegex(ValueError, "stages"):
                CAPACITY.validate_plan({**self.plan, "stages": stages})

    def test_invalid_primary_id_mode_batch_and_interval_are_rejected(self) -> None:
        for key, value in (
            ("gie-unique-id", "0"), ("process-mode", "2"),
            ("batch-size", "0"), ("batch-size", "513"),
            ("interval", "1"), ("interval", "-1"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                config = self.config("bad-primary", {key: value})
                CAPACITY.validate_plan(self.with_stages(config))

    def test_secondary_requires_unique_id_secondary_mode_and_prior_parent(self) -> None:
        for key, value in (
            ("gie-unique-id", "1"), ("gie-unique-id", "0"),
            ("process-mode", "1"), ("operate-on-gie-id", "0"),
            ("operate-on-gie-id", "2"), ("operate-on-gie-id", "3"),
            ("interval", "1"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                config = self.secondary("bad-secondary", {key: value})
                CAPACITY.validate_plan(self.with_stages(self.primary, config))

    def test_async_and_skipped_secondary_reinference_are_rejected(self) -> None:
        for key in ("classifier-async-mode", "secondary-reinfer-interval"):
            for value in ("1", "-1"):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    ValueError, "synchronous per-frame"
                ):
                    config = self.secondary("async-secondary", {key: value})
                    CAPACITY.validate_plan(self.with_stages(self.primary, config))
        synchronous = self.secondary("sync-secondary", {
            "classifier-async-mode": "0", "secondary-reinfer-interval": "0",
        })
        _, records = CAPACITY.validate_plan({
            **self.with_stages(self.primary, synchronous),
            "task": "classification", "decoder_kind": "native-classification",
        })
        self.assertEqual(len(records), 2)

    def test_final_network_type_must_match_each_decoder_kind(self) -> None:
        kinds = {"native-detection": "0", "native-classification": "1", "tensor-hook": "100"}
        for kind, expected in kinds.items():
            for network_type in ("0", "1", "100"):
                with self.subTest(kind=kind, network_type=network_type):
                    config = self.config("network-type", {"network-type": network_type})
                    plan = {**self.with_stages(config), "decoder_kind": kind,
                            "decoder": "reviewed_caller_decoder.py:decode"}
                    if network_type == expected:
                        _, records = CAPACITY.validate_plan(plan)
                        self.assertEqual(len(records), 1)
                    else:
                        with self.assertRaisesRegex(ValueError, "network type"):
                            CAPACITY.validate_plan(plan)

    def test_caller_relative_config_and_engine_resolve_after_external_approval(self) -> None:
        # The docs require caller-side engine/profile/plugin approval before
        # invoking this runner. validate_plan records artifacts, not approval;
        # these caller-owned fixture bytes are never deserialized.
        provenance = self.root / "caller-provenance.json"
        provenance.write_text(json.dumps({"engine_sha256": hashlib.sha256(
            self.engine.read_bytes()).hexdigest(), "scope": "unit-test-only"}), encoding="utf-8")
        config_dir = self.root / "configs"
        config_dir.mkdir()
        config = self.config("caller-relative", {"model-engine-file": "../" + self.engine.name})
        config = config.rename(config_dir / "nvinfer.txt")
        plan = {
            **self.plan, "video": self.video.name,
            "stages": [{"config": str(config.relative_to(self.root))}],
            "execution_approval": {"approved": True, "approver": "unit-test-caller",
                                   "approved_at": "2026-10-08T00:00:00+00:00",
                                   "scope": "deserialize-supplied-tensorrt-engine",
                                   "trusted_root": str(self.root),
                                   "engine_provenance": str(provenance)},
        }
        with contextlib.chdir(self.root), patch.dict(sys.modules, {"gi": None, "pyds": None}):
            media, records = CAPACITY.validate_plan(plan)
        self.assertEqual(media, self.video)
        self.assertEqual(records[0]["config"], str(config))
        self.assertEqual(records[0]["engine"], str(self.engine))
        self.assertEqual(records[0]["engine_sha256"], hashlib.sha256(self.engine.read_bytes()).hexdigest())

    def test_decoder_kind_and_tensor_hook_presence_are_required(self) -> None:
        for kind in (None, "unsupported", "tensor-hook"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "decoder"):
                CAPACITY.validate_plan({**self.plan, "decoder_kind": kind})

    def test_existing_output_file_or_directory_is_preserved_before_gpu_import(self) -> None:
        for directory in (False, True):
            with self.subTest(directory=directory):
                output = self.root / ("directory-attempt" if directory else "file-attempt")
                if directory:
                    output.mkdir()
                sentinel = output / "result.json" if directory else output
                sentinel.write_text("existing evidence", encoding="utf-8")
                with patch.dict(sys.modules, {"gi": None, "pyds": None}), self.assertRaisesRegex(
                    ValueError, "already exists"
                ):
                    CAPACITY.run(self.plan, output)
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing evidence")
                if directory:
                    self.assertEqual(list(output.iterdir()), [sentinel])

    def test_output_symlinks_do_not_overwrite_existing_or_missing_targets(self) -> None:
        for existing in (False, True):
            with self.subTest(existing=existing):
                target = self.root / f"target-{existing}"
                if existing:
                    target.mkdir()
                    (target / "sentinel").write_text("preserve", encoding="utf-8")
                output = self.root / f"link-{existing}"
                output.symlink_to(target, target_is_directory=True)
                with patch.dict(sys.modules, {"gi": None, "pyds": None}), self.assertRaises(
                    (ValueError, FileExistsError)
                ):
                    CAPACITY.run(self.plan, output)
                self.assertTrue(output.is_symlink())
                if existing:
                    self.assertEqual((target / "sentinel").read_text(encoding="utf-8"), "preserve")
                    self.assertFalse((target / "result.json").exists())
                else:
                    self.assertFalse(target.exists())

    def test_competing_output_directory_creation_is_exclusive(self) -> None:
        output = self.root / "racing-attempt"
        original_mkdir = Path.mkdir

        def competing_mkdir(path: Path, *args, **kwargs):
            original_mkdir(path)
            (path / "result.json").write_text("competing evidence", encoding="utf-8")
            return original_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", autospec=True, side_effect=competing_mkdir), \
             patch.dict(sys.modules, {"gi": None, "pyds": None}), \
             self.assertRaises(FileExistsError):
            CAPACITY.run(self.plan, output)
        self.assertEqual((output / "result.json").read_text(encoding="utf-8"), "competing evidence")
        self.assertFalse((output / "plan.json").exists())

    def test_main_reports_existing_output_as_blocked_without_touching_evidence(self) -> None:
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        output = self.root / "existing-attempt"
        output.mkdir()
        stderr = io.StringIO()
        with patch.object(sys, "argv", ["run_task_capacity.py", str(plan_path),
                                      "--output-dir", str(output)]), \
             patch.dict(sys.modules, {"gi": None, "pyds": None}), \
             contextlib.redirect_stderr(stderr):
            self.assertEqual(CAPACITY.main(), 4)
        result = json.loads(stderr.getvalue())
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("already exists", result["reason"])
        self.assertEqual(list(output.iterdir()), [])

    def test_missing_decoder_symbol_is_validation_error(self) -> None:
        decoder = self.root / "decoder.py"
        decoder.write_text("other = 1\n", encoding="utf-8")
        with self.assertRaisesRegex(TypeError, "decoder is not callable"):
            CAPACITY.load_decoder(str(decoder) + ":decode")


class TaskCapacitySummaryTests(unittest.TestCase):
    @staticmethod
    def windows(*rows: list[float]) -> list[dict]:
        return [{"duration_seconds": 2, "fps": row} for row in rows]

    def test_two_source_sum_and_worst_final_five_window_control_status(self) -> None:
        windows = self.windows([0, 0], [1, 1], [40, 20], [41, 25], [42, 30], [43, 35], [44, 40])
        result = CAPACITY.summarize(windows, 2, 30)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["window_count"], 7)
        self.assertEqual(result["aggregate_average_fps"], 72)
        self.assertEqual(result["per_stream"], [
            {"source_id": 0, "windows_fps": [40, 41, 42, 43, 44], "minimum_fps": 40, "average_fps": 42},
            {"source_id": 1, "windows_fps": [20, 25, 30, 35, 40], "minimum_fps": 20, "average_fps": 30},
        ])

    def test_final_five_meeting_target_pass_despite_early_slow_windows(self) -> None:
        windows = self.windows([0, 0], *[[30, 40]] * 5)
        result = CAPACITY.summarize(windows, 2, 30)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["aggregate_average_fps"], 70)
        self.assertEqual(result["streams"], 2)
        self.assertEqual(result["target_fps_per_stream"], 30)
        self.assertEqual([item["minimum_fps"] for item in result["per_stream"]], [30, 40])

    def test_missing_measurement_windows_fail_even_when_available_fps_is_high(self) -> None:
        for count in (0, 1, 4):
            with self.subTest(count=count):
                result = CAPACITY.summarize(self.windows(*[[100, 100]] * count), 2, 30)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(result["window_count"], count)
                self.assertEqual(result["aggregate_average_fps"], 200 if count else 0)
                for source in result["per_stream"]:
                    self.assertEqual(len(source["windows_fps"]), count)
                    self.assertEqual(source["minimum_fps"], 100 if count else None)
                    self.assertEqual(source["average_fps"], 100 if count else None)


def meta_list(values):
    """Represent pyds linked lists without importing the actual bindings."""
    node = None
    for value in reversed(list(values)):
        node = SimpleNamespace(data=value, next=node)
    return node


def fake_frame(source_id, frame_id, *, objects=(), users=()):
    return SimpleNamespace(source_id=source_id, frame_num=frame_id,
                           obj_meta_list=meta_list(objects), frame_user_meta_list=meta_list(users))


def fake_detection(*, component=1, confidence=0.9):
    return SimpleNamespace(unique_component_id=component, class_id=0, confidence=confidence,
                           rect_params=SimpleNamespace(left=0, top=0, width=20, height=10))


def fake_classification(*, component=2, score=0.9, label="fixture-label"):
    classifier = SimpleNamespace(unique_component_id=component, label_info_list=meta_list([
        SimpleNamespace(result_label=label, result_prob=score),
    ]))
    return SimpleNamespace(classifier_meta_list=meta_list([classifier]))


class FakeCapacityRuntime:
    """Run the real probe/measurement loop with deterministic CPU metadata.

    Frames and pad-added callbacks are delivered only by GLib iteration. The
    fake bus advances time by two seconds rather than sleeping or using GPUs.
    """

    def __init__(self, frame_factory, *, source_ids=(0, 1), batch_missing=False, eos=False):
        self.now = 0.0
        self.tick = 0
        self.iterations = 0
        self.pending = False
        self.callbacks = []
        self.states = []
        self.elements = {}
        self.probe = None
        self.batch = None
        self.buffer = object()
        self.source_ids = source_ids
        self.frame_factory = frame_factory
        self.batch_missing = batch_missing
        self.eos = eos
        gi = ModuleType("gi")
        gi.require_version = Mock()
        repository = ModuleType("gi.repository")
        gi.repository = repository
        pyds = ModuleType("pyds")
        pyds.gst_buffer_get_nvds_batch_meta = lambda _address: self.batch
        for name in ("NvDsFrameMeta", "NvDsObjectMeta", "NvDsUserMeta", "NvDsInferTensorMeta",
                     "NvDsClassifierMeta", "NvDsLabelInfo"):
            setattr(pyds, name, SimpleNamespace(cast=lambda value: value))
        pyds.NvDsMetaType = SimpleNamespace(NVDSINFER_TENSOR_OUTPUT_META=77)
        self.context = SimpleNamespace(pending=lambda: self.pending, iteration=self.iterate)
        repository.GLib = SimpleNamespace(MainContext=SimpleNamespace(default=lambda: self.context))
        self.sink_pad = SimpleNamespace(add_probe=Mock(side_effect=self.add_probe))
        self.pipeline = SimpleNamespace(add=Mock(), get_bus=lambda: SimpleNamespace(
            timed_pop_filtered=self.bus_pop), set_state=self.set_state)
        repository.Gst = SimpleNamespace(
            init=Mock(), Pipeline=SimpleNamespace(new=lambda _name: self.pipeline),
            ElementFactory=SimpleNamespace(make=self.make_element),
            State=SimpleNamespace(PLAYING="PLAYING", NULL="NULL"),
            StateChangeReturn=SimpleNamespace(FAILURE="FAILURE"),
            PadProbeType=SimpleNamespace(BUFFER=1), PadProbeReturn=SimpleNamespace(OK=0),
            PadLinkReturn=SimpleNamespace(OK=0), MSECOND=1,
            MessageType=SimpleNamespace(ERROR=1, EOS=2),
        )
        self.modules = {"gi": gi, "gi.repository": repository, "pyds": pyds}

    def make_element(self, factory, name):
        properties = {}
        element = SimpleNamespace(
            factory=factory, properties=properties,
            set_property=lambda key, value: properties.__setitem__(key, value),
            link=lambda _other: True,
            request_pad_simple=lambda _name: object(),
            get_static_pad=lambda _name: self.sink_pad,
        )
        element.connect = lambda _signal, callback, source_id: self.callbacks.append(
            (element, callback, source_id))
        self.elements[name] = element
        return element

    def add_probe(self, _kind, callback):
        self.probe = callback

    def set_state(self, state):
        self.states.append(state)
        self.pending = state == "PLAYING"
        return "SUCCESS"

    def iterate(self, may_block):
        assert may_block is False
        self.pending = False
        self.iterations += 1
        if self.tick == 0:
            caps = SimpleNamespace(to_string=lambda: "video/x-raw(memory:NVMM)",
                                   get_structure=lambda _index: SimpleNamespace(
                                       get_name=lambda: "video/x-raw"))
            pad = SimpleNamespace(get_current_caps=lambda: caps, link=lambda _target: 0)
            for element, callback, source_id in self.callbacks:
                callback(element, pad, source_id)
        frames = []
        for source_id in self.source_ids:
            for index in range(80):
                frame = self.frame_factory(source_id, self.tick, self.tick * 80 + index)
                if frame is not None:
                    frames.append(frame)
        self.batch = None if self.batch_missing else SimpleNamespace(frame_meta_list=meta_list(frames))
        self.probe(self.sink_pad, SimpleNamespace(get_buffer=lambda: self.buffer))
        self.tick += 1
        return True

    def bus_pop(self, _timeout, _kinds):
        self.now += 2.0
        self.pending = True
        return SimpleNamespace(type=2) if self.eos else None


class TaskCapacityRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = TaskCapacityPlanTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.attempt = 0

    def run_fake(self, frame_factory, *, plan=None, **runtime_options):
        runtime = FakeCapacityRuntime(frame_factory, **runtime_options)
        self.attempt += 1
        output = self.fixture.root / f"runtime-attempt-{self.attempt}"
        plan = {**(plan or self.fixture.plan), "warmup_seconds": 0}
        with patch.dict(sys.modules, runtime.modules), \
             patch.object(CAPACITY.time, "monotonic", side_effect=lambda: runtime.now), \
             contextlib.redirect_stdout(io.StringIO()):
            status = CAPACITY.run(plan, output)
        return status, json.loads((output / "result.json").read_text(encoding="utf-8")), runtime

    def classification_plan(self):
        secondary = self.fixture.secondary("runtime-classifier")
        return {**self.fixture.with_stages(self.fixture.primary, secondary),
                "task": "classification", "decoder_kind": "native-classification"}

    def tensor_plan(self):
        config = self.fixture.config("runtime-tensor", {"network-type": "100"})
        decoder = self.fixture.root / "reviewed_fixture_decoder.py"
        decoder.write_text("def decode(values, context):\n    return {'label': 'fixture'}\n",
                           encoding="utf-8")
        return {**self.fixture.with_stages(config), "task": "segmentation",
                "decoder_kind": "tensor-hook", "decoder": str(decoder) + ":decode"}

    def test_two_actual_producers_pass_and_require_glib_file_loop_callbacks(self) -> None:
        status, result, runtime = self.run_fake(lambda source, _tick, frame: fake_frame(
            source, frame, objects=[fake_detection()]))
        self.assertEqual(status, 0)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["frames_per_source"], [2400, 2400])
        self.assertEqual(result["task_records_per_source"], [2400, 2400])
        self.assertEqual(result["aggregate_average_fps"], 80)
        self.assertEqual(result["window_count"], 30)
        self.assertEqual(runtime.iterations, 30)
        self.assertEqual(runtime.states, ["PLAYING", "NULL"])
        for source in ("source_0", "source_1"):
            self.assertTrue(runtime.elements[source].properties["file-loop"])

    def test_all_source_zero_metadata_cannot_pass_for_two_declared_streams(self) -> None:
        status, result, _ = self.run_fake(lambda source, _tick, frame: fake_frame(
            source, frame, objects=[fake_detection()]), source_ids=(0, 0))
        self.assertEqual(status, 1)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["frames_per_source"], [4800, 0])
        self.assertEqual(result["task_records_per_source"], [4800, 0])
        self.assertEqual(result["per_stream"][1]["minimum_fps"], 0)
        self.assertEqual(result["aggregate_average_fps"], 80)

    def test_one_missing_frame_producer_or_both_missing_fail(self) -> None:
        for sources in ((0,), ()):
            with self.subTest(sources=sources):
                status, result, _ = self.run_fake(lambda source, _tick, frame: fake_frame(
                    source, frame, objects=[fake_detection()]), source_ids=sources)
                self.assertEqual(status, 1)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(result["frames_per_source"][1], 0)

    def test_frames_without_any_task_producer_fail_despite_high_fps(self) -> None:
        status, result, _ = self.run_fake(lambda source, _tick, frame: fake_frame(source, frame))
        self.assertEqual(status, 1)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["aggregate_average_fps"], 80)
        self.assertEqual(result["task_records_per_source"], [0, 0])

    def test_wrong_final_component_never_produces_task_evidence(self) -> None:
        status, result, _ = self.run_fake(lambda source, _tick, frame: fake_frame(
            source, frame, objects=[fake_detection(component=99)]))
        self.assertEqual(status, 1)
        self.assertEqual(result["task_records_per_source"], [0, 0])

    def test_task_producer_disappearing_before_final_windows_cannot_pass(self) -> None:
        for classification in (False, True):
            with self.subTest(classification=classification):
                producer = fake_classification if classification else fake_detection
                status, result, _ = self.run_fake(lambda source, tick, frame, producer=producer, classification=classification: fake_frame(
                    source, frame, objects=[producer(component=1 if not classification else 2)]
                    if tick == 0 else [producer(component=99)]),
                    plan=self.classification_plan() if classification else self.fixture.plan)
                # Foreign-stage output continues; this does not demand a
                # nonempty detection on every otherwise valid empty frame.
                self.assertGreater(min(result["task_records_per_source"]), 0)
                self.assertEqual(result["aggregate_average_fps"], 80)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(status, 1)

    def test_native_nonfinite_confidences_scores_and_unprintable_labels_fail(self) -> None:
        cases = (
            (False, fake_detection(confidence=float("nan")), "non-finite confidence"),
            (False, fake_detection(confidence=float("inf")), "non-finite confidence"),
            (True, fake_classification(score=float("nan")), "non-finite score"),
            (True, fake_classification(score=float("inf")), "non-finite score"),
            (True, fake_classification(label=""), "invalid label"),
            (True, fake_classification(label="line\nbreak"), "invalid label"),
            (True, fake_classification(label="\x7f"), "invalid label"),
            (True, fake_classification(label="x" * 257), "invalid label"),
        )
        for classification, obj, error in cases:
            with self.subTest(error=error, obj=obj):
                status, result, runtime = self.run_fake(lambda source, _tick, frame, obj=obj: fake_frame(
                    source, frame, objects=[obj]), plan=self.classification_plan() if classification else None)
                self.assertEqual(status, 1)
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(any(error in entry for entry in result["errors"]), result["errors"])
                self.assertEqual(runtime.states[-1], "NULL")

    def test_missing_frame_metadata_and_early_eos_fail_closed(self) -> None:
        for options, expected in (({"batch_missing": True}, "no frame metadata"),
                                  ({"eos": True}, "unexpected EOS")):
            with self.subTest(options=options):
                status, result, _ = self.run_fake(lambda source, _tick, frame: fake_frame(
                    source, frame, objects=[fake_detection()]), **options)
                self.assertEqual(status, 1)
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(any(expected in error for error in result["errors"]))

    def test_full_frame_tensor_route_requires_tensor_metadata(self) -> None:
        status, result, _ = self.run_fake(lambda source, _tick, frame: fake_frame(source, frame),
                                        plan=self.tensor_plan())
        self.assertEqual(status, 1)
        self.assertTrue(any("tensor metadata is missing" in error for error in result["errors"]))

    def test_tensor_metadata_without_output_layers_cannot_pass_with_constant_hook(self) -> None:
        tensor = SimpleNamespace(unique_id=1, num_output_layers=0)
        user = SimpleNamespace(base_meta=SimpleNamespace(meta_type=77), user_meta_data=tensor)
        status, result, _ = self.run_fake(lambda source, _tick, frame: fake_frame(
            source, frame, users=[user]), plan=self.tensor_plan())
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(status, 1)


class Int64TensorPointerTests(unittest.TestCase):
    def test_dtype_four_reads_int64_host_pointer_without_truncation_and_copies(self) -> None:
        values = [0, -1, 2**40 + 123, -(2**40) - 7]
        buffer = (ctypes.c_int64 * len(values))(*values)
        address = ctypes.addressof(buffer)
        layer = SimpleNamespace(
            inferDims=SimpleNamespace(d=[2, 2], numDims=2, numElements=4),
            dataType=4, layerName="indices", buffer=address,
        )
        for host_pointer in (True, False):
            with self.subTest(host_pointer=host_pointer):
                buffer[:] = values
                meta = SimpleNamespace(out_buf_ptrs_host=[address]) if host_pointer else SimpleNamespace()
                pyds = SimpleNamespace(get_nvds_LayerInfo=Mock(return_value=layer),
                                       get_ptr=Mock(side_effect=lambda pointer: pointer))
                name, array = TENSORS._layer_array(pyds, meta, 0)
                self.assertEqual(name, "indices")
                self.assertEqual(array.dtype, np.dtype(np.int64))
                self.assertEqual(array.shape, (2, 2))
                np.testing.assert_array_equal(array, np.array(values, dtype=np.int64).reshape(2, 2))
                pyds.get_nvds_LayerInfo.assert_called_once_with(meta, 0)
                pyds.get_ptr.assert_called_once_with(address)
                buffer[2] = 17
                self.assertEqual(int(array[1, 0]), 2**40 + 123)


if __name__ == "__main__":
    unittest.main()
