# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for Litmus findings in NVBug 6800443."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = REPO_ROOT / "skills" / "deepstream-generate-config"
EVALS = SKILL_ROOT / "evals" / "evals.json"
EVAL_CONFIG = SKILL_ROOT / "evals" / "config.yml"
SKILL_MD = SKILL_ROOT / "SKILL.md"
SKILL_CARD = SKILL_ROOT / "skill-card.md"
INTERACTIVE_QA = SKILL_ROOT / "references" / "interactive_qa.md"


RENDER = SKILL_ROOT / "scripts" / "render.py"
RENDER_INFER = SKILL_ROOT / "scripts" / "render_infer.py"
RENDER_TRACKER = SKILL_ROOT / "scripts" / "render_tracker.py"
VALIDATE = SKILL_ROOT / "scripts" / "validate.py"


def run_script(script, *args):
    return subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        cwd=SKILL_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )



def load_evals():
    return json.loads(EVALS.read_text(encoding="utf-8"))["cases"]


def assertion_text(case, text_prefix):
    for assertion in case["expected_behavior"]:
        if assertion.startswith(text_prefix):
            return assertion
    raise AssertionError(f"assertion starting with {text_prefix!r} not found")


class TestEvalContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_evals()

    def test_schema_and_tier3_assertions(self):
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        for case in self.cases:
            for field in (
                "id",
                "name",
                "question",
                "expected_skill",
                "should_trigger",
                "ground_truth",
                "files",
                "expected_behavior",
            ):
                self.assertIn(field, case)
            self.assertIsInstance(case["should_trigger"], bool)
            self.assertTrue(case["expected_behavior"])
            for assertion in case["expected_behavior"]:
                self.assertIsInstance(assertion, str)
                self.assertTrue(assertion.strip())

    def test_litmus_and_skillevaluator_views_are_in_sync(self):
        payload = json.loads(EVALS.read_text(encoding="utf-8"))
        agent_cases = payload["evals"]
        litmus_cases = payload["cases"]
        self.assertEqual(len(agent_cases), len(litmus_cases))
        for agent_case, litmus_case in zip(agent_cases, litmus_cases):
            for field in (
                "id",
                "name",
                "expected_skill",
                "expected_script",
                "should_trigger",
                "files",
            ):
                self.assertEqual(agent_case.get(field), litmus_case.get(field))
            self.assertEqual(agent_case["prompt"], litmus_case["question"])
            self.assertEqual(
                agent_case["expected_output"],
                litmus_case["ground_truth"],
            )
            self.assertEqual(
                agent_case["assertions"],
                litmus_case["expected_behavior"],
            )


    def test_activation_and_non_activation_coverage(self):
        positives = [case for case in self.cases if case["should_trigger"]]
        negatives = [case for case in self.cases if not case["should_trigger"]]
        self.assertEqual(len(positives), 4)
        self.assertEqual(len(negatives), 3)
        for case in self.cases:
            expected = "deepstream-generate-config" if case["should_trigger"] else None
            self.assertEqual(case["expected_skill"], expected)
            if case["id"].startswith("negative-"):
                self.assertFalse(case["should_trigger"])
                self.assertIsNone(case["expected_skill"])

    def test_version_is_under_metadata_and_scripts_are_discoverable(self):
        text = SKILL_MD.read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(text.split("---", 2)[1])
        self.assertNotIn("version", frontmatter)
        self.assertEqual(frontmatter["metadata"]["version"], "1.0.10")
        self.assertIn(
            f'{frontmatter["metadata"]["version"]} (source: frontmatter)',
            SKILL_CARD.read_text(encoding="utf-8"),
        )
        self.assertIn("scripts/render_infer.py", text)
        self.assertIn("scripts/render_tracker.py", text)

    def test_independent_audits_route_to_auditor(self):
        text = SKILL_MD.read_text(encoding="utf-8")
        self.assertIn("use deepstream-audit-config", text)
        self.assertIn("only the", text)
        self.assertIn("render-time linter", text)

    def test_canonical_tier3_source_is_declared(self):
        config = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(config["harbor"]["task_source"], "evals_json")
        self.assertEqual(config["harbor"]["n_attempts"], 3)

    def test_validation_retry_is_bounded(self):
        text = SKILL_MD.read_text(encoding="utf-8")
        qa_text = INTERACTIVE_QA.read_text(encoding="utf-8")
        self.assertIn("at most 1 targeted correction", text)
        self.assertIn("abort further corrections", text)
        self.assertIn("do not loop", text)
        self.assertIn("at most 1 writable-path retry", text)
        self.assertIn("retry fails, abort generation", text)
        self.assertIn("at most 1 profile clarification", text)
        self.assertIn("at most 1 profile clarification", qa_text)
        self.assertIn("abort without rendering", qa_text)

    def test_invalid_profile_rubric_rejects_successful_render(self):
        case = next(
            case
            for case in self.cases
            if case["id"] == "invalid-profile-rejected"
        )
        rubric = assertion_text(case, "Skill does not claim")
        self.assertIn("generated/rendered", rubric)
        self.assertIn("turbo-max", rubric)

    def test_render_rubric_requires_explicit_command_arguments(self):
        case = next(
            case for case in self.cases if case["id"] == "rtsp-low-latency-config"
        )
        rubric = assertion_text(case, "Skill invokes render.py").lower()
        self.assertIn("explicit spec", rubric)
        self.assertIn("output path", rubric)

    def test_latency_rubric_requires_materialized_setting(self):
        case = next(
            case for case in self.cases if case["id"] == "rtsp-low-latency-config"
        )
        rubric = assertion_text(case, "Skill materializes the latency").lower()
        self.assertIn("spec or concrete streammux settings", rubric)

    def test_rtsp_credential_rubric_distinguishes_placeholders(self):
        case = next(
            case for case in self.cases if case["id"] == "rtsp-low-latency-config"
        )
        rubric = assertion_text(case, "Response does not expose").lower()
        self.assertIn("literal rtsp password", rubric)
        self.assertIn("placeholders are allowed", rubric)


class TestReadinessFixes(unittest.TestCase):
    def test_metadata_docs_and_dependency_contract(self):
        text = SKILL_MD.read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(text.split("---", 2)[1])
        self.assertEqual(frontmatter["license"], "CC-BY-4.0 AND Apache-2.0")
        self.assertNotIn("data_classification", frontmatter)
        self.assertIsInstance(frontmatter["metadata"]["tags"], str)
        self.assertIsInstance(frontmatter["metadata"]["languages"], str)
        self.assertNotIn("compatibility", frontmatter)
        self.assertIn("DeepStream SDK 9.2.x", text)
        self.assertIn("## Purpose", text)
        self.assertIn("## When to Use This Skill", text)
        self.assertIn("## Workflow", text)
        self.assertIn("## Inputs", text)
        self.assertIn("## Output Format", text)
        self.assertIn("## Limitations", text)
        self.assertIn("## Troubleshooting", text)
        self.assertIn("run_script", text)
        self.assertIn("references/reference.md", text)
        self.assertIn("--print-resolved-spec", text)
        requirements = (SKILL_ROOT / "scripts" / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("PyYAML==6.0.3", requirements)

    def test_validate_help(self):
        result = run_script(VALIDATE, "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
        self.assertIn("config_file", result.stdout)

    def test_enabled_inference_requires_config_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "missing-infer-config.txt"
            config_path.write_text(
                """[streammux]
batch-size=1

[primary-gie]
enable=1
batch-size=1

[sink0]
enable=1
type=1
""",
                encoding="utf-8",
            )
            result = run_script(VALIDATE, config_path)
            self.assertEqual(result.returncode, 1)
            self.assertIn("primary-gie", result.stderr)
            self.assertIn("config-file", result.stderr)

            disabled_path = Path(temp_dir) / "disabled-infer.txt"
            disabled_path.write_text(
                """[streammux]
batch-size=1

[primary-gie]
enable=0

[sink0]
enable=1
type=1
""",
                encoding="utf-8",
            )
            disabled = run_script(VALIDATE, disabled_path)
            self.assertEqual(disabled.returncode, 0, disabled.stderr)

    def test_yaml_parse_errors_and_null_config_file_are_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            malformed_path = temp / "malformed.yml"
            malformed_path.write_text("streammux: [\n", encoding="utf-8")
            malformed = run_script(VALIDATE, malformed_path)
            self.assertEqual(malformed.returncode, 1)
            self.assertIn("YAML parse error", malformed.stderr)
            self.assertNotIn("Traceback", malformed.stderr)

            null_path = temp / "null-config-file.yml"
            null_path.write_text(
                """streammux:
  batch-size: 1
primary-gie:
  enable: 1
  batch-size: 1
  config-file:
sink0:
  enable: 1
  type: 1
""",
                encoding="utf-8",
            )
            null_config = run_script(VALIDATE, null_path)
            self.assertEqual(null_config.returncode, 1)
            self.assertIn("config-file", null_config.stderr)
            self.assertNotIn("Traceback", null_config.stderr)

    def test_disabled_custom_gie_renders_and_validates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            spec_path = temp / "disabled-custom.yml"
            output_path = temp / "disabled-custom.txt"
            spec_path.write_text(
                """sources:
  kind: multi_uri
  uris:
    - file:///tmp/input.mp4
  num_sources: 1
primary_gie:
  enable: 0
  preset: custom
sinks:
  - type: fakesink
""",
                encoding="utf-8",
            )
            rendered = run_script(
                RENDER,
                "--spec",
                spec_path,
                "--out",
                output_path,
            )
            self.assertEqual(rendered.returncode, 0, rendered.stderr)
            validated = run_script(VALIDATE, output_path)
            self.assertEqual(validated.returncode, 0, validated.stderr)


    def test_validator_rejects_source_count_mismatches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            direct_path = temp / "direct-source-mismatch.txt"
            direct_path.write_text(
                """[streammux]
batch-size=2

[primary-gie]
enable=0

[source0]
enable=1
type=4
uri=rtsp://camera.example/stream
num-sources=1

[sink0]
enable=1
type=1
""",
                encoding="utf-8",
            )
            direct = run_script(VALIDATE, direct_path)
            self.assertEqual(direct.returncode, 1)
            self.assertIn("materialized source count (1)", direct.stderr)
            self.assertIn("streammux.batch-size (2)", direct.stderr)

            source_list_path = temp / "source-list-mismatch.txt"
            source_list_path.write_text(
                """[source-list]
num-source-bins=2
list=file:///tmp/input.mp4
use-nvmultiurisrcbin=0
max-batch-size=2

[streammux]
batch-size=2

[primary-gie]
enable=0

[sink0]
enable=1
type=1
""",
                encoding="utf-8",
            )
            source_list = run_script(VALIDATE, source_list_path)
            self.assertEqual(source_list.returncode, 1)
            self.assertIn("source-list.list has 1 entries", source_list.stderr)
            self.assertIn("num-source-bins=2", source_list.stderr)


    def test_invalid_specs_fail_without_traceback_or_output(self):
        cases = {
            "unknown-key": (
                "bogus_key: 1\n",
                ("unknown top-level spec key", "bogus_key"),
            ),
            "unknown-source-key": (
                "sources:\n  num_sorces: 8\n",
                ("unknown spec.sources key", "num_sorces"),
            ),
            "wrong-streammux-batch-shape": (
                "streammux:\n  batch_size: []\n",
                ("spec.streammux.batch_size", "integer", "[]"),
            ),
            "non-string-top-level-key": (
                "1: value\n",
                ("unknown top-level spec key", "1"),
            ),
            "non-string-uri": (
                """sources:
  kind: source_list
  uris:
    - 1
""",
                ("spec.sources.uris[0]", "string", "int"),
            ),
            "unknown-source-kind": (
                """sources:
  kind: bogus
""",
                ("spec.sources.kind", "must be one of", "bogus"),
            ),
            "non-string-source-kind": (
                """sources:
  kind: [bogus]
""",
                ("spec.sources.kind", "must be one of", "['bogus']"),
            ),
            "invalid-num-sources": (
                """sources:
  kind: multi_uri
  num_sources: nope
""",
                ("spec.sources.num_sources", "positive integer", "nope"),
            ),
            "rtsp-source-count-mismatch": (
                """sources:
  kind: rtsp
  uris:
    - rtsp://camera.example/stream
  num_sources: 2
""",
                ("spec.sources.num_sources (2)", "len(spec.sources.uris) (1)", "rtsp"),
            ),
            "source-list-count-mismatch": (
                """sources:
  kind: source_list
  uris:
    - file:///tmp/one.mp4
    - file:///tmp/two.mp4
  num_sources: 3
""",
                ("one reusable URI or exactly", "num_sources (3)", "got 2"),
            ),
            "device-source-count-mismatch": (
                """sources:
  kind: v4l2
  num_sources: 2
""",
                ("spec.sources.num_sources", "must be 1", "v4l2"),
            ),
            "multi-uri-discards-extra-uri": (
                """sources:
  kind: multi_uri
  uris:
    - file:///tmp/one.mp4
    - file:///tmp/two.mp4
  num_sources: 2
""",
                ("multi_uri", "at most one URI", "got 2"),
            ),
            "wrong-sources-shape": (
                "sources: []\n",
                ("spec.sources", "mapping", "list"),
            ),
            "wrong-custom-gie-field-types": (
                """primary_gie:
  enable: true
  preset: custom
  config_file: []
  model_engine_file: {}
""",
                ("spec.primary_gie", "config_file", "model_engine_file"),
            ),
            "wrong-generated-infer-path-type": (
                """primary_gie:
  enable: true
  preset: custom
  generate_infer_config: true
  infer_config:
    onnx_file: []
""",
                ("spec.primary_gie.infer_config.onnx_file", "string", "list"),
            ),
            "incomplete-custom-gie": (
                """primary_gie:
  enable: true
  preset: custom
  config_file: ''
  model_engine_file: ''
""",
                ("spec.primary_gie", "config_file", "model_engine_file"),
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            for name, (contents, expected_parts) in cases.items():
                with self.subTest(name=name):
                    spec_path = temp / f"{name}.yaml"
                    output_path = temp / f"{name}.txt"
                    spec_path.write_text(contents, encoding="utf-8")
                    result = run_script(
                        RENDER,
                        "--spec",
                        spec_path,
                        "--out",
                        output_path,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    for part in expected_parts:
                        self.assertIn(part, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertFalse(output_path.exists())

    def test_generated_sidecars_cannot_escape_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            output_dir = temp / "output"
            spec_path = temp / "sidecar-escape.yaml"
            output_path = output_dir / "generated.txt"
            escaped_path = temp / "escaped-infer.txt"
            spec_path.write_text(
                """sources:
  kind: multi_uri
  uris:
    - file:///tmp/input.mp4
  num_sources: 1
primary_gie:
  enable: true
  preset: trafficcamnet
  generate_infer_config: true
  infer_config_path: ../escaped-infer.txt
sinks:
  - type: fakesink
""",
                encoding="utf-8",
            )

            result = run_script(RENDER, "--spec", spec_path, "--out", output_path)

            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("escapes the main output directory", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(escaped_path.exists())
            self.assertFalse(output_path.exists())

    def test_quickstart_and_asset_backed_helpers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            output_path = temp / "quickstart.txt"
            rendered = run_script(
                RENDER,
                "--quickstart",
                "--print-resolved-spec",
                "--out",
                output_path,
            )
            self.assertEqual(rendered.returncode, 0, rendered.stderr)
            self.assertTrue(output_path.is_file())
            self.assertIn("[render] resolved spec:", rendered.stderr)

            validated = run_script(VALIDATE, output_path)
            self.assertEqual(validated.returncode, 0, validated.stderr)

            source_list_spec = temp / "source-list-replication.yaml"
            source_list_output = temp / "source-list-replication.txt"
            source_list_spec.write_text(
                """sources:
  kind: source_list
  uris:
    - file:///tmp/input.mp4
  num_sources: 2
primary_gie:
  enable: 0
sinks:
  - type: fakesink
""",
                encoding="utf-8",
            )
            source_list_rendered = run_script(
                RENDER,
                "--spec",
                source_list_spec,
                "--out",
                source_list_output,
            )
            self.assertEqual(source_list_rendered.returncode, 0, source_list_rendered.stderr)
            self.assertIn(
                "list=file:///tmp/input.mp4;file:///tmp/input.mp4",
                source_list_output.read_text(encoding="utf-8"),
            )
            source_list_validated = run_script(VALIDATE, source_list_output)
            self.assertEqual(
                source_list_validated.returncode,
                0,
                source_list_validated.stderr,
            )

            infer_path = temp / "config_infer.txt"
            infer = run_script(
                RENDER_INFER,
                "--network-type",
                "detector",
                "--batch-size",
                "1",
                "--onnx-file",
                "model.onnx",
                "--out",
                infer_path,
            )
            self.assertEqual(infer.returncode, 0, infer.stderr)
            self.assertTrue(infer_path.is_file())

            tracker_path = temp / "config_tracker.yml"
            tracker = run_script(
                RENDER_TRACKER,
                "--algorithm",
                "NvDCF",
                "--max-targets-per-stream",
                "200",
                "--out",
                tracker_path,
            )
            self.assertEqual(tracker.returncode, 0, tracker.stderr)
            self.assertTrue(tracker_path.is_file())
            self.assertIn(
                "maxTargetsPerStream: 200",
                tracker_path.read_text(encoding="utf-8"),
            )

    def test_security_guidance_and_static_access(self):
        tracker_text = RENDER_TRACKER.read_text(encoding="utf-8")
        self.assertNotIn("getattr(args", tracker_text)
        self.assertIn("argument_values.get(cli_name)", tracker_text)

        # NVCARPS regenerates skill-card.md with evaluation results before this
        # test runs. The executable safety contract belongs in SKILL.md, so do
        # not couple it to wording in that generated catalog artifact.
        skill_text = SKILL_MD.read_text(encoding="utf-8").lower()
        self.assertIn("already-present output path", skill_text)
        self.assertIn("explicit confirmation", skill_text)
        self.assertIn("commands are proposals for", skill_text)
        self.assertIn("not permission to execute them", skill_text)

        bundle_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in SKILL_ROOT.rglob("*")
            if path.is_file()
            and path.suffix in {".json", ".md", ".py", ".txt", ".yaml", ".yml"}
            and path.name not in {"BENCHMARK.md", "skill-card.md"}
        ).lower()
        self.assertNotIn("overwrite existing file", bundle_text)

        qa_text = INTERACTIVE_QA.read_text(encoding="utf-8")
        self.assertNotIn("](profiles.md)", qa_text)



if __name__ == "__main__":
    unittest.main()
