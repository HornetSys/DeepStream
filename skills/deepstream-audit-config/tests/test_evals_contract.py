# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for Litmus findings in NVBug 6800443."""

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
SKILL = SKILL_ROOT / "SKILL.md"
EVALS = SKILL_ROOT / "evals" / "evals.json"
EVAL_CONFIG = SKILL_ROOT / "evals" / "config.yml"
VALIDATOR = SKILL_ROOT / "scripts" / "validate_config.py"
NTP_PROBE = SKILL_ROOT / "scripts" / "ntp_probe.py"
DESIGN_REVIEW = SKILL_ROOT / "references" / "design_review.md"
CONFIG_EDITS = SKILL_ROOT / "references" / "config_edits.md"
REQUIREMENTS = SKILL_ROOT / "requirements.txt"
BENCHMARK = SKILL_ROOT / "BENCHMARK.md"
SKILL_CARD = SKILL_ROOT / "skill-card.md"


def load_evals():
    return json.loads(EVALS.read_text(encoding="utf-8"))["evals"]


class TestAuditEvalContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_evals()

    def test_schema_activation_and_tier3_assertions(self):
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        script_cases = {
            "deepstream-audit-config-001",
            "deepstream-audit-config-002",
            "deepstream-audit-config-004",
        }
        for case in self.cases:
            for field in (
                "id",
                "name",
                "prompt",
                "expected_skill",
                "expected_script",
                "should_trigger",
                "expected_output",
                "files",
                "assertions",
            ):
                self.assertIn(field, case)
            self.assertIsInstance(case["should_trigger"], bool)
            expected = "deepstream-audit-config" if case["should_trigger"] else None
            self.assertEqual(case["expected_skill"], expected)
            expected_script = (
                "scripts/validate_config.py" if case["id"] in script_cases else None
            )
            self.assertEqual(case["expected_script"], expected_script)
            if case["should_trigger"]:
                self.assertFalse(case["name"].startswith("negative-"))
            self.assertTrue(case["assertions"])
            for assertion in case["assertions"]:
                self.assertIsInstance(assertion, str)
                self.assertTrue(assertion.strip())

    def test_three_file_backed_fixtures(self):
        fixture_cases = [case for case in self.cases if case["files"]]
        self.assertEqual(len(fixture_cases), 3)
        for case in fixture_cases:
            for relative in case["files"]:
                self.assertTrue(
                    (SKILL_ROOT / relative).is_file(),
                    f"{case['id']} fixture missing: {relative}",
                )

    def test_creation_request_routes_without_audit_activation(self):
        case = next(
            case for case in self.cases if case["id"] == "deepstream-audit-config-005"
        )
        self.assertFalse(case["should_trigger"])
        self.assertIsNone(case["expected_skill"])
        self.assertIsNone(case["expected_script"])
        self.assertEqual(case["files"], [])
        contract = " ".join([case["expected_output"], *case["assertions"]]).lower()
        self.assertIn("names deepstream-generate-config", contract)
        self.assertIn("does not invoke deepstream-audit-config", contract)
        self.assertIn("does not run scripts/validate_config.py", contract)

        skill_contract = SKILL.read_text(encoding="utf-8").lower()
        self.assertIn("creation belongs", skill_contract)
        self.assertIn(
            "do not load this skill or run `validate_config.py`", skill_contract
        )
        self.assertIn("in that creation turn", skill_contract)

    def test_published_metadata_matches_eval_contract(self):
        total = len(self.cases)
        positive = sum(case["should_trigger"] for case in self.cases)
        negative = total - positive
        benchmark = BENCHMARK.read_text(encoding="utf-8")
        skill_card = SKILL_CARD.read_text(encoding="utf-8")
        self.assertIn(
            f"- Tasks: {total} evaluation tasks "
            f"({positive} positive, {negative} negative)",
            benchmark,
        )
        self.assertIn(
            f"{total} evaluation tasks ({positive} positive, {negative} negative)",
            skill_card,
        )

        skill_text = SKILL.read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(skill_text.split("---", 2)[1])
        version = frontmatter["metadata"]["version"]
        self.assertIn(f"{version} (source: frontmatter)", skill_card)

    def test_canonical_tier3_source_is_declared(self):
        config = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(config["harbor"]["task_source"], "evals_json")

    def test_readiness_contract_is_documented(self):
        text = SKILL.read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(text.split("---", 2)[1])
        self.assertIn("Use only when", frontmatter["description"])
        self.assertIn("Not for", frontmatter["description"])
        self.assertIn("design/code change", frontmatter["description"])
        self.assertEqual(frontmatter["license"], "CC-BY-4.0 AND Apache-2.0")
        self.assertIn("## When to Use This Skill", text)
        self.assertIn("## Workflow", text)
        self.assertIn("The validator steps above apply only", text)
        self.assertIn("Expected: exit `1`", text)
        self.assertIn("bundled skill currently owns those workflows", text)
        self.assertIn("--target-platform", text)
        self.assertIn("--json", text)
        self.assertIn(
            "### `validate_config.py` machine-readable output and exit codes", text
        )
        self.assertIn("| `2` | CLI/input error", text)
        self.assertIn("PyYAML>=6.0", REQUIREMENTS.read_text(encoding="utf-8"))
        self.assertTrue((SKILL_ROOT / "evals" / "files").is_dir())

        repo_root = SKILL_ROOT.parents[1]
        repo_license = (repo_root / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("CC-BY-4.0", repo_license)
        self.assertIn("Apache-2.0", repo_license)
        for sibling in (
            "deepstream-generate-config",
            "deepstream-inspect-rtsp",
            "deepstream-profile-pipeline",
        ):
            self.assertTrue((SKILL_ROOT.parent / sibling).is_dir(), sibling)

        for discovered in (
            "deepstream-audit-config",
            "deepstream-generate-config",
            "deepstream-inspect-rtsp",
            "deepstream-profile-pipeline",
        ):
            alias = repo_root / ".claude" / "skills" / discovered
            self.assertTrue(alias.is_symlink(), discovered)
            self.assertTrue(alias.resolve().is_dir(), discovered)

    def test_design_review_source_paths_resolve(self):
        text = DESIGN_REVIEW.read_text(encoding="utf-8")
        source_paths = (
            "src/apps/common/src/deepstream_config_file_parser.c",
            "src/apps/common/src/deepstream_streammux.c",
            "src/apps/common/src/deepstream_source_bin.c",
            "src/apps/sample_apps/deepstream-test5/deepstream_test5_app_main.c",
            "src/gst-plugins/gst-nvmultiurisrcbin/gstdsnvmultiurisrcbin.cpp",
            "src/gst-utils/gstnvdscustomhelper/gst-nvmultiurisrcbincreator.cpp",
        )
        for relative in source_paths:
            self.assertIn(f"`{relative}`", text)
            self.assertTrue((SKILL_ROOT.parents[1] / relative).is_file(), relative)

    def test_ntp_probe_help_is_real_help(self):
        result = subprocess.run(
            [sys.executable, str(NTP_PROBE), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Usage:", result.stdout)
        self.assertIn("timeout-sec", result.stdout)
        self.assertIn("1  Unhandled Python/runtime failure", result.stdout)
        self.assertNotIn("2  Unexpected failure", result.stdout)
        self.assertNotIn('"status": "pipeline-error"', result.stdout)

    def test_http_probe_rejects_non_http_redirects(self):
        spec = importlib.util.spec_from_file_location("audit_validator", VALIDATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        handler = module._HttpOnlyRedirectHandler()
        request = urllib.request.Request("https://example.invalid/media")
        with self.assertRaises(urllib.error.HTTPError):
            handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "file:///path/to/config",
            )

    def test_fatal_marker_reader_is_complete(self):
        spec = importlib.util.spec_from_file_location(
            "audit_validator_markers", VALIDATOR
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.assertEqual(
            module.fatal_markers(
                "pipeline started\nERROR missing model\nSegmentation fault\n"
            ),
            ["ERROR missing model", "Segmentation fault"],
        )

    def test_pipeline_gate_lands_in_executable_workflow(self):
        text = SKILL.read_text(encoding="utf-8")
        self.assertIn("Step 4: If static validation reports any ERROR, STOP", text)
        self.assertIn("Step 5: STOP with the static result", text)
        self.assertIn(
            "--probe-pipeline --confirm-pipeline-probe --timeout-sec <N>", text
        )
        self.assertIn("broker messages cannot be", text)

    def test_edit_revalidation_cycle_is_bounded(self):
        skill_text = SKILL.read_text(encoding="utf-8")
        edit_policy = CONFIG_EDITS.read_text(encoding="utf-8")
        self.assertIn("at most 1 edit/revalidation retry", skill_text)
        self.assertIn("abort further edits", skill_text)
        self.assertIn("do not repeat the edit/re-validation cycle", skill_text)
        self.assertIn("at most 1 edit/revalidation retry", edit_policy)
        self.assertIn("1 static re-validation", edit_policy)
        self.assertIn("abort further edits", edit_policy)

    def test_malformed_yaml_returns_structured_finding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "malformed.yml"
            config.write_text("source0:\n  uri: [unterminated\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(config), "--json"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=10,
            )

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertNotIn("Traceback", result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["summary"]["errors"], 1)
        self.assertEqual(payload["findings"][0]["severity"], "ERROR")
        self.assertEqual(payload["findings"][0]["area"], "parse")
        self.assertIn("Malformed YAML", payload["findings"][0]["message"])

    def test_yaml_read_error_returns_structured_finding(self):
        spec = importlib.util.spec_from_file_location("audit_validator_yaml", VALIDATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.FINDINGS.clear()

        with mock.patch.object(
            Path,
            "open",
            side_effect=PermissionError("do-not-expose-this-detail"),
        ):
            config = module.load_yml(Path("unreadable.yml"))

        self.assertEqual(config, {})
        self.assertEqual(len(module.FINDINGS), 1)
        finding = module.FINDINGS[0]
        self.assertEqual(finding.severity, "ERROR")
        self.assertEqual(finding.area, "parse")
        self.assertIn("Unable to read YAML config (PermissionError)", finding.message)
        self.assertNotIn("do-not-expose-this-detail", finding.message)

    def run_validator(self, fixture, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                str(SKILL_ROOT / "evals" / "files" / fixture),
                *extra,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=20,
        )

    def test_rtsp_fixture_produces_real_findings(self):
        result = self.run_validator(
            "app_rtsp_bad.txt", "--intent", "4 production RTSP cameras"
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("primary-gie.batch-size (1)", result.stdout)
        self.assertIn("active source count appears to be 4", result.stdout)
        self.assertIn("attach-sys-ts-as-ntp", result.stdout)
        self.assertIn("live-source", result.stdout)

    def test_latency_fixture_produces_jetson_and_latency_findings(self):
        result = self.run_validator(
            "latency_filesrc_jetson.yml",
            "--intent",
            "low-latency file pipeline on Jetson",
            "--target-platform",
            "jetson",
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("should not use EglSink type=2", result.stdout)
        self.assertIn("batched-push-timeout=100000", result.stdout)
        self.assertIn("fakesink", result.stdout)

    def test_pipeline_probe_requires_explicit_confirmation(self):
        result = self.run_validator(
            "rtsp_probe_gate.txt",
            "--intent",
            "check exact pipeline",
            "--probe-pipeline",
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "--probe-pipeline requires --confirm-pipeline-probe", result.stdout
        )

    def test_static_default_has_no_runtime_probe_findings(self):
        result = self.run_validator(
            "latency_filesrc_jetson.yml",
            "--intent",
            "low-latency file pipeline on Jetson",
            "--target-platform",
            "jetson",
        )
        self.assertNotIn("[runtime]", result.stdout)


if __name__ == "__main__":
    unittest.main()
