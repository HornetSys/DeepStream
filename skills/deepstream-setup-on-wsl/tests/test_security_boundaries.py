#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = SKILL_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


configure_target = load("configure_target_test", SCRIPTS_DIR / "configure_target.py")
winrm_exec = load("winrm_exec_test", SCRIPTS_DIR / "winrm_exec.py")
run_verification = load("run_verification_test", SCRIPTS_DIR / "run_verification.py")
run_custom = load("run_custom_test", SCRIPTS_DIR / "run_custom_pipeline.py")


class TargetProfileTests(unittest.TestCase):
    def test_create_and_delete_private_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(
                name="test-box",
                host="win.example.test",
                user="Administrator",
                targets_dir=temp,
            )
            with mock.patch.object(sys.stdin, "isatty", return_value=True), mock.patch.object(
                configure_target.getpass,
                "getpass",
                side_effect=["correct horse", "correct horse"],
            ):
                configure_target.create(args)
            profile = Path(temp) / "test-box" / "target.json"
            password = profile.parent / "winrm.pass"
            self.assertEqual(stat.S_IMODE(profile.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(password.stat().st_mode), 0o600)
            self.assertEqual(json.loads(profile.read_text())["transport"], "https")
            self.assertEqual(password.read_text(), "correct horse")

            configure_target.delete(SimpleNamespace(profile=str(profile), yes=True))
            self.assertFalse(profile.exists())
            self.assertFalse(password.exists())

    def test_password_file_with_loose_permissions_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "password"
            path.write_text("secret")
            path.chmod(0o644)
            with self.assertRaises(SystemExit):
                winrm_exec._private_regular_file(str(path), "password file")


class TransportTests(unittest.TestCase):
    def test_ssh_requires_a_preverified_host_key(self):
        script = (SCRIPTS_DIR / "remote_wsl_exec.sh").read_text()
        self.assertIn("StrictHostKeyChecking=yes", script)
        self.assertEqual(script.count("StrictHostKeyChecking="), 1)

    def test_ssh_remote_execution_requires_confirmation_and_logging(self):
        script_path = SCRIPTS_DIR / "remote_wsl_exec.sh"
        result = subprocess.run(
            [
                str(script_path),
                "--host",
                "host.example.test",
                "--cmd",
                "printf ok",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("explicit confirmation required", result.stderr)
        script = script_path.read_text()
        self.assertIn("REMOTE_EXECUTION=command", script)
        self.assertIn("REMOTE_COMMAND=", script)
        self.assertIn("REMOTE_EXECUTION=payload", script)
        self.assertIn("REMOTE_PAYLOAD=", script)

    def _profile(self, root: Path, transport: str) -> Path:
        root.chmod(0o700)
        password = root / "winrm.pass"
        password.write_text("secret")
        password.chmod(0o600)
        data = {
            "schema_version": 1,
            "host": "win.example.test",
            "user": "Administrator",
            "transport": transport,
            "pass_file": "winrm.pass",
        }
        profile = root / "target.json"
        profile.write_text(json.dumps(data))
        profile.chmod(0o600)
        return profile

    def test_public_profile_resolves_https(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = self._profile(Path(temp), "https")
            args = SimpleNamespace(
                target_profile=str(profile),
                host=None,
                user=None,
                pass_file=None,
                use_https=False,
            )
            self.assertEqual(winrm_exec.resolve_connection(args), "secret")
            self.assertTrue(args.use_https)
            self.assertFalse(args.allow_http)

    def test_http_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = self._profile(Path(temp), "http")
            args = SimpleNamespace(
                target_profile=str(profile),
                host=None,
                user=None,
                pass_file=None,
                use_https=False,
            )
            with self.assertRaises(SystemExit):
                winrm_exec.resolve_connection(args)

    def test_upload_propagates_https_to_every_call(self):
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(kwargs)
            command = args[3]
            if "Get-Item" in command:
                return 0, "4", ""
            return 0, "", ""

        with tempfile.NamedTemporaryFile() as source, mock.patch.object(
            winrm_exec, "_run_ps", side_effect=fake_run
        ):
            source.write(b"x")
            source.flush()
            # One input byte becomes four base64 bytes.
            winrm_exec.upload_binary(
                "host",
                "user",
                "password",
                source.name,
                r"C:\dsw\x",
                use_https=True,
                allow_http=False,
            )
        self.assertGreaterEqual(len(calls), 4)
        self.assertTrue(all(call["use_https"] for call in calls))
        self.assertTrue(all(not call["allow_http"] for call in calls))

    def test_upload_chunk_fits_winrm_command_line(self):
        self.assertLessEqual(winrm_exec.CHUNK_BYTES, 1800)

    def test_staging_paths_reject_command_injection(self):
        rejected = (
            r"C:\dsw\x'; Write-Output injected; '",
            r"C:\Windows\System32\x",
        )
        for path in rejected:
            with self.subTest(path=path), self.assertRaises(SystemExit):
                winrm_exec._safe_windows_staging_path(path)
        with self.assertRaises(SystemExit):
            winrm_exec._safe_wsl_path("/tmp/result;id")

    def test_transport_encryption_settings_are_fail_closed(self):
        protocol = mock.MagicMock()
        protocol.open_shell.return_value = "shell"
        protocol.run_command.return_value = "command"
        protocol.get_command_output.return_value = (b"", b"", 0)
        factory = mock.Mock(return_value=protocol)
        with mock.patch.object(winrm_exec, "_protocol_class", return_value=factory):
            with contextlib.redirect_stderr(io.StringIO()):
                winrm_exec._run_ps(
                    "host", "user", "password", "Write-Output ok", 10,
                    use_https=False, **{"allow_http": bool(1)},
                )
        self.assertEqual(factory.call_args.kwargs["message_encryption"], "always")
        self.assertNotIn("server_cert_validation", factory.call_args.kwargs)

        factory.reset_mock()
        with mock.patch.object(winrm_exec, "_protocol_class", return_value=factory):
            winrm_exec._run_ps(
                "host", "user", "password", "Write-Output ok", 10,
                use_https=True, allow_http=False,
            )
        self.assertEqual(factory.call_args.kwargs["server_cert_validation"], "validate")
        self.assertNotIn("message_encryption", factory.call_args.kwargs)


class CustomCommandTests(unittest.TestCase):
    def test_single_gstreamer_invocation_is_accepted(self):
        argv = run_custom._validate_cmd(
            "gst-launch-1.0 filesrc location=/work/in.mp4 ! decodebin ! fakesink"
        )
        self.assertEqual(argv[0], "gst-launch-1.0")

    def test_shell_chaining_and_python_code_are_rejected(self):
        rejected = (
            "gst-launch-1.0 --version; printf rejected",
            "gst-launch-1.0 --version | tee /work/out",
            "python3 -c 'import os'",
            "python-malicious /work/app.py",
            "/usr/bin/bash /work/payload.sh",
        )
        for command in rejected:
            with self.subTest(command=command), self.assertRaises(SystemExit):
                run_custom._validate_cmd(command)

    def test_missing_yes_fails_before_connection_or_upload(self):
        args = SimpleNamespace(
            ds_image="nvcr.io/nvidia/deepstream:9.1-triton-multiarch",
            distro="Ubuntu-24.04",
            cmd="gst-launch-1.0 --version",
            name="test",
            yes=False,
        )
        with mock.patch.object(
            run_custom.W,
            "resolve_connection",
            side_effect=AssertionError("connection reached before confirmation"),
        ):
            with self.assertRaisesRegex(SystemExit, "explicit confirmation required"):
                run_custom.run(args)

    def test_custom_container_overrides_image_entrypoint(self):
        job, _ = run_custom._remote_job(
            "nvcr.io/nvidia/deepstream:9.1-triton-multiarch",
            60,
            "0123456789abcdef",
            "/mnt/c/dsw/bundle.tgz",
            False,
        )
        self.assertIn("--entrypoint /bin/bash", job)
        self.assertIn("--network none", job)

    def test_custom_input_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "input.mp4"
            target.write_bytes(b"video")
            link = Path(temp) / "link.mp4"
            link.symlink_to(target)
            with self.assertRaises(SystemExit):
                run_custom._make_bundle("gst-launch-1.0 --version", [str(link)])


class ProvisioningContractTests(unittest.TestCase):
    def setUp(self):
        self.script = (SCRIPTS_DIR / "wsl_provision_deepstream.sh").read_text()

    def test_host_toolkit_is_opt_in_and_versioned(self):
        self.assertIn('HOST_CUDA_TOOLKIT_MODE="${HOST_CUDA_TOOLKIT_MODE:-skip}"', self.script)
        self.assertIn('package="cuda-toolkit-${HOST_CUDA_TOOLKIT_VERSION//./-}"', self.script)
        self.assertIn("CUDA repository bootstrap checksum mismatch", self.script)
        self.assertEqual(self.script.count('repo_sha256="'), 3)
        self.assertNotIn("cuda-drivers", self.script)

    def test_driver_gate_precedes_state_changes(self):
        main = self.script.rsplit("\nrequire_wsl\n", 1)[1]
        self.assertLess(main.index("verify_driver"), main.index("handle_host_cuda_toolkit"))
        self.assertIn('MIN_WINDOWS_DRIVER="${MIN_WINDOWS_DRIVER:-580.00}"', self.script)
        self.assertIn('VALIDATED_WINDOWS_DRIVER="${VALIDATED_WINDOWS_DRIVER:-595.79}"', self.script)

    def test_container_cuda_and_tensorrt_are_verified(self):
        self.assertIn("verify_container_stack", self.script)
        self.assertIn("EXPECTED_CONTAINER_CUDA", self.script)
        self.assertIn("EXPECTED_TENSORRT", self.script)
        self.assertIn("libnvinfer", self.script)
        self.assertIn("--entrypoint /bin/bash", self.script)
        self.assertIn("--entrypoint /usr/bin/nvidia-smi", self.script)


if __name__ == "__main__":
    unittest.main()
