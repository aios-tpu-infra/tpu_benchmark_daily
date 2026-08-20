import importlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "vllm-service-launch"
LAUNCHER = VENDOR_ROOT / "bin" / "vllm-service-launch"
INSTALLER = PROJECT_ROOT / "scripts" / "install_vllm_service_launcher.sh"
LIBRARY_ROOT = VENDOR_ROOT / "lib"

if str(LIBRARY_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBRARY_ROOT))

from vllm_service_launch import schema as launcher_schema
from vllm_service_launch import service as launcher_service
from vllm_service_launch import state as launcher_state


class VllmServiceLauncherProcessTest(unittest.TestCase):
    def test_reused_pid_identity_is_never_signaled(self) -> None:
        process = importlib.import_module("vllm_service_launch.process")
        child = subprocess.Popen(["sleep", "10"], start_new_session=True)
        try:
            observed = process.read_process(child.pid)
            self.assertIsNotNone(observed)
            identity, process_group, session_id, _ = observed
            reused = launcher_schema.ProcessIdentity(
                identity.pid,
                identity.start_time + 1,
            )

            signaled = process.signal_process_group(
                reused,
                process_group,
                session_id,
                signal.SIGTERM,
            )

            self.assertFalse(signaled)
            self.assertIsNone(child.poll())
        finally:
            child.terminate()
            child.wait(timeout=3)

    def test_missing_leader_does_not_authorize_numeric_process_group(self) -> None:
        process = importlib.import_module("vllm_service_launch.process")
        leader = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import os,time; child=os.fork(); "
                    "time.sleep(10) if child == 0 else os._exit(0)"
                ),
            ],
            start_new_session=True,
        )
        try:
            observed = process.read_process(leader.pid)
            self.assertIsNotNone(observed)
            identity, process_group, session_id, _ = observed
            leader.wait(timeout=3)
            self.assertIsNone(process.read_process(identity.pid))
            self.assertTrue(process.group_members(process_group, session_id))

            self.assertFalse(
                process.process_group_is_alive(
                    identity,
                    process_group,
                    session_id,
                )
            )
            self.assertFalse(
                process.signal_process_group(
                    identity,
                    process_group,
                    session_id,
                    signal.SIGTERM,
                )
            )
        finally:
            try:
                os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if leader.poll() is None:
                leader.wait(timeout=3)


class VllmServiceLauncherStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.state_root = self.root / "state"
        self.target_root = self.root / "targets"
        self.target_root.mkdir()
        try:
            self.layout = launcher_state.RuntimeLayout.from_roots(
                self.state_root,
                self.target_root,
            )
        except AttributeError as exc:
            self.fail(f"explicit state/target roots are missing: {exc}")
        candidate = launcher_schema.CandidateRequest.from_dict(
            {
                "schema_version": 1,
                "service_id": "test-prefill",
                "role": "prefill",
                "model_alias": "fake-model",
                "environment": {
                    "kind": "uv",
                    "executable": "/usr/bin/uv",
                    "prefix": "/tmp/fake-venv",
                    "vllm_executable": "/tmp/fake-venv/bin/vllm",
                    "project": "/tmp/fake-project",
                },
                "env_files": [],
                "working_directory": "/tmp",
                "listen_host": "127.0.0.1",
                "port_policy": {"mode": "fixed", "port": 18100},
                "runtime": "vllm",
                "vllm_argv": ["serve", "/tmp/fake-model"],
            }
        )
        try:
            self.request = launcher_schema.ServiceRequest(
                request_id="2" * 32,
                candidate=candidate,
                starter=launcher_schema.ProcessIdentity(101, 1001),
                supervisor=None,
                cancellation_requested=False,
            )
        except TypeError as exc:
            self.fail(f"systemd-free request schema is missing: {exc}")
        self.runtime = launcher_schema.RuntimeState(
            request_id=self.request.request_id,
            service_id=self.request.service_id,
            supervisor=launcher_schema.ProcessIdentity(101, 1001),
            server=launcher_schema.ProcessIdentity(202, 2002),
            server_pgid=202,
            server_session_id=202,
            listen_host="127.0.0.1",
            scrape_host="127.0.0.1",
            port=18100,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_runtime_publication_requires_claimed_supervisor(self) -> None:
        launcher_state.reserve_request(self.layout, self.request)

        with self.assertRaises(launcher_state.StateError):
            launcher_state.publish_runtime(
                self.layout,
                self.request,
                self.runtime,
            )

        self.assertFalse(self.layout.runtime_path(self.request.service_id).exists())
        self.assertFalse(self.layout.target_path(self.request.service_id).exists())

    def test_status_retains_unclaimed_request_for_explicit_stop(self) -> None:
        try:
            stale_request = launcher_schema.ServiceRequest(
                request_id="3" * 32,
                candidate=self.request.candidate,
                starter=launcher_schema.ProcessIdentity(2_147_483_647, 1),
                supervisor=None,
                cancellation_requested=False,
            )
        except TypeError as exc:
            self.fail(f"pending process ownership is missing: {exc}")
        launcher_state.reserve_request(self.layout, stale_request)

        with self.assertRaises(launcher_service.ServiceError):
            launcher_service.service_status(
                self.layout,
                stale_request.service_id,
            )

        self.assertTrue(
            self.layout.request_path(stale_request.service_id).exists()
        )
        launcher_service.stop_service(self.layout, stale_request.service_id)
        self.assertFalse(
            self.layout.request_path(stale_request.service_id).exists()
        )


class VllmServiceLauncherLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.state_root = self.root / "state"
        self.target_root = self.root / "targets"
        self.target_root.mkdir()
        self.project = self.root / "project"
        self.project.mkdir()
        self.bin_directory = self.root / "bin"
        self.bin_directory.mkdir()
        self.capture_path = self.root / "capture.json"
        self.log_path = self.root / "server.log"
        self.launcher = LAUNCHER
        self.fake_server = self.root / "fake-server"
        self._write_executable(
            self.fake_server,
            """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import signal
import sys
import time

capture = Path(os.environ["FAKE_CAPTURE_PATH"])
capture.write_text(json.dumps({
    "argv": sys.argv[1:],
    "feature": os.environ.get("FEATURE_WITHOUT_WHITELIST"),
    "pid": os.getpid(),
}), encoding="utf-8")
print("fake-server-stdout", flush=True)
signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))
signal.signal(signal.SIGINT, lambda _signum, _frame: sys.exit(0))
if os.environ.get("FAKE_FORK_WORKER") == "1":
    worker_pid = os.fork()
    if worker_pid:
        payload = json.loads(capture.read_text(encoding="utf-8"))
        payload["worker_pid"] = worker_pid
        capture.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(0.1)
        raise SystemExit(0)
exit_after = os.environ.get("FAKE_EXIT_AFTER")
if exit_after is not None:
    time.sleep(float(exit_after))
    raise SystemExit(0)
while True:
    time.sleep(0.05)
""",
        )
        fake_vllm = self.bin_directory / "vllm"
        self._write_executable(fake_vllm, "#!/bin/sh\nexit 0\n")
        fake_uv = self.bin_directory / "uv"
        self._write_executable(
            fake_uv,
            """#!/usr/bin/env python3
import os
from pathlib import Path
import signal
import sys
import time

arguments = sys.argv[1:]
if not arguments or arguments.pop(0) != "run":
    raise SystemExit(2)
while arguments and arguments[0] in {"--project", "--no-sync"}:
    option = arguments.pop(0)
    if option == "--project":
        arguments.pop(0)
if arguments == ["/usr/bin/env", "-0"] and os.environ.get("FAKE_UV_HANG") == "1":
    Path(os.environ["FAKE_ACTIVATION_PID_PATH"]).write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))
    while True:
        time.sleep(0.05)
os.execvpe(arguments[0], arguments, os.environ)
""",
        )
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "PATH": f"{self.bin_directory}:{self.environment['PATH']}",
                "FAKE_CAPTURE_PATH": str(self.capture_path),
                "FEATURE_WITHOUT_WHITELIST": "visible",
                "PYTHONUNBUFFERED": "1",
            }
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]

    def tearDown(self) -> None:
        if self.state_root.exists():
            self._run_launcher(
                "stop",
                "--service-id",
                "test-prefill",
                check=False,
            )
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_executable(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    def _run_launcher(
        self,
        *arguments: str,
        check: bool = False,
        stdout: object = subprocess.PIPE,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(self.launcher),
            arguments[0],
            "--state-root",
            str(self.state_root),
            "--target-root",
            str(self.target_root),
            *arguments[1:],
        ]
        return subprocess.run(
            command,
            cwd=self.root,
            env=self.environment,
            check=check,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )

    def _wait_for_path(self, path: Path) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if path.exists():
                return
            time.sleep(0.02)
        self.fail(f"timed out waiting for {path}")

    def _wait_for_absence(self, path: Path) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if not path.exists():
                return
            time.sleep(0.02)
        self.fail(f"timed out waiting for removal of {path}")

    def _start(self) -> subprocess.CompletedProcess[str]:
        with self.log_path.open("a", encoding="utf-8") as log_file:
            return self._run_launcher(
                "start",
                "--service-id",
                "test-prefill",
                "--role",
                "prefill",
                "--model-alias",
                "fake-model",
                "--runtime",
                "dashllm",
                "--uv-project",
                str(self.project),
                "--working-directory",
                str(self.root),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--",
                str(self.fake_server),
                "--feature",
                "enabled",
                stdout=log_file,
            )

    def _runtime_payload(self) -> dict[str, object]:
        path = self.state_root / "services/test-prefill/runtime.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_stop_cancels_start_during_environment_activation(self) -> None:
        activation_pid_path = self.root / "stop-activation.pid"
        environment = dict(self.environment)
        environment.update(
            {
                "FAKE_UV_HANG": "1",
                "FAKE_ACTIVATION_PID_PATH": str(activation_pid_path),
            }
        )
        command = [
            sys.executable,
            str(self.launcher),
            "start",
            "--state-root",
            str(self.state_root),
            "--target-root",
            str(self.target_root),
            "--service-id",
            "test-prefill",
            "--role",
            "prefill",
            "--model-alias",
            "fake-model",
            "--runtime",
            "dashllm",
            "--uv-project",
            str(self.project),
            "--working-directory",
            str(self.root),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--",
            str(self.fake_server),
        ]
        starter: subprocess.Popen[str] | None = None
        try:
            with self.log_path.open("a", encoding="utf-8") as log_file:
                starter = subprocess.Popen(
                    command,
                    cwd=self.root,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self._wait_for_path(activation_pid_path)
                stopped = self._run_launcher(
                    "stop",
                    "--service-id",
                    "test-prefill",
                )
                self.assertEqual(stopped.returncode, 0, stopped.stdout)
                starter.wait(timeout=3)

            activation_pid = int(activation_pid_path.read_text(encoding="utf-8"))
            self.assertFalse(Path(f"/proc/{activation_pid}/stat").exists())
            self.assertFalse(
                (
                    self.state_root
                    / "services/test-prefill/request.json"
                ).exists()
            )
        finally:
            if starter is not None and starter.poll() is None:
                starter.kill()
                starter.wait(timeout=3)
            self._run_launcher(
                "stop",
                "--service-id",
                "test-prefill",
                check=False,
            )

    def test_start_inherits_environment_argv_and_stdout_then_stop_cleans(self) -> None:
        start = self._start()

        self.assertEqual(
            start.returncode,
            0,
            self.log_path.read_text(encoding="utf-8"),
        )
        self._wait_for_path(self.capture_path)
        capture = json.loads(self.capture_path.read_text(encoding="utf-8"))
        self.assertEqual(capture["argv"], ["--feature", "enabled"])
        self.assertEqual(capture["feature"], "visible")
        request_path = self.state_root / "services/test-prefill/request.json"
        request_text = request_path.read_text(encoding="utf-8")
        self.assertNotIn("FEATURE_WITHOUT_WHITELIST", request_text)
        self.assertNotIn("visible", request_text)
        target_path = self.target_root / "test-prefill.json"
        self.assertEqual(
            json.loads(target_path.read_text(encoding="utf-8")),
            [
                {
                    "targets": [f"127.0.0.1:{self.port}"],
                    "labels": {
                        "service_id": "test-prefill",
                        "role": "prefill",
                        "model_alias": "fake-model",
                    },
                }
            ],
        )
        self.assertEqual(request_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(target_path.stat().st_mode & 0o777, 0o644)

        status = self._run_launcher(
            "status",
            "--service-id",
            "test-prefill",
            "--json",
        )
        self.assertEqual(status.returncode, 0, status.stdout)
        self.assertEqual(json.loads(status.stdout)["state"], "running")

        stopped = self._run_launcher(
            "stop",
            "--service-id",
            "test-prefill",
        )
        self.assertEqual(stopped.returncode, 0, stopped.stdout)
        layout = launcher_state.RuntimeLayout.from_roots(
            self.state_root,
            self.target_root,
        )
        self.assertFalse(layout.request_path("test-prefill").exists())
        self.assertFalse(layout.runtime_path("test-prefill").exists())
        self.assertFalse(layout.target_path("test-prefill").exists())
        self.assertIn("fake-server-stdout", self.log_path.read_text(encoding="utf-8"))

    def test_killed_supervisor_reports_orphaned_and_stop_cleans(self) -> None:
        start = self._start()
        self.assertEqual(start.returncode, 0, self.log_path.read_text())
        runtime = self._runtime_payload()
        os.kill(runtime["supervisor"]["pid"], signal.SIGKILL)
        time.sleep(0.1)

        status = self._run_launcher(
            "status", "--service-id", "test-prefill", "--json"
        )
        self.assertEqual(status.returncode, 0, status.stdout)
        self.assertEqual(json.loads(status.stdout)["state"], "orphaned")

        stopped = self._run_launcher("stop", "--service-id", "test-prefill")
        self.assertEqual(stopped.returncode, 0, stopped.stdout)
        self.assertFalse(
            (self.state_root / "services/test-prefill/request.json").exists()
        )

    def test_leader_exit_does_not_leave_worker_process_group(self) -> None:
        self.environment["FAKE_FORK_WORKER"] = "1"
        start = self._start()
        self.assertEqual(start.returncode, 0, self.log_path.read_text())
        deadline = time.monotonic() + 3
        capture: dict[str, object] = {}
        while time.monotonic() < deadline:
            if self.capture_path.exists():
                capture = json.loads(self.capture_path.read_text(encoding="utf-8"))
                if "worker_pid" in capture:
                    break
            time.sleep(0.02)
        self.assertIn("worker_pid", capture)

        self._wait_for_absence(
            self.state_root / "services/test-prefill/request.json"
        )
        worker_pid = capture["worker_pid"]
        self.assertFalse(Path(f"/proc/{worker_pid}/stat").exists())

    def test_missing_target_root_rolls_back_spawned_server(self) -> None:
        self.target_root.rmdir()

        start = self._start()

        self.assertNotEqual(start.returncode, 0)
        self.assertFalse(
            (self.state_root / "services/test-prefill/request.json").exists()
        )
        if self.capture_path.exists():
            server_pid = json.loads(
                self.capture_path.read_text(encoding="utf-8")
            )["pid"]
            self.assertFalse(Path(f"/proc/{server_pid}/stat").exists())

    def test_start_reconciles_state_after_supervisor_and_server_are_killed(self) -> None:
        first_start = self._start()
        self.assertEqual(first_start.returncode, 0, self.log_path.read_text())
        runtime = self._runtime_payload()
        os.kill(runtime["supervisor"]["pid"], signal.SIGKILL)
        os.killpg(runtime["server_pgid"], signal.SIGKILL)
        time.sleep(0.1)

        second_start = self._start()

        self.assertEqual(
            second_start.returncode,
            0,
            self.log_path.read_text(encoding="utf-8"),
        )
        stopped = self._run_launcher("stop", "--service-id", "test-prefill")
        self.assertEqual(stopped.returncode, 0, stopped.stdout)

    def test_duplicate_start_is_rejected_without_replacing_service(self) -> None:
        first = self._start()
        self.assertEqual(first.returncode, 0, self.log_path.read_text())
        first_request_id = self._runtime_payload()["request_id"]

        second = self._start()

        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(self._runtime_payload()["request_id"], first_request_id)
        stopped = self._run_launcher("stop", "--service-id", "test-prefill")
        self.assertEqual(stopped.returncode, 0, stopped.stdout)

class VllmServiceLauncherTest(unittest.TestCase):
    def test_vendored_launcher_help_is_runnable(self) -> None:
        result = subprocess.run(
            [sys.executable, str(LAUNCHER), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("start", "status", "stop"):
            self.assertIn(command, result.stdout)
        self.assertNotIn("supervise", result.stdout)

    def test_installer_stages_complete_runtime_without_external_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            install_root = Path(temporary_directory) / "root"
            obsolete_module = (
                install_root
                / "usr/local/lib/vllm-service-launch/vllm_service_launch/identity.py"
            )
            obsolete_module.parent.mkdir(parents=True)
            obsolete_module.write_text("legacy", encoding="utf-8")
            result = subprocess.run(
                ["bash", str(INSTALLER), "--root", str(install_root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            installed_launcher = (
                install_root / "usr/local/bin/vllm-service-launch"
            )
            installed_library = (
                install_root
                / "usr/local/lib/vllm-service-launch/vllm_service_launch"
            )
            self.assertEqual(installed_launcher.stat().st_mode & 0o777, 0o755)
            self.assertEqual(
                {path.name for path in installed_library.glob("*.py")},
                {
                    "__init__.py",
                    "cli.py",
                    "endpoint.py",
                    "environment.py",
                    "process.py",
                    "schema.py",
                    "service.py",
                    "state.py",
                },
            )
            installed_files = {
                path.relative_to(install_root).as_posix()
                for path in install_root.rglob("*")
                if path.is_file()
            }
            expected_files = {"usr/local/bin/vllm-service-launch"}
            expected_files.update(
                {
                    f"usr/local/lib/vllm-service-launch/"
                    f"vllm_service_launch/{name}"
                    for name in {
                        "__init__.py",
                        "cli.py",
                        "endpoint.py",
                        "environment.py",
                        "process.py",
                        "schema.py",
                        "service.py",
                        "state.py",
                    }
                }
            )
            self.assertEqual(installed_files, expected_files)
            self.assertFalse((install_root / "etc").exists())
            self.assertFalse((install_root / "run").exists())

            installed_help = subprocess.run(
                [sys.executable, str(installed_launcher), "--help"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(installed_help.returncode, 0, installed_help.stderr)

    def test_installer_rejects_legacy_systemd_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            install_root = Path(temporary_directory) / "root"
            legacy_unit = install_root / "etc/systemd/system/vllm@.service"
            legacy_unit.parent.mkdir(parents=True)
            legacy_unit.write_text("legacy", encoding="utf-8")

            result = subprocess.run(
                ["bash", str(INSTALLER), "--root", str(install_root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("legacy", result.stderr.lower())
            self.assertFalse(
                (install_root / "usr/local/bin/vllm-service-launch").exists()
            )

if __name__ == "__main__":
    unittest.main()
