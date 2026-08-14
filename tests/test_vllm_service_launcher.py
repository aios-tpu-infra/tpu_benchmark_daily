import configparser
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "vllm-service-launch"
LAUNCHER = VENDOR_ROOT / "bin" / "vllm-service-launch"
INSTALLER = PROJECT_ROOT / "scripts" / "install_vllm_service_launcher.sh"


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

    def test_installer_stages_complete_runtime_without_external_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            install_root = Path(temporary_directory) / "root"
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
                    "identity.py",
                    "schema.py",
                    "service.py",
                    "state.py",
                },
            )
            self.assertEqual(
                (
                    install_root / "etc/sudoers.d/vllm-service-launch"
                ).stat().st_mode
                & 0o777,
                0o440,
            )
            for relative_path in (
                "etc/systemd/system/vllm@.service",
                "usr/lib/tmpfiles.d/vllm-metrics-targets.conf",
                "run/vllm-services",
                "run/vllm-metrics-targets/targets",
            ):
                self.assertTrue((install_root / relative_path).exists())

            installed_help = subprocess.run(
                [sys.executable, str(installed_launcher), "--help"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(installed_help.returncode, 0, installed_help.stderr)

    def test_systemd_unit_preserves_launcher_lifecycle_contract(self) -> None:
        parser = configparser.ConfigParser(interpolation=None, strict=False)
        parser.read(
            VENDOR_ROOT / "systemd" / "vllm@.service",
            encoding="utf-8",
        )
        service = parser["Service"]

        self.assertEqual(service["Type"], "exec")
        self.assertEqual(service["KillMode"], "control-group")
        self.assertEqual(
            service["ExecStart"],
            "/usr/local/bin/vllm-service-launch run --service-id %i",
        )
        self.assertEqual(
            service["ExecStopPost"],
            "/usr/local/bin/vllm-service-launch cleanup --service-id %i",
        )


if __name__ == "__main__":
    unittest.main()
