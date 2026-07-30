from pathlib import Path
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATE_ENVIRONMENT = PROJECT_ROOT / "scripts" / "update_environment.sh"


class UpdateEnvironmentCliTest(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(UPDATE_ENVIRONMENT), *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_help_documents_commit_selection(self) -> None:
        result = self.run_cli("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--commit COMMIT", result.stdout)
        self.assertIn("requested commit", result.stdout)
        self.assertIn("latest origin/main", result.stdout)

    def test_commit_requires_a_value(self) -> None:
        result = self.run_cli("--commit")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--commit requires a Git commit", result.stderr)

    def test_commit_must_be_hexadecimal(self) -> None:
        result = self.run_cli("--commit", "main")

        self.assertEqual(result.returncode, 2)
        self.assertIn("hexadecimal Git commit ID", result.stderr)

    def test_empty_commit_is_rejected(self) -> None:
        result = self.run_cli("--commit=")

        self.assertEqual(result.returncode, 2)
        self.assertIn("hexadecimal Git commit ID", result.stderr)

    def test_commit_cannot_be_combined_with_no_source_update(self) -> None:
        result = self.run_cli(
            "--no-source-update",
            "--commit",
            "0123456789abcdef0123456789abcdef01234567",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "--commit and --no-source-update cannot be used together",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
