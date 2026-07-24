from pathlib import Path
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_RUNNER = PROJECT_ROOT / "scripts" / "daily_benchmark.sh"


class DailyBenchmarkSelectionTest(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(DAILY_RUNNER), *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_help_documents_each_selective_benchmark(self) -> None:
        result = self.run_cli("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--only BENCHMARK", result.stdout)
        self.assertIn("dp-decode", result.stdout)
        self.assertIn("dp-prefill", result.stdout)
        self.assertIn("pcp-prefill", result.stdout)

    def test_only_requires_a_value(self) -> None:
        result = self.run_cli("--only")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--only requires a benchmark name", result.stderr)

    def test_only_rejects_an_unknown_benchmark(self) -> None:
        result = self.run_cli("--only", "unknown")

        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid --only benchmark 'unknown'", result.stderr)

    def test_script_guards_each_benchmark_group(self) -> None:
        script = DAILY_RUNNER.read_text(encoding="utf-8")

        self.assertIn("if (( RUN_DP_DECODE )); then", script)
        self.assertIn("if (( RUN_DP_PREFILL )); then", script)
        self.assertIn("if (( RUN_PCP_PREFILL )); then", script)
        self.assertIn(
            "if (( PUBLISH_REPORTS && REPORT_GENERATED )); then",
            script,
        )


if __name__ == "__main__":
    unittest.main()
