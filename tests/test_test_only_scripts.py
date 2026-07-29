import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestOnlyScriptsTest(unittest.TestCase):
    def run_script(
        self,
        script_name: str,
        *arguments: str,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        process_environment = os.environ.copy()
        if environment:
            process_environment.update(environment)
        return subprocess.run(
            ["bash", str(PROJECT_ROOT / "scripts" / script_name), *arguments],
            cwd=PROJECT_ROOT,
            env=process_environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_server_start_scripts_skip_before_touching_runtime(self) -> None:
        for script_name, config in (
            ("start_dp_server.sh", "DP8"),
            ("start_pcp_server.sh", "PCP8"),
        ):
            with self.subTest(script_name=script_name):
                result = self.run_script(
                    script_name,
                    "--test-only",
                    environment={
                        "VENV_DIR": "/missing-test-only-venv",
                        "MODEL_DIR": "/missing-test-only-model",
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"TEST_ONLY: {config} server startup skipped.",
                    result.stdout,
                )
                self.assertIn("max model length:        262144", result.stdout)

    def test_throughput_test_only_accepts_flag_before_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_script(
                "bench_all.sh",
                "--test-only",
                temporary_directory,
                environment={
                    "BENCHMARK_CONFIG": "dp8",
                    "VENV_DIR": "/missing-test-only-venv",
                    "MODEL_DIR": "/missing-test-only-model",
                },
            )
            summary = (
                Path(temporary_directory) / "results" / "dp8" / "summary.json"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(summary.is_file())
            self.assertIn("skipped vLLM benchmark requests", result.stdout)

    def test_ttft_test_only_replays_raw_output_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_script(
                "bench_prefill_ttft.sh",
                "--test-only",
                temporary_directory,
                environment={
                    "BENCHMARK_CONFIG": "dp8",
                    "VENV_DIR": "/missing-test-only-venv",
                    "MODEL_DIR": "/missing-test-only-model",
                },
            )
            result_dir = (
                Path(temporary_directory)
                / "results"
                / "dp8"
                / "single_request_ttft"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((result_dir / "summary.json").is_file())
            self.assertEqual(len(list(result_dir.glob("vllm_*.json"))), 6)
            summary = json.loads(
                (result_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["benchmark"]["samples_per_length"], 16)
            self.assertEqual(
                {item["completed"] for item in summary["results"]},
                {16},
            )
            self.assertEqual(
                {len(item["raw_ttft_ms"]) for item in summary["results"]},
                {16},
            )


if __name__ == "__main__":
    unittest.main()
