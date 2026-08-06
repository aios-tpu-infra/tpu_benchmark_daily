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
        for script_name, config, parallelism, batched_tokens in (
            ("start_dp_server.sh", "DP8", "DP=8, PCP=1, TP=1", 4096),
            ("start_pcp_server.sh", "PCP8", "DP=1, PCP=8, TP=1", 32768),
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
                self.assertIn(f"parallelism:             {parallelism}", result.stdout)
                self.assertIn(
                    f"max batched tokens:      {batched_tokens}", result.stdout
                )
                self.assertIn(
                    "compile sizes:           512,1024,2048,4096",
                    result.stdout,
                )

    def test_prefill_server_requires_a_known_config(self) -> None:
        missing = self.run_script("start_prefill_server.sh", "--test-only")
        unknown = self.run_script(
            "start_prefill_server.sh", "--config", "tp8", "--test-only"
        )

        self.assertEqual(missing.returncode, 2)
        self.assertIn("--config must be dp8 or pcp8", missing.stderr)
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("--config must be dp8 or pcp8", unknown.stderr)

    def test_prefill_server_config_matrix(self) -> None:
        dp = self.run_script(
            "start_prefill_server.sh", "--config=dp8", "--test-only"
        )
        pcp = self.run_script(
            "start_prefill_server.sh", "--config=pcp8", "--test-only"
        )

        self.assertEqual(dp.returncode, 0, dp.stderr)
        self.assertIn("max sequences:           64", dp.stdout)
        self.assertIn("compile sizes:           512,1024,2048,4096", dp.stdout)
        self.assertNotIn("long prefill threshold", dp.stdout)
        self.assertEqual(pcp.returncode, 0, pcp.stderr)
        self.assertIn("max sequences:           8", pcp.stdout)
        self.assertIn("compile sizes:           512,1024,2048,4096", pcp.stdout)
        self.assertIn("long prefill threshold:  32768", pcp.stdout)

    def test_prefill_wrapper_config_ignores_inherited_config(self) -> None:
        result = self.run_script(
            "start_dp_server.sh",
            "--test-only",
            environment={"BENCHMARK_CONFIG": "pcp8"},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TEST_ONLY: DP8 server startup skipped", result.stdout)
        self.assertIn("DP=8, PCP=1, TP=1", result.stdout)

    def test_prefill_server_validates_cache_reset_toggle(self) -> None:
        result = self.run_script(
            "start_prefill_server.sh",
            "--config",
            "pcp8",
            "--test-only",
            environment={"RESET_COMPILE_CACHE": "invalid"},
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("RESET_COMPILE_CACHE must be 0 or 1", result.stderr)

    def test_prefill_wrappers_only_select_the_config(self) -> None:
        for script_name, config in (
            ("start_dp_server.sh", "dp8"),
            ("start_pcp_server.sh", "pcp8"),
        ):
            with self.subTest(script_name=script_name):
                script = (
                    PROJECT_ROOT / "scripts" / script_name
                ).read_text(encoding="utf-8")
                self.assertIn('start_prefill_server.sh" --config', script)
                self.assertIn(f"--config {config}", script)
                self.assertLessEqual(len(script.splitlines()), 8)
                self.assertNotIn("vllm-service-launch", script)

    def test_all_server_configs_use_auto_sized_unified_pool(self) -> None:
        for script_name in (
            "start_dp_decode_server.sh",
            "start_prefill_server.sh",
        ):
            with self.subTest(script_name=script_name):
                script = (
                    PROJECT_ROOT / "scripts" / script_name
                ).read_text(encoding="utf-8")
                self.assertIn(
                    "export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1",
                    script,
                )
                self.assertNotIn("--block-size", script)

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

    def test_ttft_test_only_can_inject_one_failed_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_script(
                "bench_prefill_ttft.sh",
                "--test-only",
                temporary_directory,
                environment={
                    "BENCHMARK_CONFIG": "dp8",
                    "TTFT_TEST_ONLY_FAILED_LENGTHS": "258048",
                    "VENV_DIR": "/missing-test-only-venv",
                    "MODEL_DIR": "/missing-test-only-model",
                },
            )
            summary_path = (
                Path(temporary_directory)
                / "results"
                / "dp8"
                / "single_request_ttft"
                / "summary.json"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["failed_input_lengths"], [258048])
            failed = summary["results"][-1]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["completed"], 0)
            self.assertEqual(failed["failed"], 16)
            self.assertIsNone(failed["ttft_ms"])

    def test_speed_bench_mix_test_only_replays_default_concurrencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_script(
                "bench_speed_bench_mix.sh",
                "--test-only",
                temporary_directory,
                environment={
                    "VENV_DIR": "/missing-test-only-venv",
                    "MODEL_DIR": "/missing-test-only-model",
                },
            )
            summary_path = (
                Path(temporary_directory)
                / "results"
                / "dp8"
                / "speed_bench_mix"
                / "summary.json"
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["throughput"]["status"], "success")
            self.assertEqual(
                summary["throughput"]["requested_concurrencies"], [8, 64]
            )
            self.assertEqual(summary["serial_ttft"]["status"], "not-run")
            self.assertIn(
                "replayed SPEED-Bench fixture at concurrency 8", result.stdout
            )
            self.assertIn(
                "replayed SPEED-Bench fixture at concurrency 64", result.stdout
            )
            self.assertTrue((summary_path.parent / "throughput_c8.json").is_file())
            self.assertTrue((summary_path.parent / "throughput_c64.json").is_file())

    def test_speed_bench_mix_test_only_supports_pcp8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_script(
                "bench_speed_bench_mix.sh",
                "--test-only",
                temporary_directory,
                environment={
                    "BENCHMARK_CONFIG": "pcp8",
                    "VENV_DIR": "/missing-test-only-venv",
                    "MODEL_DIR": "/missing-test-only-model",
                },
            )
            summary_path = (
                Path(temporary_directory)
                / "results"
                / "pcp8"
                / "speed_bench_mix"
                / "summary.json"
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["benchmark"]["benchmark_config"], "pcp8")


if __name__ == "__main__":
    unittest.main()
