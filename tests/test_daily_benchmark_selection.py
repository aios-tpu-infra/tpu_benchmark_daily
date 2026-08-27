import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_RUNNER = PROJECT_ROOT / "scripts" / "daily_benchmark.sh"


class DailyBenchmarkSelectionTest(unittest.TestCase):
    def run_cli(
        self,
        *arguments: str,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        process_environment = os.environ.copy()
        if environment:
            process_environment.update(environment)
        return subprocess.run(
            ["bash", str(DAILY_RUNNER), *arguments],
            cwd=PROJECT_ROOT,
            env=process_environment,
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
        self.assertIn("--prefill-mode MODE", result.stdout)
        self.assertIn("all, throughput, or ttft", result.stdout)
        self.assertIn("--prefill-workload WORKLOAD", result.stdout)
        self.assertIn("all, synthetic, or", result.stdout)
        self.assertIn("--test-only", result.stdout)
        self.assertIn("--commit COMMIT", result.stdout)
        self.assertIn("--torchtpu-commit is an alias", result.stdout)

    def test_decode_workflow_uses_latest_main_by_default(self) -> None:
        script = DAILY_RUNNER.read_text(encoding="utf-8")

        self.assertIn("TORCHTPU_COMMIT=\n", script)
        self.assertIn("latest origin/main", script)
        self.assertIn("dp4_tp2_decode_c256", script)
        self.assertIn("--data-parallel-size 4", script)
        self.assertIn("--tensor-parallel-size 2", script)
        self.assertIn("--window-seconds 1", script)
        self.assertIn("--step-seconds 0.1", script)
        self.assertIn("SHARED_MODEL_DIR=", script)
        self.assertIn("model.safetensors.index.json", script)

    def test_only_requires_a_value(self) -> None:
        result = self.run_cli("--only")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--only requires a benchmark name", result.stderr)

    def test_only_rejects_an_unknown_benchmark(self) -> None:
        result = self.run_cli("--only", "unknown")

        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid --only benchmark 'unknown'", result.stderr)

    def test_prefill_mode_requires_a_value(self) -> None:
        result = self.run_cli("--prefill-mode")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--prefill-mode requires a mode", result.stderr)

    def test_prefill_mode_rejects_an_unknown_mode(self) -> None:
        result = self.run_cli("--prefill-mode", "latency")

        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid --prefill-mode 'latency'", result.stderr)

    def test_prefill_mode_rejects_decode_only_selection(self) -> None:
        result = self.run_cli(
            "--only", "dp-decode", "--prefill-mode", "throughput"
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "--prefill-mode requires a selected DP/PCP prefill benchmark",
            result.stderr,
        )

    def test_prefill_workload_rejects_decode_only_selection(self) -> None:
        result = self.run_cli(
            "--only", "dp-decode", "--prefill-workload", "synthetic"
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "--prefill-workload requires a selected DP/PCP prefill benchmark",
            result.stderr,
        )

    def test_speed_bench_workload_supports_pcp_only_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            result = self.run_cli(
                "--test-only",
                "--only",
                "pcp-prefill",
                "--prefill-workload",
                "speed-bench",
                environment={
                    "MACHINE_IP": "127.0.0.1",
                    "STATE_DIR": str(state_dir),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("TEST_ONLY: PCP8 server startup skipped", result.stdout)
            self.assertIn(
                "replayed SPEED-Bench fixture at concurrency 64", result.stdout
            )
            self.assertNotIn("replayed fixed throughput summary", result.stdout)
            summary_path = next(
                state_dir.glob(
                    "test-only-preview/*/runs/*/results/pcp8/"
                    "speed_bench_mix/summary.json"
                )
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["benchmark"]["benchmark_config"], "pcp8"
            )

    def test_test_only_reports_launcher_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            result = self.run_cli(
                "--test-only",
                "--only",
                "dp-prefill",
                environment={
                    "MACHINE_IP": "127.0.0.1",
                    "STATE_DIR": str(state_dir),
                    "VLLM_SERVICE_LAUNCH": "/usr/local/bin/vllm-service-launch",
                    "VLLM_SERVICE_STATE_ROOT": "/project/.state/vllm-service-launch",
                    "VLLM_SERVICE_TARGET_ROOT": "/run/vllm-metrics-targets/targets",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "vLLM service launcher: /usr/local/bin/vllm-service-launch",
                result.stdout,
            )
            self.assertIn(
                "vLLM service state: /project/.state/vllm-service-launch",
                result.stdout,
            )
            self.assertIn(
                "vLLM metrics targets: /run/vllm-metrics-targets/targets",
                result.stdout,
            )

    def test_throughput_only_replays_no_ttft_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            result = self.run_cli(
                "--test-only",
                "--only",
                "dp-prefill",
                "--prefill-mode",
                "throughput",
                environment={
                    "MACHINE_IP": "127.0.0.1",
                    "STATE_DIR": str(state_dir),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("replayed fixed throughput summary", result.stdout)
            self.assertNotIn("single-request TTFT benchmark", result.stdout)
            latest_path = next(
                state_dir.glob("test-only-preview/*/project/reports/latest.json")
            )
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            dp8 = latest["benchmarks"]["dp8"]
            self.assertEqual(dp8["status"], "success")
            self.assertEqual(dp8["prefill_ttft_status"], "not-run")

    def test_ttft_only_replays_no_throughput_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            result = self.run_cli(
                "--test-only",
                "--only",
                "pcp-prefill",
                "--prefill-mode=ttft",
                environment={
                    "MACHINE_IP": "127.0.0.1",
                    "STATE_DIR": str(state_dir),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Single-request TTFT benchmark", result.stdout)
            self.assertNotIn("replayed fixed throughput summary", result.stdout)
            latest_path = next(
                state_dir.glob("test-only-preview/*/project/reports/latest.json")
            )
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            pcp8 = latest["benchmarks"]["pcp8"]
            self.assertEqual(pcp8["status"], "not-run")
            self.assertEqual(pcp8["prefill_ttft_status"], "success")
            self.assertIn(
                "replayed SPEED-Bench fixture at concurrency 8", result.stdout
            )

    def test_speed_bench_only_replays_no_synthetic_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            result = self.run_cli(
                "--test-only",
                "--only",
                "dp-prefill",
                "--prefill-workload",
                "speed-bench",
                environment={
                    "MACHINE_IP": "127.0.0.1",
                    "STATE_DIR": str(state_dir),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "replayed SPEED-Bench fixture at concurrency 8", result.stdout
            )
            self.assertIn(
                "replayed SPEED-Bench fixture at concurrency 64", result.stdout
            )
            self.assertNotIn("replayed fixed throughput summary", result.stdout)
            speed_latest_path = next(
                state_dir.glob(
                    "test-only-preview/*/project/reports/speed_bench_latest.json"
                )
            )
            speed_latest = json.loads(
                speed_latest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(speed_latest["benchmark"]["status"], "success")
            self.assertEqual(
                [
                    result["concurrency"]
                    for result in speed_latest["benchmark"][
                        "concurrency_results"
                    ]
                ],
                [8, 64],
            )
            fixed_latest_paths = list(
                state_dir.glob("test-only-preview/*/project/reports/latest.json")
            )
            self.assertEqual(len(fixed_latest_paths), 1)
            fixed_latest = json.loads(
                fixed_latest_paths[0].read_text(encoding="utf-8")
            )
            self.assertNotEqual(
                fixed_latest["benchmarks"]["dp8"]["run_id"],
                speed_latest["benchmark"]["run_id"],
            )

    def test_speed_bench_all_selection_records_dp8_and_pcp8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            result = self.run_cli(
                "--test-only",
                "--prefill-workload",
                "speed-bench",
                environment={
                    "MACHINE_IP": "127.0.0.1",
                    "STATE_DIR": str(state_dir),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summaries = list(
                state_dir.glob(
                    "test-only-preview/*/runs/*/results/*/"
                    "speed_bench_mix/summary.json"
                )
            )
            configs = {
                json.loads(path.read_text(encoding="utf-8"))["benchmark"][
                    "benchmark_config"
                ]
                for path in summaries
            }
            self.assertEqual(configs, {"dp8", "pcp8"})
            history_path = next(
                state_dir.glob(
                    "test-only-preview/*/project/reports/"
                    "speed_bench_history.json"
                )
            )
            history = json.loads(history_path.read_text(encoding="utf-8"))
            fixture_configs = {
                run["benchmark_config"]
                for run in history["runs"]
                if run["torchtpu_vllm_revision"] == "test-only-fixture"
            }
            self.assertEqual(fixture_configs, {"dp8", "pcp8"})

    def test_commit_requires_a_value(self) -> None:
        result = self.run_cli("--commit")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--commit requires a Git commit", result.stderr)

    def test_commit_rejects_test_only_mode(self) -> None:
        result = self.run_cli(
            "--test-only",
            "--commit",
            "0123456789abcdef0123456789abcdef01234567",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--commit cannot be used with --test-only", result.stderr)

    def test_commit_must_be_hexadecimal(self) -> None:
        result = self.run_cli("--commit", "main")

        self.assertEqual(result.returncode, 2)
        self.assertIn("hexadecimal Git commit ID", result.stderr)

    def test_empty_commit_is_rejected(self) -> None:
        result = self.run_cli("--commit=")

        self.assertEqual(result.returncode, 2)
        self.assertIn("hexadecimal Git commit ID", result.stderr)

    def test_test_only_rejects_decode_only_selection(self) -> None:
        result = self.run_cli("--test-only", "--only", "dp-decode")

        self.assertEqual(result.returncode, 2)
        self.assertIn("covers DP/PCP prefill benchmarks only", result.stderr)

    def test_script_records_each_group_before_one_final_publish(self) -> None:
        script = DAILY_RUNNER.read_text(encoding="utf-8")

        self.assertIn("if (( RUN_DP_DECODE )); then", script)
        self.assertIn("if (( RUN_DP_PREFILL )); then", script)
        self.assertIn(
            "if (( RUN_PCP_PREFILL )); then",
            script,
        )
        self.assertIn("DP_DECODE_STATUS=failed", script)
        self.assertIn("DP_PREFILL_STATUS=failed", script)
        self.assertIn("PCP_PREFILL_STATUS=failed", script)
        self.assertIn("PCP_SPEED_BENCH_STATUS=failed", script)
        self.assertIn("record_dp_report", script)
        self.assertIn("record_pcp_report", script)
        self.assertIn("UPDATE_REPORTS=0", script)
        self.assertIn(
            "if (( PUBLISH_REPORTS && REPORT_GENERATED )); then",
            script,
        )
        self.assertIn("if (( BENCHMARK_FAILURES )); then", script)
        self.assertLess(
            script.index("if (( PUBLISH_REPORTS && REPORT_GENERATED )); then"),
            script.index("if (( BENCHMARK_FAILURES )); then"),
        )

    def test_test_only_branch_precedes_environment_update_and_server_start(
        self,
    ) -> None:
        script = DAILY_RUNNER.read_text(encoding="utf-8")

        branch = script.index("if (( TEST_ONLY )); then")
        environment_update = script.index('"$SCRIPT_DIR/update_environment.sh"')
        normal_server_start = script.index("if (( RUN_DP_DECODE )); then")
        self.assertLess(branch, environment_update)
        self.assertLess(branch, normal_server_start)
        self.assertIn("TEST_ONLY=1 UPDATE_REPORTS=0", script)
        self.assertIn('"$SCRIPT_DIR/bench_prefill_ttft.sh"', script)
        self.assertIn("TEST_ONLY preview README", script)

    def test_requested_commit_is_forwarded_to_environment_update(self) -> None:
        script = DAILY_RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            'environment_update_args+=(--commit "$TORCHTPU_COMMIT")',
            script,
        )
        self.assertIn(
            '"$SCRIPT_DIR/update_environment.sh" "${environment_update_args[@]}"',
            script,
        )


if __name__ == "__main__":
    unittest.main()
