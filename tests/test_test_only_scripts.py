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
                self.assertIn("parallel precompile:     1", result.stdout)

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
        self.assertIn("skip padded MoE tokens:  1", dp.stdout)
        self.assertIn("MoE collection chunk size: 16384", dp.stdout)
        self.assertIn("fused EP MoE kernel:      1", dp.stdout)
        self.assertIn("fused EP minimum tokens:  1024", dp.stdout)
        self.assertIn("sharded routing plan:     1", dp.stdout)
        self.assertIn("split activation gather:  1", dp.stdout)
        self.assertNotIn("long prefill threshold", dp.stdout)
        self.assertEqual(pcp.returncode, 0, pcp.stderr)
        self.assertIn("max sequences:           64", pcp.stdout)
        self.assertIn("compile sizes:           512,1024,2048,4096", pcp.stdout)
        self.assertIn("skip padded MoE tokens:  0", pcp.stdout)
        self.assertIn("MoE collection chunk size: 16384", pcp.stdout)
        self.assertIn("fused EP MoE kernel:      1", pcp.stdout)
        self.assertIn("fused EP minimum tokens:  1024", pcp.stdout)
        self.assertIn("sharded routing plan:     1", pcp.stdout)
        self.assertIn("split activation gather:  1", pcp.stdout)
        self.assertIn("long prefill threshold:  32768", pcp.stdout)

    def test_prefill_server_supports_fused_ep_moe_overrides(self) -> None:
        result = self.run_script(
            "start_prefill_server.sh",
            "--config",
            "dp8",
            "--test-only",
            environment={
                "USE_MOE_FUSED_EP_KERNEL": "false",
                "MOE_FUSED_EP_KERNEL_MIN_TOKENS": "2048",
                "MOE_FUSED_EP_V2_SHARDED_PLAN": "false",
                "MOE_FUSED_EP_V2_SPLIT_ACTIVATION_GATHER": "false",
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fused EP MoE kernel:      0", result.stdout)
        self.assertIn("fused EP minimum tokens:  2048", result.stdout)
        self.assertIn("sharded routing plan:     0", result.stdout)
        self.assertIn("split activation gather:  0", result.stdout)

    def test_prefill_server_validates_fused_ep_moe_settings(self) -> None:
        invalid_settings = (
            ("USE_MOE_FUSED_EP_KERNEL", "invalid", "must be a boolean"),
            (
                "MOE_FUSED_EP_KERNEL_MIN_TOKENS",
                "invalid",
                "must be a non-negative integer",
            ),
            ("MOE_FUSED_EP_V2_SHARDED_PLAN", "invalid", "must be a boolean"),
            (
                "MOE_FUSED_EP_V2_SPLIT_ACTIVATION_GATHER",
                "invalid",
                "must be a boolean",
            ),
        )
        for name, value, message in invalid_settings:
            with self.subTest(name=name):
                result = self.run_script(
                    "start_prefill_server.sh",
                    "--config",
                    "dp8",
                    "--test-only",
                    environment={name: value},
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(f"{name} {message}", result.stderr)

    def test_prefill_server_supports_moe_chunk_size_override(self) -> None:
        result = self.run_script(
            "start_prefill_server.sh",
            "--config",
            "dp8",
            "--test-only",
            environment={"TPU_MOE_COLLECTION_CHUNK_SIZE": "8192"},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MoE collection chunk size: 8192", result.stdout)

    def test_prefill_server_validates_moe_chunk_size(self) -> None:
        result = self.run_script(
            "start_prefill_server.sh",
            "--config",
            "dp8",
            "--test-only",
            environment={"TPU_MOE_COLLECTION_CHUNK_SIZE": "invalid"},
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "TPU_MOE_COLLECTION_CHUNK_SIZE must be a non-negative integer",
            result.stderr,
        )

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

    def test_decode_server_validates_cache_reset_toggle(self) -> None:
        result = self.run_script(
            "start_dp_decode_server.sh",
            environment={"RESET_COMPILE_CACHE": "invalid"},
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("RESET_COMPILE_CACHE must be 0 or 1", result.stderr)

    def test_all_server_configs_support_cache_retention(self) -> None:
        for script_name in (
            "start_dp_decode_server.sh",
            "start_prefill_server.sh",
        ):
            with self.subTest(script_name=script_name):
                script = (
                    PROJECT_ROOT / "scripts" / script_name
                ).read_text(encoding="utf-8")
                self.assertIn(
                    'RESET_COMPILE_CACHE="${RESET_COMPILE_CACHE:-1}"',
                    script,
                )
                self.assertIn("if (( RESET_COMPILE_CACHE )); then", script)
                self.assertIn("COMPILE_CACHE_ACTION=cleared", script)
                self.assertIn("COMPILE_CACHE_ACTION=retained", script)

    def test_prefill_server_validates_parallel_precompile_toggle(self) -> None:
        disabled = self.run_script(
            "start_prefill_server.sh",
            "--config",
            "dp8",
            "--test-only",
            environment={"TPU_PARALLEL_PRECOMPILE": "false"},
        )
        invalid = self.run_script(
            "start_prefill_server.sh",
            "--config",
            "dp8",
            "--test-only",
            environment={"TPU_PARALLEL_PRECOMPILE": "invalid"},
        )

        self.assertEqual(disabled.returncode, 0, disabled.stderr)
        self.assertIn("parallel precompile:     0", disabled.stdout)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn(
            "TPU_PARALLEL_PRECOMPILE must be a boolean",
            invalid.stderr,
        )

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

    def test_all_server_configs_use_unified_pool(self) -> None:
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
        decode_script = (
            PROJECT_ROOT / "scripts" / "start_dp_decode_server.sh"
        ).read_text(encoding="utf-8")
        prefill_script = (
            PROJECT_ROOT / "scripts" / "start_prefill_server.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('--block-size "$BLOCK_SIZE"', decode_script)
        self.assertNotIn("--block-size", prefill_script)

    def test_all_server_configs_enable_parallel_precompile(self) -> None:
        for script_name in (
            "start_dp_decode_server.sh",
            "start_prefill_server.sh",
        ):
            with self.subTest(script_name=script_name):
                script = (
                    PROJECT_ROOT / "scripts" / script_name
                ).read_text(encoding="utf-8")
                self.assertIn(
                    'TPU_PARALLEL_PRECOMPILE="${TPU_PARALLEL_PRECOMPILE:-1}"',
                    script,
                )
                self.assertIn("export TPU_PARALLEL_PRECOMPILE", script)

    def test_server_scripts_preserve_vllm_args_after_separator(self) -> None:
        cases = (
            ("start_dp_decode_server.sh", ("--test-only",)),
            (
                "start_prefill_server.sh",
                ("--config", "dp8", "--test-only"),
            ),
        )
        for script_name, script_arguments in cases:
            with self.subTest(script_name=script_name):
                result = self.run_script(
                    script_name,
                    *script_arguments,
                    "--",
                    "--enable-feature-x",
                    "value with space",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    "extra vLLM args: --enable-feature-x value\\ with\\ space",
                    result.stdout,
                )
                self.assertNotIn(
                    "extra vLLM args: -- --enable-feature-x",
                    result.stdout,
                )

    def test_prefill_treats_script_flags_after_separator_as_vllm_args(self) -> None:
        result = self.run_script(
            "start_prefill_server.sh",
            "--config",
            "dp8",
            "--test-only",
            "--",
            "--config",
            "server-owned",
            "--test-only",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "extra vLLM args: --config server-owned --test-only",
            result.stdout,
        )

    def test_all_server_configs_set_premapped_buffer_size(self) -> None:
        for script_name in (
            "start_dp_decode_server.sh",
            "start_prefill_server.sh",
        ):
            with self.subTest(script_name=script_name):
                script = (
                    PROJECT_ROOT / "scripts" / script_name
                ).read_text(encoding="utf-8")
                self.assertIn(
                    'TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-17179869184}"',
                    script,
                )
                self.assertIn("export TPU_PREMAPPED_BUFFER_SIZE", script)

    def test_decode_server_uses_validated_dp4_tp2_configuration(self) -> None:
        script = (
            PROJECT_ROOT / "scripts" / "start_dp_decode_server.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"',
            script,
        )
        self.assertIn(
            'COMPILE_SIZES="${COMPILE_SIZES:-8,16,32,64,72,4096}"',
            script,
        )
        self.assertIn('BLOCK_SIZE="${BLOCK_SIZE:-2304}"', script)
        self.assertIn('GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"', script)
        self.assertIn("--tensor-parallel-size 2", script)
        self.assertIn("--data-parallel-size 4", script)
        self.assertIn("--data-parallel-size-local 4", script)
        self.assertIn('--mamba-ssm-cache-dtype "$MAMBA_SSM_CACHE_DTYPE"', script)
        self.assertIn("export USE_BATCHED_RPA_SEQ_ON_LANE=1", script)
        self.assertIn("export TPU_MOE_OWNER_OUTPUT_MODE", script)
        self.assertIn("--no-enable-prefix-caching", script)
        self.assertIn("SHARED_MODEL_DIR=", script)
        self.assertIn("local model weights are incomplete", script)

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
            self.assertIsNone(summary["benchmark"]["samples_per_length"])
            self.assertEqual(
                summary["benchmark"]["samples_by_input_length"],
                {
                    "8192": 16,
                    "16384": 16,
                    "32768": 16,
                    "65536": 4,
                    "131072": 4,
                    "258048": 4,
                },
            )
            self.assertEqual(
                [item["completed"] for item in summary["results"]],
                [16, 16, 16, 4, 4, 4],
            )
            self.assertEqual(
                [len(item["raw_ttft_ms"]) for item in summary["results"]],
                [16, 16, 16, 4, 4, 4],
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
            self.assertEqual(failed["failed"], 4)
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
