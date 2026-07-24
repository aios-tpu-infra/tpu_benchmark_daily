from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECODE_SERVER = PROJECT_ROOT / "scripts" / "start_dp_decode_server.sh"
DAILY_RUNNER = PROJECT_ROOT / "scripts" / "daily_benchmark.sh"
PREFILL_BENCHMARK = PROJECT_ROOT / "scripts" / "bench_all.sh"


class HighConcurrencyDecodeServiceContractTest(unittest.TestCase):
    def test_decode_service_matches_c256_scheduler_and_cache_contract(self) -> None:
        script = DECODE_SERVER.read_text(encoding="utf-8")

        required_fragments = [
            'MODEL_DIR="${MODEL_DIR:-/mnt/data/models/Qwen3.5-397B-A17B-FP8}"',
            'MAX_MODEL_LEN="${MAX_MODEL_LEN:-66560}"',
            'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4384}"',
            'MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"',
            'BLOCK_SIZE="${BLOCK_SIZE:-4352}"',
            'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"',
            'COMPILE_SIZES="${COMPILE_SIZES:-8,16,32,4352,4384}"',
            'DECODE_BARRIER_PATCH_DIR="$SCRIPT_DIR/decode_barrier_patch"',
            "export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1",
            "unset TPU_VLLM_KV_CACHE_ALIAS_FALLBACK",
            "export TPU_KV_CACHE_HEADROOM_MIB=6144",
            "export USE_BATCHED_RPA_KERNEL=1",
            "export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd",
            "--data-parallel-size 8",
            "--data-parallel-size-local 8",
            "--mamba-cache-mode align",
            "--no-enable-prefix-caching",
            "--async-scheduling",
            "--kv-cache-dtype fp8",
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, script)

        self.assertNotIn("--load-format dummy", script)
        self.assertNotIn("--return-tokens-as-token-ids", script)
        self.assertNotIn(
            "export VLLM_MOE_ROUTING_SIMULATION_STRATEGY", script
        )

    def test_daily_runner_uses_three_separate_services_and_c256_workload(
        self,
    ) -> None:
        script = DAILY_RUNNER.read_text(encoding="utf-8")

        decode_start = script.index(
            "start_server dp8_decode_c256 "
            '"$SCRIPT_DIR/start_dp_decode_server.sh" "$DECODE_MODEL_DIR"'
        )
        dp_prefill_start = script.index(
            'start_server dp8 "$SCRIPT_DIR/start_dp_server.sh" "$MODEL_DIR"'
        )
        pcp_prefill_start = script.index(
            'start_server pcp8 "$SCRIPT_DIR/start_pcp_server.sh" "$MODEL_DIR"'
        )
        self.assertLess(decode_start, dp_prefill_start)
        self.assertLess(dp_prefill_start, pcp_prefill_start)
        self.assertIn("--concurrency 256", script)
        self.assertIn("--data-parallel-size 8", script)
        self.assertIn("--prefill-tokens 65536", script)
        self.assertIn("--decode-tokens 1024", script)
        self.assertIn("--rounds 3", script)
        self.assertIn("--window-seconds 10", script)
        self.assertIn("--step-seconds 1", script)
        benchmark_script = (
            PROJECT_ROOT / "scripts" / "bench_decode_sliding_window.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"X-AIOS-DECODE-BARRIER": barrier_group', benchmark_script)
        self.assertIn('"continuous_usage_stats": True', benchmark_script)
        self.assertNotIn('"return_token_ids": True', benchmark_script)
        self.assertNotIn("--concurrency 16", script)
        self.assertNotIn("--decode-tokens 4096", script)

    def test_decode_results_are_a_peer_of_prefill_results(self) -> None:
        daily_runner = DAILY_RUNNER.read_text(encoding="utf-8")
        prefill_benchmark = PREFILL_BENCHMARK.read_text(encoding="utf-8")

        self.assertIn(
            'local result_dir="$RUN_DIR/results/dp8_decode_c256"',
            daily_runner,
        )
        self.assertIn('--output-dir "$result_dir"', daily_runner)
        self.assertNotIn(
            '--output-dir "$result_dir/decode_sliding_window"',
            daily_runner,
        )
        self.assertIn(
            'decode_summary="$RUN_DIR/results/dp8_decode_c256/summary.json"',
            prefill_benchmark,
        )
        self.assertIn(
            'report_args+=(--decode-summary "$decode_summary")',
            prefill_benchmark,
        )


if __name__ == "__main__":
    unittest.main()
