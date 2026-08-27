import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "aggregate_decode_runs.py"
SPEC = importlib.util.spec_from_file_location("aggregate_decode_runs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AGGREGATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATE)


def summary_payload(*, data_parallel_size: int = 4) -> dict[str, object]:
    return {
        "benchmark": {
            "concurrency": 256,
            "data_parallel_size": data_parallel_size,
            "tensor_parallel_size": 2,
            "prefill_tokens": 65536,
            "decode_tokens": 1024,
        },
        "results": [
            {
                "failed_requests": 0,
                "successful_requests": 256,
                "active_requests_max": 256,
                "window_count": 19,
                "usage_tokens": 262144,
                "end_to_end_throughput_tok_s": 5900.0,
                "first_token_skew_s": 42.0,
                "window_throughput_tok_s": {"p50": 6250.0},
                "peak_active_tpot_ms": {
                    "p50": 40.9,
                    "p90": 43.0,
                    "p99": 45.0,
                },
                "ttft_s": {"p50": 20.0, "p90": 40.0},
            }
        ],
    }


class AggregateDecodeRunsTest(unittest.TestCase):
    def write_summary(
        self, result_root: Path, payload: dict[str, object]
    ) -> None:
        run_dir = result_root / "run_1"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_accepts_dp4_tp2_decode_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_root = Path(temporary_directory)
            self.write_summary(result_root, summary_payload())

            aggregate = AGGREGATE.aggregate_result_root(result_root, runs=1)

        self.assertIn("DP4/TP2 C256/P65536/D1024", aggregate["protocol"])
        self.assertEqual(
            aggregate["runs"][0]["throughput_peak_active_p50_tok_s"],
            6250.0,
        )

    def test_rejects_wrong_data_parallel_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_root = Path(temporary_directory)
            self.write_summary(
                result_root, summary_payload(data_parallel_size=8)
            )

            with self.assertRaisesRegex(ValueError, "data_parallel_size=8"):
                AGGREGATE.aggregate_result_root(result_root, runs=1)


if __name__ == "__main__":
    unittest.main()
