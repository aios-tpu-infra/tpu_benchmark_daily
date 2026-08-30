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
                "peak_window_throughput_tok_s": 6398.0,
                "peak_window_active_requests": 252,
                "peak_window_tpot_ms": {
                    "p50": 40.8,
                    "p90": 49.5,
                    "p99": 55.3,
                },
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

    def write_uniform_raw_requests(self, result_root: Path) -> None:
        raw_path = result_root / "run_1" / "raw_requests.jsonl"
        raw_path.write_text(
            "".join(
                json.dumps(
                    {
                        "round": 1,
                        "request_id": request_id,
                        "token_times_after_batch_start_s": [0.0, 1.0, 2.0],
                    }
                )
                + "\n"
                for request_id in range(256)
            ),
            encoding="utf-8",
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
        self.assertEqual(
            aggregate["runs"][0]["throughput_peak_window_tok_s"],
            6398.0,
        )
        self.assertEqual(
            aggregate["runs"][0]["peak_window_active_requests"],
            252,
        )
        self.assertEqual(
            aggregate["runs"][0]["tpot_peak_window_p50_ms"],
            40.8,
        )

    def test_rejects_wrong_data_parallel_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_root = Path(temporary_directory)
            self.write_summary(
                result_root, summary_payload(data_parallel_size=8)
            )

            with self.assertRaisesRegex(ValueError, "data_parallel_size=8"):
                AGGREGATE.aggregate_result_root(result_root, runs=1)

    def test_replays_peak_window_from_schema7_raw_requests(self) -> None:
        payload = summary_payload()
        benchmark = payload["benchmark"]
        assert isinstance(benchmark, dict)
        benchmark.update({"window_s": 1.0, "step_s": 0.1})
        result = payload["results"][0]
        assert isinstance(result, dict)
        result.pop("peak_window_throughput_tok_s")
        result.pop("peak_window_active_requests")
        result.pop("peak_window_tpot_ms")

        with tempfile.TemporaryDirectory() as temporary_directory:
            result_root = Path(temporary_directory)
            self.write_summary(result_root, payload)
            self.write_uniform_raw_requests(result_root)

            aggregate = AGGREGATE.aggregate_result_root(result_root, runs=1)

        run = aggregate["runs"][0]
        self.assertEqual(run["throughput_peak_window_tok_s"], 256.0)
        self.assertEqual(run["peak_window_active_requests"], 256)
        self.assertEqual(run["tpot_peak_window_p50_ms"], 1000.0)


if __name__ == "__main__":
    unittest.main()
