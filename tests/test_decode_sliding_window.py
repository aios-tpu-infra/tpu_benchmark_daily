import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = PROJECT_ROOT / "scripts" / "bench_decode_sliding_window.py"
SPEC = importlib.util.spec_from_file_location(
    "bench_decode_sliding_window", BENCHMARK_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def request_result(
    request_id: int, decode_start_s: int, decode_end_s: int
) -> object:
    return BENCHMARK.RequestResult(
        request_id=request_id,
        started_s=0.0,
        finished_s=float(decode_end_s),
        token_times_s=[
            float(decode_start_s - 1),
            *[float(value) for value in range(decode_start_s, decode_end_s + 1)],
        ],
        error=None,
    )


class DecodeSlidingWindowTest(unittest.TestCase):
    def test_cli_defaults_to_dp4_tp2(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "bench_decode_sliding_window.py",
                "--output-dir",
                "output",
                "--tokenizer-dir",
                "model",
            ],
        ):
            args = BENCHMARK.parse_args()

        self.assertEqual(args.data_parallel_size, 4)
        self.assertEqual(args.tensor_parallel_size, 2)
        self.assertEqual(args.min_peak_active_fraction, 0.9)

    def analyze(
        self, results: list[object], window_s: float = 8.0
    ) -> object:
        return BENCHMARK.analyze_round(
            round_index=1,
            results=results,
            concurrency=len(results),
            window_s=window_s,
            step_s=1.0,
            min_full_overlap_s=0.0,
            min_full_overlap_tokens=0,
            min_peak_active_fraction=0.5,
        )

    def test_uses_actual_peak_sustained_concurrency(self) -> None:
        analysis = self.analyze(
            [
                request_result(0, 1, 21),
                request_result(1, 1, 21),
                request_result(2, 5, 15),
                request_result(3, 8, 13),
            ]
        )

        self.assertTrue(analysis.summary["valid"])
        self.assertEqual(analysis.summary["active_requests_max"], 3)
        self.assertFalse(
            analysis.summary["timeline_valid_full_concurrency_decode"]
        )
        self.assertEqual(analysis.summary["full_concurrency_window_count"], 0)
        self.assertTrue(analysis.windows)
        self.assertTrue(
            all(window["active_requests"] == 3 for window in analysis.windows)
        )

    def test_uses_full_concurrency_when_it_is_sustained(self) -> None:
        analysis = self.analyze(
            [
                request_result(0, 1, 21),
                request_result(1, 1, 21),
                request_result(2, 5, 15),
                request_result(3, 5, 15),
            ]
        )

        self.assertTrue(analysis.summary["valid"])
        self.assertEqual(analysis.summary["active_requests_max"], 4)
        self.assertTrue(
            analysis.summary["timeline_valid_full_concurrency_decode"]
        )
        self.assertGreater(
            analysis.summary["full_concurrency_window_count"], 0
        )

    def test_disjoint_requests_do_not_report_negative_full_overlap(self) -> None:
        analysis = self.analyze(
            [
                request_result(0, 1, 10),
                request_result(1, 11, 20),
            ],
            window_s=5.0,
        )

        self.assertTrue(analysis.summary["valid"])
        self.assertEqual(analysis.summary["active_requests_max"], 1)
        self.assertEqual(analysis.summary["full_overlap_duration_s"], 0.0)
        self.assertEqual(analysis.summary["full_overlap_token_count"], 0)
        self.assertIsNone(
            analysis.summary["full_overlap_throughput_tok_s"]
        )

    def test_reports_highest_qualified_one_second_window(self) -> None:
        results = [
            BENCHMARK.RequestResult(
                request_id=request_id,
                started_s=0.0,
                finished_s=10.0,
                token_times_s=[
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    *([5.25] * 25),
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                ],
                error=None,
            )
            for request_id in range(4)
        ]

        analysis = BENCHMARK.analyze_round(
            round_index=1,
            results=results,
            concurrency=4,
            window_s=1.0,
            step_s=0.5,
            min_full_overlap_s=0.0,
            min_full_overlap_tokens=0,
            min_peak_active_fraction=0.9,
        )

        self.assertEqual(
            analysis.summary["peak_window_min_active_requests"], 4
        )
        self.assertEqual(
            analysis.summary["peak_window_active_requests"], 4
        )
        self.assertEqual(
            analysis.summary["peak_window_throughput_tok_s"], 100.0
        )
        self.assertGreater(
            analysis.summary["peak_window_throughput_tok_s"],
            analysis.summary["window_throughput_tok_s"]["p50"],
        )


if __name__ == "__main__":
    unittest.main()
