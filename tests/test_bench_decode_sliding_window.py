from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "bench_decode_sliding_window.py"
)
SPEC = importlib.util.spec_from_file_location(
    "bench_decode_sliding_window", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


class MetricStatsTest(unittest.TestCase):
    def test_reports_required_distribution_metrics(self) -> None:
        stats = BENCH.metric_stats([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["avg"], 2.5)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 4.0)
        self.assertAlmostEqual(stats["stddev"], 1.2909944487358056)
        self.assertEqual(stats["p90"], 4.0)
        self.assertEqual(stats["p99"], 4.0)


class PromptConstructionTest(unittest.TestCase):
    def test_repeats_and_truncates_fixed_text_tokens_to_exact_length(self) -> None:
        prompt = BENCH.repeat_and_truncate_token_ids([11, 22, 33], 8)

        self.assertEqual(prompt, [11, 22, 33, 11, 22, 33, 11, 22])

    def test_rejects_fixed_text_without_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "produced no tokens"):
            BENCH.repeat_and_truncate_token_ids([], 8)


class StreamTokenAccountingTest(unittest.TestCase):
    def test_reads_complete_sse_lines_from_raw_stream(self) -> None:
        response = SimpleNamespace(
            raw=io.BytesIO(
                b": keep-alive\n\n"
                b'data: {"choices":[{"token_ids":[101]}]}\r\n\r\n'
                b"data: [DONE]\n\n"
            )
        )

        events = list(BENCH.iter_sse_data(response))

        self.assertEqual(
            events,
            ['{"choices":[{"token_ids":[101]}]}', "[DONE]"],
        )

    def test_records_every_token_id_in_a_coalesced_stream_chunk(self) -> None:
        timestamps = BENCH.token_timestamps_from_choice(
            {"token_ids": [101, 102, 103]}, 12.5
        )

        self.assertEqual(timestamps, [12.5, 12.5, 12.5])

    def test_rejects_stream_chunk_without_token_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing token_ids"):
            BENCH.token_timestamps_from_choice({"text": "answer"}, 12.5)


class SlidingWindowAnalysisTest(unittest.TestCase):
    def test_uses_full_concurrency_second_token_to_first_done_window(self) -> None:
        results = [
            BENCH.RequestResult(0, 0.0, 13.0, [float(i) for i in range(13)], None),
            BENCH.RequestResult(
                1, 0.1, 13.5, [float(i) + 0.5 for i in range(13)], None
            ),
        ]

        analysis = BENCH.analyze_round(
            round_index=1,
            results=results,
            concurrency=2,
            decode_tokens=13,
            window_s=2.0,
            step_s=1.0,
        )

        self.assertTrue(analysis.summary["valid"])
        self.assertEqual(analysis.summary["full_overlap_duration_s"], 10.5)
        self.assertEqual(analysis.summary["window_count"], 9)
        self.assertTrue(
            all(window["active_requests"] == 2 for window in analysis.windows)
        )
        self.assertTrue(
            all(window["throughput_tok_s"] == 2.0 for window in analysis.windows)
        )
        self.assertEqual(analysis.summary["decode_tpot_ms"]["avg"], 1000.0)

    def test_rejects_incomplete_request(self) -> None:
        results = [
            BENCH.RequestResult(0, 0.0, 3.0, [0.0, 1.0, 2.0], None),
            BENCH.RequestResult(1, 0.0, 2.0, [0.0, 1.0], None),
        ]

        analysis = BENCH.analyze_round(
            round_index=1,
            results=results,
            concurrency=2,
            decode_tokens=3,
            window_s=1.0,
            step_s=0.5,
        )

        self.assertFalse(analysis.summary["valid"])
        self.assertEqual(analysis.summary["invalid_reason"], "incomplete_requests")


if __name__ == "__main__":
    unittest.main()
