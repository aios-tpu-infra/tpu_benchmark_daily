import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_PATH = PROJECT_ROOT / "scripts" / "aggregate_prefill_ttft.py"
SPEC = importlib.util.spec_from_file_location(
    "aggregate_prefill_ttft", AGGREGATE_PATH
)
assert SPEC is not None and SPEC.loader is not None
AGGREGATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATE)


class PrefillTtftAggregationTest(unittest.TestCase):
    def test_real_vllm_shape_is_validated_and_compacted(self) -> None:
        lengths = [8192, 16384, 32768, 65536, 131072, 258048]
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_dir = Path(temporary_directory)
            for index, input_length in enumerate(lengths, start=1):
                value_ms = float(index * 1000)
                path = (
                    result_dir
                    / f"vllm_dp8_single_request_ttft_len{input_length}.json"
                )
                path.write_text(
                    json.dumps(
                        {
                            "backend": "openai",
                            "num_prompts": 1,
                            "max_concurrency": 1,
                            "completed": 1,
                            "failed": 0,
                            "input_lens": [input_length],
                            "output_lens": [1],
                            "ttfts": [value_ms / 1000.0],
                            "mean_ttft_ms": value_ms,
                            "median_ttft_ms": value_ms,
                            "p90_ttft_ms": value_ms,
                            "p99_ttft_ms": value_ms,
                        }
                    ),
                    encoding="utf-8",
                )

            summary = AGGREGATE.aggregate(
                result_dir,
                "dp8",
                lengths,
                output_length=1,
                samples_per_length=1,
            )

        self.assertEqual(summary["benchmark"]["concurrency"], 1)
        self.assertEqual(summary["benchmark"]["statistic"], "median_ttft_ms")
        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["failed_input_lengths"], [])
        self.assertEqual(summary["results"][-1]["label"], "252K")
        self.assertEqual(summary["results"][-1]["status"], "success")
        self.assertEqual(summary["results"][-1]["ttft_ms"], 6000.0)
        self.assertEqual(summary["results"][-1]["raw_ttft_ms"], [6000.0])

    def test_failed_request_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_dir = Path(temporary_directory)
            path = result_dir / "vllm_dp8_single_request_ttft_len8192.json"
            path.write_text(
                json.dumps(
                    {
                        "num_prompts": 1,
                        "max_concurrency": 1,
                        "completed": 0,
                        "failed": 1,
                        "errors": ["request failed"],
                    }
                ),
                encoding="utf-8",
            )
            summary = AGGREGATE.aggregate(
                result_dir,
                "dp8",
                [8192],
                output_length=1,
                samples_per_length=1,
            )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["successful_input_lengths"], [])
        self.assertEqual(summary["failed_input_lengths"], [8192])
        self.assertEqual(summary["results"][0]["status"], "failed")
        self.assertIsNone(summary["results"][0]["ttft_ms"])
        self.assertEqual(summary["results"][0]["error"], "request failed")

    def test_missing_length_produces_partial_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_dir = Path(temporary_directory)
            path = result_dir / "vllm_dp8_single_request_ttft_len8192.json"
            path.write_text(
                json.dumps(
                    {
                        "num_prompts": 1,
                        "max_concurrency": 1,
                        "completed": 1,
                        "failed": 0,
                        "input_lens": [8192],
                        "output_lens": [1],
                        "ttfts": [1.5],
                        "mean_ttft_ms": 1500.0,
                        "median_ttft_ms": 1500.0,
                        "p90_ttft_ms": 1500.0,
                        "p99_ttft_ms": 1500.0,
                    }
                ),
                encoding="utf-8",
            )
            summary = AGGREGATE.aggregate(
                result_dir,
                "dp8",
                [8192, 258048],
                output_length=1,
                samples_per_length=1,
            )

        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["successful_input_lengths"], [8192])
        self.assertEqual(summary["failed_input_lengths"], [258048])
        self.assertEqual(summary["results"][1]["status"], "failed")
        self.assertEqual(summary["results"][1]["failed"], 1)
        self.assertIn("missing", summary["results"][1]["error"])

    def test_ttfts_seconds_must_match_reported_milliseconds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_dir = Path(temporary_directory)
            path = result_dir / "vllm_dp8_single_request_ttft_len8192.json"
            path.write_text(
                json.dumps(
                    {
                        "num_prompts": 1,
                        "max_concurrency": 1,
                        "completed": 1,
                        "failed": 0,
                        "input_lens": [8192],
                        "output_lens": [1],
                        "ttfts": [1.5],
                        "mean_ttft_ms": 1500.0,
                        "median_ttft_ms": 1.5,
                        "p90_ttft_ms": 1500.0,
                        "p99_ttft_ms": 1500.0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "does not match ttfts seconds",
            ):
                AGGREGATE.aggregate(
                    result_dir,
                    "dp8",
                    [8192],
                    output_length=1,
                    samples_per_length=1,
                )


if __name__ == "__main__":
    unittest.main()
