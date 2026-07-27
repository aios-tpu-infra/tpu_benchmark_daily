import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATE_REPORT_PATH = PROJECT_ROOT / "scripts" / "update_report.py"
SPEC = importlib.util.spec_from_file_location("update_report", UPDATE_REPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
UPDATE_REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPDATE_REPORT)


class UpdateReportTest(unittest.TestCase):
    def test_failed_attempt_records_minus_one_without_a_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            run_dir = project_root / "runs" / "20260727T043341Z"
            run_dir.mkdir(parents=True)
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "started_at": "2026-07-27T04:33:41+00:00",
                        "machine_ip": "10.42.4.56",
                        "torchtpu_vllm_revision": "9e00af3440f4",
                        "torch_tpu_version": "0.1.1.dev20260714090201",
                    }
                ),
                encoding="utf-8",
            )

            record = UPDATE_REPORT.build_record(
                project_root=project_root,
                run_dir=run_dir,
                summary_path=None,
                input_length=8192,
                output_length=1,
                model="Qwen3.5-397B-A17B-FP8",
                benchmark_config="dp8",
                status="failed",
                decode_status="failed",
            )

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["decode_status"], "failed")
        self.assertEqual(record["best_total_token_throughput"], -1.0)
        self.assertEqual(record["decode_window_p50_throughput"], -1.0)
        self.assertIsNone(record["best_concurrency"])
        self.assertEqual(record["concurrency_results"], [])

        block = UPDATE_REPORT.render_readme_block([record], table_limit=10)
        self.assertIn("Latest DP8: **failed (-1.00 total tok/s)**", block)
        self.assertIn("| -1.00 | — | -1.00 | — | failed |", block)

        latest = json.loads(UPDATE_REPORT.render_latest_json([record]))
        self.assertEqual(latest["schema_version"], 5)
        self.assertEqual(latest["benchmarks"]["dp8"]["status"], "failed")
        self.assertEqual(
            latest["benchmarks"]["dp8"]["total_token_throughput"],
            -1.0,
        )
        self.assertEqual(
            latest["benchmarks"]["dp8"]["decode_window_p50_throughput"],
            -1.0,
        )
        csv_report = UPDATE_REPORT.render_csv([record])
        self.assertIn("status,decode_status", csv_report.splitlines()[0])
        self.assertIn(",failed,failed,", csv_report)

        prefill_series, _ = UPDATE_REPORT.history_chart_data([record])
        decode_series, _ = UPDATE_REPORT.decode_history_chart_data([record])
        self.assertEqual(prefill_series, [])
        self.assertEqual(decode_series, [])

    def test_not_run_prefill_can_carry_a_successful_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            run_dir = project_root / "runs" / "decode-only"
            run_dir.mkdir(parents=True)
            decode_summary_path = run_dir / "aggregate.json"
            decode_summary_path.write_text(
                json.dumps(
                    {
                        "aggregate": {
                            "throughput_peak_active_p50_tok_s": {
                                "avg": 3900.0
                            },
                            "tpot_peak_active_p50_ms": {"avg": 47.0},
                        }
                    }
                ),
                encoding="utf-8",
            )

            record = UPDATE_REPORT.build_record(
                project_root=project_root,
                run_dir=run_dir,
                summary_path=None,
                input_length=8192,
                output_length=1,
                model="Qwen3.5-397B-A17B-FP8",
                benchmark_config="dp8",
                decode_summary_path=decode_summary_path,
                status="not-run",
                decode_status="success",
            )

        self.assertIsNone(record["best_total_token_throughput"])
        self.assertEqual(record["decode_window_p50_throughput"], 3900.0)
        self.assertEqual(record["decode_peak_active_tpot_p50_ms"], 47.0)

    def test_schema3_decode_metrics_migrate_without_changing_protocol(self) -> None:
        history = {
            "schema_version": 3,
            "runs": [
                {
                    "run_id": "20260724T003214Z",
                    "benchmark_config": "dp8",
                    "completed_at": "2026-07-24T01:03:52+00:00",
                    "machine_ip": "10.42.4.22",
                    "decode_peak_output_throughput": 632.2387,
                    "decode_min_tpot_ms": 18.4131,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            history_path = Path(temporary_directory) / "throughput_history.json"
            history_path.write_text(json.dumps(history), encoding="utf-8")

            [run] = UPDATE_REPORT.load_history(history_path)

        self.assertEqual(
            run[UPDATE_REPORT.LEGACY_DECODE_THROUGHPUT_FIELD],
            632.2387,
        )
        self.assertEqual(run[UPDATE_REPORT.LEGACY_DECODE_TPOT_FIELD], 18.4131)
        self.assertIsNone(run["decode_window_p50_throughput"])
        self.assertIsNone(run["decode_peak_active_tpot_p50_ms"])
        self.assertNotIn("decode_peak_output_throughput", run)
        self.assertNotIn("decode_min_tpot_ms", run)

        replacement = {
            **run,
            UPDATE_REPORT.LEGACY_DECODE_THROUGHPUT_FIELD: None,
            UPDATE_REPORT.LEGACY_DECODE_TPOT_FIELD: None,
        }
        [updated] = UPDATE_REPORT.update_history([run], replacement)
        self.assertEqual(
            updated[UPDATE_REPORT.LEGACY_DECODE_THROUGHPUT_FIELD],
            632.2387,
        )
        self.assertEqual(
            updated[UPDATE_REPORT.LEGACY_DECODE_TPOT_FIELD],
            18.4131,
        )

    def test_decode_history_keeps_legacy_and_peak_active_series_separate(
        self,
    ) -> None:
        runs = [
            {
                "run_id": "legacy-run",
                "benchmark_config": "dp8",
                "completed_at": "2026-07-23T18:00:00+00:00",
                UPDATE_REPORT.LEGACY_DECODE_THROUGHPUT_FIELD: 632.0,
                "decode_window_p50_throughput": None,
            },
            {
                "run_id": "current-run",
                "benchmark_config": "dp8",
                "completed_at": "2026-07-24T18:00:00+00:00",
                UPDATE_REPORT.LEGACY_DECODE_THROUGHPUT_FIELD: None,
                "decode_window_p50_throughput": 3968.0,
            },
            {
                "run_id": "pcp-run",
                "benchmark_config": "pcp8",
                "completed_at": "2026-07-24T18:30:00+00:00",
                UPDATE_REPORT.LEGACY_DECODE_THROUGHPUT_FIELD: None,
                "decode_window_p50_throughput": None,
            },
        ]

        series, labels = UPDATE_REPORT.decode_history_chart_data(runs)

        self.assertEqual(labels, ["07-23 18:00", "07-24 18:00"])
        self.assertEqual(
            [item["config"] for item in series],
            ["legacy", "peak-active-p50"],
        )
        self.assertEqual(
            [point["value"] for point in series[0]["points"]],
            [632.0],
        )
        self.assertEqual(
            [point["value"] for point in series[1]["points"]],
            [3968.0],
        )


if __name__ == "__main__":
    unittest.main()
