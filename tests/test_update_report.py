import importlib.util
import json
import os
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
    def test_summary_mtime_wins_over_timezone_less_detail_date(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            run_dir = project_root / "runs" / "20260727T043341Z"
            result_dir = run_dir / "results" / "dp8"
            result_dir.mkdir(parents=True)
            detail_path = result_dir / "best.json"
            detail_path.write_text(
                json.dumps({"date": "20990101-000000"}), encoding="utf-8"
            )
            result = {
                "concurrency": 8,
                "total_token_throughput": 1234.0,
                "request_throughput": 1.0,
                "mean_ttft_ms": 100.0,
                "p99_ttft_ms": 110.0,
                "completed": 1,
                "failed": 0,
                "file": detail_path.name,
            }
            summary_path = result_dir / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "benchmark": {
                            "benchmark_config": "dp8",
                            "input_length": 8192,
                            "output_length": 1,
                            "model": "fixture",
                        },
                        "best": result,
                        "results": [result],
                    }
                ),
                encoding="utf-8",
            )
            summary_mtime = 1_700_000_000
            os.utime(summary_path, (summary_mtime, summary_mtime))

            record = UPDATE_REPORT.build_record(
                project_root=project_root,
                run_dir=run_dir,
                summary_path=summary_path,
                input_length=8192,
                output_length=1,
                model="fixture",
                benchmark_config="dp8",
                status="success",
                decode_status="not-run",
            )

        self.assertEqual(record["completed_at"], "2023-11-14T22:13:20+00:00")

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
        self.assertEqual(record["decode_parallelism"], "DP8/TP1/EP8")
        self.assertEqual(record["best_total_token_throughput"], -1.0)
        self.assertEqual(record["decode_window_p50_throughput"], -1.0)
        self.assertIsNone(record["best_concurrency"])
        self.assertEqual(record["concurrency_results"], [])

        block = UPDATE_REPORT.render_readme_block([record], table_limit=10)
        self.assertIn("Latest DP8: **failed (-1.00 total tok/s)**", block)
        self.assertIn("| -1.00 | — | — | — | — |", block)

        latest = json.loads(UPDATE_REPORT.render_latest_json([record]))
        self.assertEqual(latest["schema_version"], 10)
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
                            "throughput_peak_window_tok_s": {
                                "avg": 6400.0
                            },
                            "tpot_peak_window_p50_ms": {"avg": 40.8},
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
                decode_parallelism="DP4/TP2/EP8",
                status="not-run",
                decode_status="success",
            )

        self.assertIsNone(record["best_total_token_throughput"])
        self.assertEqual(record["decode_parallelism"], "DP4/TP2/EP8")
        self.assertEqual(record["decode_peak_1s_throughput"], 6400.0)
        self.assertEqual(record["decode_peak_1s_tpot_p50_ms"], 40.8)
        self.assertEqual(record["decode_window_p50_throughput"], 3900.0)
        self.assertEqual(record["decode_peak_active_tpot_p50_ms"], 47.0)
        block = UPDATE_REPORT.render_readme_block([record], table_limit=10)
        self.assertIn("Recent DP4/TP2 decode throughput", block)
        self.assertIn("DP4/TP2 decode tok/s", block)
        self.assertIn(
            "DP4/TP2/EP8 C256 peak 1s (>=90% active)", block
        )
        latest = json.loads(UPDATE_REPORT.render_latest_json([record]))
        self.assertEqual(latest["prefill_benchmarks"], {})
        self.assertEqual(latest["decode"]["run_id"], "decode-only")
        self.assertEqual(
            latest["decode"]["decode_peak_1s_throughput"], 6400.0
        )
        self.assertEqual(
            latest["decode"]["decode_peak_1s_tpot_p50_ms"], 40.8
        )
        self.assertEqual(
            latest["decode"]["decode_window_p50_throughput"], 3900.0
        )

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

    def test_decode_only_record_does_not_replace_latest_prefill(self) -> None:
        prefill_record = {
            "run_id": "prefill",
            "benchmark_config": "dp8",
            "status": "success",
        }
        decode_record = {
            "run_id": "decode",
            "benchmark_config": "dp8",
            "status": "not-run",
            "decode_status": "success",
        }

        latest = UPDATE_REPORT.latest_prefill_runs_by_config(
            [prefill_record, decode_record]
        )

        self.assertEqual(latest["dp8"]["run_id"], "prefill")

    def test_decode_history_only_shows_dp4_tp2_and_keeps_metrics_separate(
        self,
    ) -> None:
        runs = [
            {
                "run_id": "dp8-legacy-run",
                "benchmark_config": "dp8",
                "completed_at": "2026-07-23T18:00:00+00:00",
                "decode_parallelism": "DP8/TP1/EP8",
                UPDATE_REPORT.LEGACY_DECODE_THROUGHPUT_FIELD: 632.0,
                "decode_window_p50_throughput": None,
            },
            {
                "run_id": "dp4-legacy-run",
                "benchmark_config": "dp8",
                "completed_at": "2026-07-24T18:00:00+00:00",
                "decode_parallelism": "DP4/TP2/EP8",
                UPDATE_REPORT.LEGACY_DECODE_THROUGHPUT_FIELD: 700.0,
                "decode_window_p50_throughput": None,
            },
            {
                "run_id": "dp4-current-run",
                "benchmark_config": "dp8",
                "completed_at": "2026-07-25T18:00:00+00:00",
                "decode_parallelism": "DP4/TP2/EP8",
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

        self.assertEqual(labels, ["07-24 18:00", "07-25 18:00"])
        self.assertEqual(
            [item["config"] for item in series],
            ["legacy", "reported-peak"],
        )
        self.assertEqual(
            [point["value"] for point in series[0]["points"]],
            [700.0],
        )
        self.assertEqual(
            [point["value"] for point in series[1]["points"]],
            [3968.0],
        )

    def test_single_request_ttft_is_recorded_and_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            run_dir = project_root / "runs" / "ttft-run"
            run_dir.mkdir(parents=True)
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "started_at": "2026-07-29T07:00:00+00:00",
                        "machine_ip": "10.42.4.56",
                        "torchtpu_vllm_revision": "fixture",
                        "torch_tpu_version": "fixture",
                    }
                ),
                encoding="utf-8",
            )
            ttft_summary = run_dir / "ttft-summary.json"
            ttft_summary.write_text(
                json.dumps(
                    {
                        "benchmark": {
                            "benchmark_config": "dp8",
                            "concurrency": 1,
                        },
                        "results": [
                            {
                                "label": "8K",
                                "input_length": 8192,
                                "output_length": 1,
                                "completed": 1,
                                "failed": 0,
                                "status": "success",
                                "ttft_ms": 1234.5,
                            },
                            {
                                "label": "252K",
                                "input_length": 258048,
                                "output_length": 1,
                                "completed": 1,
                                "failed": 0,
                                "status": "success",
                                "ttft_ms": 23456.7,
                            },
                        ],
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
                ttft_summary_path=ttft_summary,
                status="not-run",
                decode_status="not-run",
                ttft_status="success",
            )

        self.assertEqual(record["prefill_ttft_status"], "success")
        self.assertEqual(
            [item["input_length"] for item in record["single_request_ttft_results"]],
            [8192, 258048],
        )
        latest_ttft = UPDATE_REPORT.latest_ttft_runs_by_config([record])
        series, labels = UPDATE_REPORT.prefill_ttft_chart_data(latest_ttft)
        self.assertEqual(labels, ["8K", "252K"])
        self.assertEqual(
            [point["value"] for point in series[0]["points"]],
            [1234.5, 23456.7],
        )
        block = UPDATE_REPORT.render_readme_block([record], table_limit=10)
        self.assertIn("reports/prefill_ttft.svg", block)
        self.assertIn("1 serial sample/length", block)
        self.assertIn("DP TTFT 8K (ms)", block)
        self.assertIn("PCP TTFT 252K (ms)", block)
        self.assertNotIn("| Input length | DP8 TTFT", block)
        self.assertIn("| 1,234.50 | — | 23,456.70 | — |", block)
        latest = json.loads(UPDATE_REPORT.render_latest_json([record]))
        self.assertEqual(
            latest["benchmarks"]["dp8"]["single_request_ttft_results"][0][
                "ttft_ms"
            ],
            1234.5,
        )
        csv_report = UPDATE_REPORT.render_prefill_ttft_csv([record])
        self.assertIn("ttft-run,dp8,success,success", csv_report)

    def test_partial_ttft_renders_failed_cell_and_successful_chart_points(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            run_dir = project_root / "runs" / "partial-ttft-run"
            run_dir.mkdir(parents=True)
            ttft_summary = run_dir / "ttft-summary.json"
            ttft_summary.write_text(
                json.dumps(
                    {
                        "status": "partial",
                        "benchmark": {
                            "benchmark_config": "dp8",
                            "concurrency": 1,
                        },
                        "results": [
                            {
                                "label": "8K",
                                "input_length": 8192,
                                "output_length": 1,
                                "completed": 16,
                                "failed": 0,
                                "status": "success",
                                "ttft_ms": 1234.5,
                            },
                            {
                                "label": "252K",
                                "input_length": 258048,
                                "output_length": 1,
                                "completed": 0,
                                "failed": 16,
                                "status": "failed",
                                "ttft_ms": None,
                                "error": "service unavailable",
                            },
                        ],
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
                ttft_summary_path=ttft_summary,
                status="not-run",
                decode_status="not-run",
                ttft_status="partial",
            )

        self.assertEqual(record["prefill_ttft_status"], "partial")
        self.assertEqual(
            record["single_request_ttft_results"][1]["status"],
            "failed",
        )
        self.assertIsNone(
            record["single_request_ttft_results"][1]["ttft_ms"]
        )
        series, labels = UPDATE_REPORT.prefill_ttft_chart_data(
            {"dp8": record}
        )
        self.assertEqual(labels, ["8K", "252K"])
        self.assertEqual(
            [point["value"] for point in series[0]["points"]],
            [1234.5],
        )
        block = UPDATE_REPORT.render_readme_block([record], table_limit=10)
        self.assertIn(
            "Latest DP8 single-request TTFT: **partial**, "
            "**16 serial samples/length**",
            block,
        )
        self.assertIn("| 1,234.50 | — | failed | — |", block)
        csv_report = UPDATE_REPORT.render_prefill_ttft_csv([record])
        self.assertIn(
            "partial-ttft-run,dp8,partial,failed",
            csv_report,
        )

    def test_history_table_combines_dp_and_pcp_ttft_by_run(self) -> None:
        def record(
            *,
            run_id: str,
            config: str,
            revision: str,
            throughput: float,
            ttft_ms: float | None,
            ttft_status: str = "success",
        ) -> dict[str, object]:
            ttft_result = {
                "label": "8K",
                "input_length": 8192,
                "output_length": 1,
                "completed": 16 if ttft_ms is not None else 0,
                "failed": 0 if ttft_ms is not None else 16,
                "status": "success" if ttft_ms is not None else "failed",
                "ttft_ms": ttft_ms,
            }
            return {
                "run_id": run_id,
                "benchmark_config": config,
                "status": "success",
                "decode_status": "not-run",
                "started_at": f"2026-07-{30 if run_id == 'new' else 29}T00:00:00+00:00",
                "completed_at": f"2026-07-{30 if run_id == 'new' else 29}T01:00:00+00:00",
                "best_total_token_throughput": throughput,
                "best_concurrency": 16,
                "decode_window_p50_throughput": None,
                "decode_peak_active_tpot_p50_ms": None,
                UPDATE_REPORT.LEGACY_DECODE_THROUGHPUT_FIELD: None,
                UPDATE_REPORT.LEGACY_DECODE_TPOT_FIELD: None,
                "prefill_ttft_status": ttft_status,
                "single_request_ttft_results": [ttft_result],
                "torchtpu_vllm_revision": revision,
            }

        runs = [
            record(
                run_id="old",
                config="dp8",
                revision="oldrevision",
                throughput=49000.0,
                ttft_ms=None,
                ttft_status="failed",
            ),
            record(
                run_id="new",
                config="dp8",
                revision="newrevision",
                throughput=50000.0,
                ttft_ms=1500.0,
            ),
            record(
                run_id="new",
                config="pcp8",
                revision="newrevision",
                throughput=41000.0,
                ttft_ms=800.0,
            ),
        ]

        block = UPDATE_REPORT.render_readme_block(runs, table_limit=10)

        self.assertIn("DP TTFT 8K (ms) | PCP TTFT 8K (ms)", block)
        self.assertIn(
            "| `newrevision` | 2026-07-30 00:00 | "
            "50,000.00 | 41,000.00 | — | — | — | 1,500.00 | 800.00 |",
            block,
        )
        self.assertIn(
            "| `oldrevision` | 2026-07-29 00:00 | "
            "49,000.00 | — | — | — | — | failed | — |",
            block,
        )


if __name__ == "__main__":
    unittest.main()
