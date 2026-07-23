from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "update_report.py"
SPEC = importlib.util.spec_from_file_location("update_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def make_run(
    config: str,
    *,
    run_id: str = "20260722T180001Z",
    completed_at: str = "2026-07-22T18:20:00+00:00",
    throughput: float = 40_000.0,
    decode_throughput: float | None = None,
    decode_min_tpot_ms: float | None = None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "benchmark_config": config,
        "started_at": "2026-07-22T18:00:01+00:00",
        "completed_at": completed_at,
        "machine_ip": "10.42.4.22",
        "model": "test-model",
        "input_length": 8192,
        "output_length": 1,
        "best_total_token_throughput": throughput,
        "best_request_throughput": 4.0,
        "best_concurrency": 2,
        "mean_ttft_ms": 100.0,
        "p99_ttft_ms": 200.0,
        "decode_peak_output_throughput": decode_throughput,
        "decode_min_tpot_ms": decode_min_tpot_ms,
        "torchtpu_vllm_revision": "abc123def4567890",
        "torch_tpu_revision": "",
        "torch_tpu_version": "1.0",
        "summary_path": f"runs/{run_id}/results/{config}/summary.json",
        "concurrency_results": [
            {
                "concurrency": 1,
                "total_token_throughput": throughput / 2,
                "request_throughput": 2.0,
                "mean_ttft_ms": 80.0,
                "p99_ttft_ms": 160.0,
            },
            {
                "concurrency": 2,
                "total_token_throughput": throughput,
                "request_throughput": 4.0,
                "mean_ttft_ms": 100.0,
                "p99_ttft_ms": 200.0,
            },
        ],
    }


class HistoryMigrationTest(unittest.TestCase):
    def test_schema_one_history_is_migrated_to_dp_and_pcp_configs(self) -> None:
        payload = {
            "schema_version": 1,
            "runs": [
                {"run_id": "20260721T180001Z"},
                {
                    "run_id": "manual-pcp8-bench-20260722T050712Z",
                    "summary_path": "runs/manual-pcp/results/summary.json",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            runs = REPORT.load_history(path)

        self.assertEqual(
            [run["benchmark_config"] for run in runs], ["dp8", "pcp8"]
        )
        self.assertTrue(
            all(run["decode_peak_output_throughput"] is None for run in runs)
        )
        self.assertTrue(all(run["decode_min_tpot_ms"] is None for run in runs))

    def test_same_run_id_keeps_both_benchmark_configs(self) -> None:
        dp_run = make_run("dp8")
        pcp_run = make_run(
            "pcp8",
            completed_at="2026-07-22T18:40:00+00:00",
            throughput=35_000.0,
        )

        runs = REPORT.update_history([], dp_run)
        runs = REPORT.update_history(runs, pcp_run)

        self.assertEqual(len(runs), 2)
        self.assertEqual(
            {(run["run_id"], run["benchmark_config"]) for run in runs},
            {("20260722T180001Z", "dp8"), ("20260722T180001Z", "pcp8")},
        )


class DualSeriesReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runs = [
            make_run("dp8", throughput=40_000.0),
            make_run(
                "pcp8",
                completed_at="2026-07-22T18:40:00+00:00",
                throughput=35_000.0,
            ),
        ]

    def test_concurrency_chart_contains_two_curves(self) -> None:
        latest = REPORT.latest_runs_by_config(self.runs)
        series, labels = REPORT.concurrency_chart_data(latest)

        svg = REPORT.chart_svg(
            series,
            labels,
            title="Latest DP8 vs PCP8 throughput by concurrency",
        )
        root = ET.fromstring(svg)
        polylines = root.findall("{http://www.w3.org/2000/svg}polyline")

        self.assertEqual(len(polylines), 2)
        self.assertIn("DP8", svg)
        self.assertIn("PCP8", svg)
        self.assertIn("#1570ef", svg)
        self.assertIn("#7a5af8", svg)

    def test_latest_json_contains_both_configs(self) -> None:
        payload = json.loads(REPORT.render_latest_json(self.runs))

        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(set(payload["benchmarks"]), {"dp8", "pcp8"})
        self.assertEqual(
            payload["benchmarks"]["pcp8"]["total_token_throughput"], 35_000.0
        )

    def test_history_chart_contains_two_time_series(self) -> None:
        history_runs = [
            make_run(
                "dp8",
                run_id="20260721T180001Z",
                completed_at="2026-07-21T18:20:00+00:00",
                throughput=39_000.0,
            ),
            make_run(
                "pcp8",
                run_id="20260721T180001Z",
                completed_at="2026-07-21T18:40:00+00:00",
                throughput=34_000.0,
            ),
            *self.runs,
        ]
        series, labels = REPORT.history_chart_data(history_runs)

        svg = REPORT.chart_svg(
            series,
            labels,
            title="DP8 vs PCP8 peak throughput over time",
        )
        root = ET.fromstring(svg)
        polylines = root.findall("{http://www.w3.org/2000/svg}polyline")

        self.assertEqual(labels, ["07-21 18:40", "07-22 18:40"])
        self.assertEqual(len(polylines), 2)
        self.assertIn("DP8", svg)
        self.assertIn("PCP8", svg)
        last_tick = next(
            node
            for node in root.findall("{http://www.w3.org/2000/svg}text")
            if node.text == labels[-1]
        )
        self.assertEqual(last_tick.attrib["text-anchor"], "end")

    def test_readme_block_embeds_both_report_charts(self) -> None:
        block = REPORT.render_readme_block(self.runs, table_limit=10)

        self.assertIn("reports/throughput.svg", block)
        self.assertIn("reports/throughput_history.svg", block)
        self.assertNotIn("index.html", block)
        self.assertNotIn("[![", block)

    def test_readme_table_combines_dp_and_pcp_for_one_run(self) -> None:
        self.runs[0]["decode_peak_output_throughput"] = 637.685
        self.runs[0]["decode_min_tpot_ms"] = 20.507

        block = REPORT.render_readme_block(self.runs, table_limit=10)

        self.assertIn("| vllm-torchtpu commit | Test time (UTC) |", block)
        self.assertIn(
            "| `abc123def456` | 2026-07-22 18:40 | 40,000.00 | "
            "35,000.00 | 637.68 | 20.51 |",
            block,
        )
        self.assertNotIn("| Completed (UTC) | Config |", block)
        self.assertEqual(block.count("| `abc123def456` |"), 1)

    def test_decode_metrics_reads_valid_dp_summary_only(self) -> None:
        summary = {
            "decode_sliding_window": {
                "best": {"output_throughput": 637.685},
                "aggregate": {"request_tpot_ms": {"min": 20.507}},
            }
        }

        self.assertEqual(
            REPORT.decode_metrics(summary, "dp8"),
            (637.685, 20.507),
        )
        self.assertEqual(REPORT.decode_metrics(summary, "pcp8"), (None, None))
        self.assertEqual(REPORT.decode_metrics({}, "dp8"), (None, None))


if __name__ == "__main__":
    unittest.main()
