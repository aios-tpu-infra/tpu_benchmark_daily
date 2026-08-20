#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


EXPECTED_BENCHMARK = {
    "concurrency": 256,
    "prefill_tokens": 65536,
    "decode_tokens": 1024,
}

AGGREGATE_FIELDS = (
    "active_requests_max",
    "peak_active_windows",
    "end_to_end_tok_s",
    "first_token_skew_s",
    "throughput_peak_active_p50_tok_s",
    "tpot_peak_active_p50_ms",
    "tpot_peak_active_p90_ms",
    "tpot_peak_active_p99_ms",
    "ttft_p50_s",
    "ttft_p90_s",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def finite_number(value: Any, field: str, summary_path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{summary_path}: {field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{summary_path}: {field} must be a finite number")
    return parsed


def integer(value: Any, field: str, summary_path: Path) -> int:
    parsed = finite_number(value, field, summary_path)
    if not parsed.is_integer():
        raise ValueError(f"{summary_path}: {field} must be an integer")
    return int(parsed)


def nested_metric(
    result: dict[str, Any],
    group: str,
    metric: str,
    summary_path: Path,
) -> float:
    values = result.get(group)
    if not isinstance(values, dict):
        raise ValueError(f"{summary_path}: {group} must be an object")
    return finite_number(
        values.get(metric), f"{group}.{metric}", summary_path
    )


def load_run(summary_path: Path, run_index: int) -> dict[str, Any]:
    if not summary_path.is_file():
        raise ValueError(f"missing required result: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {summary_path}: {exc}") from exc

    benchmark = summary.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError(f"{summary_path}: benchmark must be an object")
    for field, expected in EXPECTED_BENCHMARK.items():
        actual = benchmark.get(field)
        if actual != expected:
            raise ValueError(
                f"{summary_path}: benchmark.{field}={actual!r}, "
                f"expected {expected}"
            )

    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"{summary_path}: expected exactly one result row")
    result = results[0]
    if not isinstance(result, dict):
        raise ValueError(f"{summary_path}: result row must be an object")

    failed_requests = integer(
        result.get("failed_requests"), "failed_requests", summary_path
    )
    if failed_requests:
        raise ValueError(
            f"{summary_path}: failed_requests={failed_requests}, expected 0"
        )
    successful_requests = integer(
        result.get("successful_requests"),
        "successful_requests",
        summary_path,
    )
    if successful_requests != EXPECTED_BENCHMARK["concurrency"]:
        raise ValueError(
            f"{summary_path}: successful_requests={successful_requests}, "
            f"expected {EXPECTED_BENCHMARK['concurrency']}"
        )

    active_requests_max = integer(
        result.get("active_requests_max"),
        "active_requests_max",
        summary_path,
    )
    window_count = integer(
        result.get("window_count"), "window_count", summary_path
    )
    if active_requests_max <= 0:
        raise ValueError(
            f"{summary_path}: active_requests_max must be positive"
        )
    if window_count <= 0:
        raise ValueError(f"{summary_path}: window_count must be positive")

    return {
        "run": run_index,
        "ok_requests": successful_requests,
        "failed_requests": failed_requests,
        "tokens": integer(
            result.get("usage_tokens"), "usage_tokens", summary_path
        ),
        "active_requests_max": active_requests_max,
        "peak_active_windows": window_count,
        "end_to_end_tok_s": finite_number(
            result.get("end_to_end_throughput_tok_s"),
            "end_to_end_throughput_tok_s",
            summary_path,
        ),
        "first_token_skew_s": finite_number(
            result.get("first_token_skew_s"),
            "first_token_skew_s",
            summary_path,
        ),
        "throughput_peak_active_p50_tok_s": nested_metric(
            result,
            "window_throughput_tok_s",
            "p50",
            summary_path,
        ),
        "tpot_peak_active_p50_ms": nested_metric(
            result, "peak_active_tpot_ms", "p50", summary_path
        ),
        "tpot_peak_active_p90_ms": nested_metric(
            result, "peak_active_tpot_ms", "p90", summary_path
        ),
        "tpot_peak_active_p99_ms": nested_metric(
            result, "peak_active_tpot_ms", "p99", summary_path
        ),
        "ttft_p50_s": nested_metric(
            result, "ttft_s", "p50", summary_path
        ),
        "ttft_p90_s": nested_metric(
            result, "ttft_s", "p90", summary_path
        ),
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def distribution(values: list[float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot aggregate an empty metric")
    return {
        "count": len(values),
        "avg": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
    }


def aggregate_result_root(result_root: Path, runs: int) -> dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs must be positive")
    run_rows = [
        load_run(result_root / f"run_{run_index}" / "summary.json", run_index)
        for run_index in range(1, runs + 1)
    ]
    aggregate = {
        field: distribution([float(row[field]) for row in run_rows])
        for field in AGGREGATE_FIELDS
    }
    client_processes = (
        "one independent client process"
        if runs == 1
        else f"{runs} independent client processes"
    )
    return {
        "schema_version": 1,
        "result_root": str(result_root),
        "protocol": (
            f"C256/P65536/D1024; {client_processes}; "
            "distinct prompt and cache_salt per request; request_id % 8 "
            "DP-rank routing; no admission barrier"
        ),
        "statistics": {
            "stddev": "sample",
            "percentile": "linear interpolation",
        },
        "runs": run_rows,
        "aggregate": aggregate,
    }


def write_outputs(
    result_root: Path, payload: dict[str, Any]
) -> tuple[Path, Path]:
    result_root.mkdir(parents=True, exist_ok=True)
    json_path = result_root / "aggregate.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    csv_path = result_root / "aggregate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "metric",
                "count",
                "avg",
                "min",
                "max",
                "stddev",
                "p90",
                "p99",
            ),
        )
        writer.writeheader()
        aggregate = payload["aggregate"]
        for metric in AGGREGATE_FIELDS:
            writer.writerow({"metric": metric, **aggregate[metric]})
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate independent C256 decode benchmark processes."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--runs", type=positive_int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = aggregate_result_root(args.result_root, args.runs)
    json_path, csv_path = write_outputs(args.result_root, payload)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
