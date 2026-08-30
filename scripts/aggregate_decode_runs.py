#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


EXPECTED_BENCHMARK = {
    "concurrency": 256,
    "data_parallel_size": 4,
    "tensor_parallel_size": 2,
    "prefill_tokens": 65536,
    "decode_tokens": 1024,
}

AGGREGATE_FIELDS = (
    "active_requests_max",
    "peak_active_windows",
    "end_to_end_tok_s",
    "first_token_skew_s",
    "throughput_peak_window_tok_s",
    "peak_window_active_requests",
    "tpot_peak_window_p50_ms",
    "tpot_peak_window_p90_ms",
    "tpot_peak_window_p99_ms",
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


def replay_peak_window(
    summary_path: Path,
    benchmark: dict[str, Any],
) -> tuple[float, int, dict[str, float]]:
    """Recover the qualified peak window for pre-schema-8 raw results."""
    raw_path = summary_path.parent / "raw_requests.jsonl"
    if not raw_path.is_file():
        raise ValueError(f"{summary_path}: missing {raw_path.name}")
    records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    concurrency = integer(
        benchmark.get("concurrency"), "benchmark.concurrency", summary_path
    )
    if len(records) != concurrency:
        raise ValueError(
            f"{raw_path}: expected {concurrency} request records, "
            f"got {len(records)}"
        )
    window_s = finite_number(
        benchmark.get("window_s", 1.0), "benchmark.window_s", summary_path
    )
    step_s = finite_number(
        benchmark.get("step_s", 0.1), "benchmark.step_s", summary_path
    )
    min_active_fraction = finite_number(
        benchmark.get("min_peak_active_fraction", 0.9),
        "benchmark.min_peak_active_fraction",
        summary_path,
    )
    if not 0 < min_active_fraction <= 1:
        raise ValueError(
            f"{summary_path}: benchmark.min_peak_active_fraction must be "
            "in (0, 1]"
        )

    token_times_by_request: list[list[float]] = []
    for record in records:
        values = record.get("token_times_after_batch_start_s")
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"{raw_path}: request has fewer than two tokens")
        token_times_by_request.append(
            [
                finite_number(value, "raw token timestamp", raw_path)
                for value in values
            ]
        )
    decode_intervals = [
        (token_times[1], token_times[-1])
        for token_times in token_times_by_request
    ]
    all_token_times = sorted(
        timestamp
        for token_times in token_times_by_request
        for timestamp in token_times
    )
    minimum_active = math.ceil(concurrency * min_active_fraction)
    cursor_s = min(start_s for start_s, _ in decode_intervals)
    scan_end_s = max(end_s for _, end_s in decode_intervals)
    best_throughput: float | None = None
    best_active = 0
    best_start_s: float | None = None
    while cursor_s + window_s <= scan_end_s + 1e-9:
        end_s = cursor_s + window_s
        active_requests = sum(
            interval_start_s <= cursor_s and interval_end_s >= end_s
            for interval_start_s, interval_end_s in decode_intervals
        )
        if active_requests >= minimum_active:
            token_count = (
                bisect.bisect_left(all_token_times, end_s)
                - bisect.bisect_left(all_token_times, cursor_s)
            )
            throughput = token_count / window_s
            if best_throughput is None or throughput > best_throughput:
                best_throughput = throughput
                best_active = active_requests
                best_start_s = cursor_s
        cursor_s += step_s
    if best_throughput is None or best_start_s is None:
        raise ValueError(
            f"{raw_path}: no peak window reached {minimum_active}/"
            f"{concurrency} active requests"
        )
    best_end_s = best_start_s + window_s
    itls_ms = [
        (current_s - previous_s) * 1000
        for token_times in token_times_by_request
        for previous_s, current_s in zip(token_times, token_times[1:])
        if best_start_s <= current_s < best_end_s
    ]
    if not itls_ms:
        raise ValueError(f"{raw_path}: peak window contains no token intervals")

    def nearest_rank_percentile(percent: float) -> float:
        ordered = sorted(itls_ms)
        index = min(
            len(ordered) - 1,
            math.ceil(percent / 100 * len(ordered)) - 1,
        )
        return ordered[index]

    return (
        best_throughput,
        best_active,
        {
            "p50": nearest_rank_percentile(50),
            "p90": nearest_rank_percentile(90),
            "p99": nearest_rank_percentile(99),
        },
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

    peak_window_throughput = result.get("peak_window_throughput_tok_s")
    peak_window_active = result.get("peak_window_active_requests")
    peak_window_tpot = result.get("peak_window_tpot_ms")
    if (
        peak_window_throughput is None
        or peak_window_active is None
        or not isinstance(peak_window_tpot, dict)
    ):
        (
            peak_window_throughput,
            peak_window_active,
            peak_window_tpot,
        ) = replay_peak_window(summary_path, benchmark)

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
        "throughput_peak_window_tok_s": finite_number(
            peak_window_throughput,
            "peak_window_throughput_tok_s",
            summary_path,
        ),
        "peak_window_active_requests": integer(
            peak_window_active,
            "peak_window_active_requests",
            summary_path,
        ),
        "tpot_peak_window_p50_ms": finite_number(
            peak_window_tpot.get("p50"),
            "peak_window_tpot_ms.p50",
            summary_path,
        ),
        "tpot_peak_window_p90_ms": finite_number(
            peak_window_tpot.get("p90"),
            "peak_window_tpot_ms.p90",
            summary_path,
        ),
        "tpot_peak_window_p99_ms": finite_number(
            peak_window_tpot.get("p99"),
            "peak_window_tpot_ms.p99",
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
            f"DP4/TP2 C256/P65536/D1024; {client_processes}; "
            "distinct prompt and cache_salt per request; request_id % 4 "
            "DP-rank routing; no admission barrier; reported peak is the "
            "highest 1s window with >=90% submitted requests continuously "
            "active"
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
