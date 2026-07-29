#!/usr/bin/env python3

"""Validate vLLM single-request TTFT outputs and write a compact summary."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


RESULT_RE = re.compile(r"_len(?P<input_length>\d+)\.json$")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def finite_float(value: Any, field: str, path: Path) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} in {path} must be finite, got {value!r}")
    return parsed


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def length_label(input_length: int) -> str:
    if input_length % 1024 == 0:
        return f"{input_length // 1024}K"
    return str(input_length)


def aggregate(
    result_dir: Path,
    benchmark_config: str,
    expected_input_lengths: list[int],
    output_length: int,
    samples_per_length: int,
) -> dict[str, Any]:
    expected = set(expected_input_lengths)
    records: dict[int, dict[str, Any]] = {}
    paths_by_length: dict[int, Path] = {}

    for path in sorted(result_dir.glob("vllm_*_single_request_ttft_len*.json")):
        match = RESULT_RE.search(path.name)
        if match is None:
            continue
        nominal_input_length = int(match.group("input_length"))
        if nominal_input_length not in expected:
            continue
        if nominal_input_length in paths_by_length:
            raise ValueError(
                f"duplicate TTFT result for {nominal_input_length}: {path}"
            )
        paths_by_length[nominal_input_length] = path

    for nominal_input_length in expected_input_lengths:
        path = paths_by_length.get(nominal_input_length)
        if path is None:
            records[nominal_input_length] = {
                "file": (
                    f"vllm_{benchmark_config}_single_request_ttft_"
                    f"len{nominal_input_length}.json"
                ),
                "label": length_label(nominal_input_length),
                "input_length": nominal_input_length,
                "output_length": output_length,
                "status": "failed",
                "completed": 0,
                "failed": samples_per_length,
                "ttft_ms": None,
                "mean_ttft_ms": None,
                "median_ttft_ms": None,
                "p90_ttft_ms": None,
                "p99_ttft_ms": None,
                "raw_ttft_ms": [],
                "error": "vLLM result file is missing",
            }
            continue

        data = load_json(path)
        completed = int(data.get("completed", 0))
        failed = int(data.get("failed", 0))
        if completed != samples_per_length or failed != 0:
            errors = data.get("errors")
            error_messages = (
                list(
                    dict.fromkeys(
                        str(value) for value in errors if str(value).strip()
                    )
                )
                if isinstance(errors, list)
                else []
            )
            records[nominal_input_length] = {
                "file": path.name,
                "label": length_label(nominal_input_length),
                "input_length": nominal_input_length,
                "output_length": output_length,
                "status": "failed",
                "completed": completed,
                "failed": failed or max(samples_per_length - completed, 1),
                "ttft_ms": None,
                "mean_ttft_ms": None,
                "median_ttft_ms": None,
                "p90_ttft_ms": None,
                "p99_ttft_ms": None,
                "raw_ttft_ms": [],
                "error": "; ".join(error_messages)
                or (
                    f"completed={completed}, failed={failed}; expected "
                    f"completed={samples_per_length}, failed=0"
                ),
            }
            continue
        if int(data.get("num_prompts", 0)) != samples_per_length:
            raise ValueError(f"unexpected num_prompts in {path}")
        if int(data.get("max_concurrency", 0)) != 1:
            raise ValueError(
                f"single-request TTFT requires max_concurrency=1 in {path}"
            )

        input_lens = data.get("input_lens")
        output_lens = data.get("output_lens")
        ttfts = data.get("ttfts")
        if (
            not isinstance(input_lens, list)
            or len(input_lens) != samples_per_length
            or {int(value) for value in input_lens} != {nominal_input_length}
        ):
            raise ValueError(f"unexpected input_lens in {path}: {input_lens!r}")
        if (
            not isinstance(output_lens, list)
            or len(output_lens) != samples_per_length
            or {int(value) for value in output_lens} != {output_length}
        ):
            raise ValueError(f"unexpected output_lens in {path}: {output_lens!r}")
        if not isinstance(ttfts, list) or len(ttfts) != samples_per_length:
            raise ValueError(f"unexpected ttfts in {path}: {ttfts!r}")
        raw_ttft_ms = [
            finite_float(value, "ttfts", path) * 1000.0 for value in ttfts
        ]
        if any(value < 0 for value in raw_ttft_ms):
            raise ValueError(f"TTFT values must be non-negative in {path}")

        median_ttft_ms = finite_float(data["median_ttft_ms"], "median_ttft_ms", path)
        raw_median_ttft_ms = statistics.median(raw_ttft_ms)
        if not math.isclose(
            median_ttft_ms,
            raw_median_ttft_ms,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"median_ttft_ms in {path} does not match ttfts seconds: "
                f"{median_ttft_ms} != {raw_median_ttft_ms}"
            )
        records[nominal_input_length] = {
            "file": path.name,
            "label": length_label(nominal_input_length),
            "input_length": nominal_input_length,
            "output_length": output_length,
            "status": "success",
            "completed": completed,
            "failed": failed,
            "ttft_ms": median_ttft_ms,
            "mean_ttft_ms": finite_float(data["mean_ttft_ms"], "mean_ttft_ms", path),
            "median_ttft_ms": median_ttft_ms,
            "p90_ttft_ms": finite_float(data["p90_ttft_ms"], "p90_ttft_ms", path),
            "p99_ttft_ms": finite_float(data["p99_ttft_ms"], "p99_ttft_ms", path),
            "raw_ttft_ms": raw_ttft_ms,
            "error": "",
        }

    successful_lengths = [
        value
        for value in expected_input_lengths
        if records[value]["status"] == "success"
    ]
    failed_lengths = [
        value
        for value in expected_input_lengths
        if records[value]["status"] == "failed"
    ]
    if not failed_lengths:
        status = "success"
    elif successful_lengths:
        status = "partial"
    else:
        status = "failed"

    return {
        "schema_version": 2,
        "status": status,
        "benchmark": {
            "benchmark_config": benchmark_config,
            "concurrency": 1,
            "input_lengths": expected_input_lengths,
            "output_length": output_length,
            "samples_per_length": samples_per_length,
            "statistic": "median_ttft_ms",
        },
        "successful_input_lengths": successful_lengths,
        "failed_input_lengths": failed_lengths,
        "results": [records[value] for value in expected_input_lengths],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--benchmark-config", required=True)
    parser.add_argument(
        "--input-lengths", nargs="+", type=positive_int, required=True
    )
    parser.add_argument("--output-length", type=positive_int, default=1)
    parser.add_argument("--samples-per-length", type=positive_int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate(
        args.result_dir,
        args.benchmark_config,
        args.input_lengths,
        args.output_length,
        args.samples_per_length,
    )
    summary_path = args.result_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote single-request TTFT summary: {summary_path}")


if __name__ == "__main__":
    main()
