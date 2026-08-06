#!/usr/bin/env python3

"""Validate and summarize the deterministic SPEED-Bench mixed workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


COMPONENTS = {
    "throughput": {"concurrency": 8},
    "serial_ttft": {"concurrency": 1},
}
BENCHMARK_CONFIGS = {"dp8", "pcp8"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def finite_float(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def percentile(values: list[float], percent: float) -> float:
    """Match NumPy's default linear percentile without depending on NumPy."""
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    manifest = load_json(manifest_path)
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported dataset manifest schema: {manifest_path}")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset manifest is missing the dataset object")
    dataset_path = manifest_path.parent / str(dataset.get("path", ""))
    if not dataset_path.is_file():
        raise ValueError(f"dataset file is missing: {dataset_path}")
    expected_sha256 = str(dataset.get("sha256", ""))
    actual_sha256 = file_sha256(dataset_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"dataset SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    input_tokens = dataset.get("input_tokens")
    if not isinstance(input_tokens, list) or not input_tokens:
        raise ValueError("dataset manifest has no input token lengths")
    normalized_lengths = [int(value) for value in input_tokens]
    if any(value <= 0 for value in normalized_lengths):
        raise ValueError("dataset input token lengths must be positive")
    if int(dataset.get("requests", 0)) != len(normalized_lengths):
        raise ValueError("dataset request count does not match input token lengths")
    if int(dataset.get("total_input_tokens", 0)) != sum(normalized_lengths):
        raise ValueError("dataset total input tokens does not match its lengths")
    dataset["input_tokens"] = normalized_lengths
    return manifest, dataset_path


def failed_component(error: str, result_path: Path | None) -> dict[str, Any]:
    return {
        "status": "failed",
        "result_path": str(result_path) if result_path is not None else "",
        "completed": 0,
        "failed": 0,
        "error": error,
    }


def summarize_component(
    *,
    name: str,
    requested: bool,
    runner_status: str,
    result_path: Path | None,
    expected_input_lengths: list[int],
    output_length: int,
) -> dict[str, Any]:
    if not requested:
        return {"status": "not-run"}
    if runner_status not in {"success", "failed"}:
        return failed_component(
            f"invalid runner status {runner_status!r}", result_path
        )
    if result_path is None or not result_path.is_file():
        return failed_component("vLLM result file is missing", result_path)

    try:
        data = load_json(result_path)
        request_count = len(expected_input_lengths)
        completed = int(data.get("completed", 0))
        failed = int(data.get("failed", 0))
        if runner_status != "success":
            errors = data.get("errors")
            messages = (
                list(dict.fromkeys(str(value) for value in errors if str(value)))
                if isinstance(errors, list)
                else []
            )
            return {
                **failed_component(
                    "; ".join(messages) or "vLLM benchmark command failed",
                    result_path,
                ),
                "completed": completed,
                "failed": failed,
            }
        if completed != request_count or failed != 0:
            raise ValueError(
                f"completed={completed}, failed={failed}; expected "
                f"completed={request_count}, failed=0"
            )
        if int(data.get("num_prompts", 0)) != request_count:
            raise ValueError("num_prompts does not match the dataset")
        expected_concurrency = int(COMPONENTS[name]["concurrency"])
        if int(data.get("max_concurrency", 0)) != expected_concurrency:
            raise ValueError(
                f"max_concurrency must be {expected_concurrency} for {name}"
            )

        input_lens = data.get("input_lens")
        output_lens = data.get("output_lens")
        ttfts = data.get("ttfts")
        if not isinstance(input_lens, list) or len(input_lens) != request_count:
            raise ValueError("input_lens does not match the dataset request count")
        actual_input_lengths = [int(value) for value in input_lens]
        if Counter(actual_input_lengths) != Counter(expected_input_lengths):
            raise ValueError("input_lens does not match the dataset manifest")
        if (
            not isinstance(output_lens, list)
            or len(output_lens) != request_count
            or {int(value) for value in output_lens} != {output_length}
        ):
            raise ValueError("output_lens does not match the configured output length")
        if not isinstance(ttfts, list) or len(ttfts) != request_count:
            raise ValueError("ttfts does not match the dataset request count")
        raw_ttft_ms = [finite_float(value, "ttfts") * 1000.0 for value in ttfts]
        if any(value < 0 for value in raw_ttft_ms):
            raise ValueError("TTFT values must be non-negative")

        duration = finite_float(data.get("duration"), "duration")
        if duration <= 0:
            raise ValueError("duration must be positive")
        total_input_tokens = int(data.get("total_input_tokens", 0))
        total_output_tokens = int(data.get("total_output_tokens", 0))
        if total_input_tokens != sum(expected_input_lengths):
            raise ValueError("total_input_tokens does not match the dataset manifest")
        if total_output_tokens != request_count * output_length:
            raise ValueError("total_output_tokens does not match the expected output")

        calculated_metrics = {
            "mean_ttft_ms": statistics.fmean(raw_ttft_ms),
            "median_ttft_ms": statistics.median(raw_ttft_ms),
            "p90_ttft_ms": percentile(raw_ttft_ms, 90),
            "p99_ttft_ms": percentile(raw_ttft_ms, 99),
        }
        for field, calculated in calculated_metrics.items():
            reported = finite_float(data.get(field), field)
            if not math.isclose(reported, calculated, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(
                    f"{field} does not match raw ttfts: {reported} != {calculated}"
                )

        observations = [
            {"input_tokens": input_length, "ttft_ms": ttft_ms}
            for input_length, ttft_ms in sorted(
                zip(actual_input_lengths, raw_ttft_ms, strict=True)
            )
        ]
        return {
            "status": "success",
            "result_path": str(result_path),
            "completed": completed,
            "failed": failed,
            "duration_s": duration,
            "configured_max_concurrency": expected_concurrency,
            "observed_max_concurrent_requests": int(
                data.get("max_concurrent_requests", 0)
            ),
            "request_throughput": finite_float(
                data.get("request_throughput"), "request_throughput"
            ),
            "input_token_throughput": total_input_tokens / duration,
            "output_token_throughput": finite_float(
                data.get("output_throughput"), "output_throughput"
            ),
            "total_token_throughput": finite_float(
                data.get("total_token_throughput"), "total_token_throughput"
            ),
            **calculated_metrics,
            "observations": observations,
            "error": "",
        }
    except (KeyError, TypeError, ValueError) as error:
        return failed_component(str(error), result_path)


def aggregate(
    *,
    manifest_path: Path,
    mode: str,
    throughput_result: Path | None,
    throughput_status: str,
    ttft_result: Path | None,
    ttft_status: str,
    benchmark_config: str = "dp8",
    output_length: int = 1,
) -> dict[str, Any]:
    if mode not in {"all", "throughput", "ttft"}:
        raise ValueError(f"invalid benchmark mode: {mode!r}")
    if benchmark_config not in BENCHMARK_CONFIGS:
        supported = ", ".join(sorted(BENCHMARK_CONFIGS))
        raise ValueError(
            f"benchmark_config must be one of {supported}, got "
            f"{benchmark_config!r}"
        )
    manifest, dataset_path = load_manifest(manifest_path)
    dataset = manifest["dataset"]
    expected_input_lengths = list(dataset["input_tokens"])
    requested = {
        "throughput": mode in {"all", "throughput"},
        "serial_ttft": mode in {"all", "ttft"},
    }
    components = {
        "throughput": summarize_component(
            name="throughput",
            requested=requested["throughput"],
            runner_status=throughput_status,
            result_path=throughput_result,
            expected_input_lengths=expected_input_lengths,
            output_length=output_length,
        ),
        "serial_ttft": summarize_component(
            name="serial_ttft",
            requested=requested["serial_ttft"],
            runner_status=ttft_status,
            result_path=ttft_result,
            expected_input_lengths=expected_input_lengths,
            output_length=output_length,
        ),
    }
    requested_statuses = [
        components[name]["status"] for name, enabled in requested.items() if enabled
    ]
    successful = requested_statuses.count("success")
    if successful == len(requested_statuses):
        status = "success"
    elif successful:
        status = "partial"
    else:
        status = "failed"

    return {
        "schema_version": 1,
        "status": status,
        "mode": mode,
        "benchmark": {
            "benchmark_config": benchmark_config,
            "workload": "speed_bench_mix",
            "source": str(manifest.get("source", "")),
            "source_revision": str(manifest.get("source_revision", "")),
            "dataset_path": str(dataset_path),
            "dataset_sha256": str(dataset["sha256"]),
            "num_prompts": len(expected_input_lengths),
            "output_length": output_length,
            "total_input_tokens": sum(expected_input_lengths),
            "min_input_tokens": min(expected_input_lengths),
            "max_input_tokens": max(expected_input_lengths),
            "mean_input_tokens": statistics.fmean(expected_input_lengths),
            "input_tokens": expected_input_lengths,
        },
        **components,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "throughput", "ttft"), required=True)
    parser.add_argument("--throughput-result", type=Path)
    parser.add_argument(
        "--throughput-status", choices=("success", "failed"), default="failed"
    )
    parser.add_argument("--ttft-result", type=Path)
    parser.add_argument("--ttft-status", choices=("success", "failed"), default="failed")
    parser.add_argument("--benchmark-config", default="dp8")
    parser.add_argument("--output-length", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_length <= 0:
        raise SystemExit("--output-length must be positive")
    summary = aggregate(
        manifest_path=args.manifest,
        mode=args.mode,
        throughput_result=args.throughput_result,
        throughput_status=args.throughput_status,
        ttft_result=args.ttft_result,
        ttft_status=args.ttft_status,
        benchmark_config=args.benchmark_config,
        output_length=args.output_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote SPEED-Bench mixed-workload summary: {args.output}")
    print(f"SPEED-Bench mixed-workload status: {summary['status']}")


if __name__ == "__main__":
    main()
