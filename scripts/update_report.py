#!/usr/bin/env python3

"""Record a benchmark attempt and regenerate the local throughput report."""

from __future__ import annotations

import argparse
import csv
import fcntl
import html
import ipaddress
import io
import json
import math
import os
import re
import socket
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 7
README_START = "<!-- BENCHMARK_REPORT_START -->"
README_END = "<!-- BENCHMARK_REPORT_END -->"
LEGACY_DECODE_THROUGHPUT_FIELD = "decode_legacy_peak_output_throughput"
LEGACY_DECODE_TPOT_FIELD = "decode_legacy_min_tpot_ms"
BENCHMARK_CONFIGS = {
    "dp8": {"label": "DP8", "color": "#1570ef"},
    "pcp8": {"label": "PCP8", "color": "#7a5af8"},
}
CSV_FIELDS = (
    "run_id",
    "benchmark_config",
    "status",
    "decode_status",
    "started_at",
    "completed_at",
    "machine_ip",
    "model",
    "input_length",
    "output_length",
    "best_total_token_throughput",
    "best_request_throughput",
    "best_concurrency",
    "mean_ttft_ms",
    "p99_ttft_ms",
    LEGACY_DECODE_THROUGHPUT_FIELD,
    LEGACY_DECODE_TPOT_FIELD,
    "decode_window_p50_throughput",
    "decode_peak_active_tpot_p50_ms",
    "torchtpu_vllm_revision",
    "torch_tpu_revision",
    "torch_tpu_version",
    "summary_path",
)


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Append one benchmark attempt and regenerate reports."
    )
    parser.add_argument("--project-root", type=Path, default=script_root)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--decode-summary", type=Path)
    parser.add_argument("--ttft-summary", type=Path)
    parser.add_argument(
        "--status",
        choices=("success", "failed", "not-run"),
        default="success",
        help="Prefill benchmark status; failed throughput is recorded as -1.",
    )
    parser.add_argument(
        "--decode-status",
        choices=("success", "failed", "not-run"),
        help=(
            "Decode benchmark status; defaults to success when --decode-summary "
            "is supplied and not-run otherwise."
        ),
    )
    parser.add_argument(
        "--ttft-status",
        choices=("success", "partial", "failed", "not-run"),
        help=(
            "Single-request prefill TTFT status; defaults to success when "
            "--ttft-summary is supplied and not-run otherwise."
        ),
    )
    parser.add_argument("--benchmark-config")
    parser.add_argument("--input-length", type=int)
    parser.add_argument("--output-length", type=int)
    parser.add_argument("--model")
    parser.add_argument("--display-limit", type=int, default=30)
    parser.add_argument("--table-limit", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def finite_float(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def optional_finite_float(value: Any, field: str) -> float | None:
    if value is None or value == "":
        return None
    return finite_float(value, field)


def positive_int(value: Any, field: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be positive, got {value!r}")
    return result


def normalized_ip_address(value: Any, field: str = "machine_ip") -> str:
    if value is None or value == "":
        return ""
    try:
        return str(ipaddress.ip_address(str(value)))
    except ValueError as error:
        message = f"{field} must be a valid IP address, got {value!r}"
        raise ValueError(message) from error


def detect_machine_ip() -> str:
    configured = os.environ.get("MACHINE_IP")
    if configured:
        return normalized_ip_address(configured, "MACHINE_IP")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(("1.1.1.1", 9))
            return normalized_ip_address(connection.getsockname()[0])
    except OSError:
        try:
            return normalized_ip_address(socket.gethostbyname(socket.gethostname()))
        except OSError as error:
            raise ValueError(
                "could not determine the machine IP; set MACHINE_IP explicitly"
            ) from error


def iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds")


def benchmark_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.strptime(value, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None
    return iso_utc(parsed)


def run_id_timestamp(run_id: str) -> str | None:
    match = re.search(r"(\d{8}T\d{6}Z)$", run_id)
    if match is None:
        return None
    try:
        parsed = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
            tzinfo=UTC
        )
    except ValueError:
        return None
    return iso_utc(parsed)


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def infer_uniform_length(detail: dict[str, Any], key: str) -> int | None:
    values = detail.get(key)
    if not isinstance(values, list) or not values:
        return None
    lengths = {int(value) for value in values}
    if len(lengths) != 1:
        return None
    return lengths.pop()


def normalize_benchmark_config(value: Any) -> str:
    config = str(value or "").strip().lower()
    if config not in BENCHMARK_CONFIGS:
        supported = ", ".join(BENCHMARK_CONFIGS)
        raise ValueError(
            f"benchmark_config must be one of {supported}, got {value!r}"
        )
    return config


def infer_legacy_benchmark_config(run: dict[str, Any]) -> str:
    searchable = " ".join(
        str(run.get(field, "")) for field in ("run_id", "summary_path")
    ).lower()
    return "pcp8" if "pcp" in searchable else "dp8"


def config_label(config: str) -> str:
    return str(BENCHMARK_CONFIGS[config]["label"])


def decode_metrics(
    summary: dict[str, Any],
    benchmark_config: str,
    decode_summary: dict[str, Any] | None = None,
) -> tuple[float | None, float | None]:
    if benchmark_config != "dp8":
        return None, None

    decode = (
        decode_summary
        if decode_summary is not None
        else summary.get("decode_sliding_window")
    )
    if not isinstance(decode, dict):
        return None, None

    aggregate = decode.get("aggregate")
    if not isinstance(aggregate, dict):
        return None, None
    throughput = aggregate.get("throughput_peak_active_p50_tok_s")
    peak_active_tpot = aggregate.get("tpot_peak_active_p50_ms")
    window_p50_throughput = optional_finite_float(
        throughput.get("avg") if isinstance(throughput, dict) else None,
        "decode_window_p50_throughput",
    )
    peak_active_tpot_p50_ms = optional_finite_float(
        peak_active_tpot.get("avg")
        if isinstance(peak_active_tpot, dict)
        else None,
        "decode_peak_active_tpot_p50_ms",
    )
    return window_p50_throughput, peak_active_tpot_p50_ms


def prefill_ttft_results(
    summary: dict[str, Any],
    benchmark_config: str,
) -> list[dict[str, Any]]:
    benchmark = summary.get("benchmark")
    results = summary.get("results")
    if not isinstance(benchmark, dict) or not isinstance(results, list) or not results:
        raise ValueError("invalid single-request TTFT summary")
    summary_config = normalize_benchmark_config(benchmark.get("benchmark_config"))
    if summary_config != benchmark_config:
        raise ValueError("TTFT summary benchmark_config does not match report record")
    if positive_int(benchmark.get("concurrency"), "TTFT concurrency") != 1:
        raise ValueError("single-request TTFT summary must use concurrency 1")

    normalized = []
    seen_lengths: set[int] = set()
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("TTFT summary results must contain objects")
        input_length = positive_int(item.get("input_length"), "TTFT input_length")
        if input_length in seen_lengths:
            raise ValueError(f"duplicate TTFT input length: {input_length}")
        seen_lengths.add(input_length)
        result_status = str(item.get("status") or "").strip().lower()
        completed = int(item.get("completed", 0))
        failed = int(item.get("failed", 0))
        if completed < 0 or failed < 0:
            raise ValueError("TTFT completed/failed counts must be non-negative")
        if result_status not in {"success", "failed"}:
            result_status = "failed" if failed or completed == 0 else "success"
        if result_status == "success":
            if completed <= 0 or failed != 0:
                raise ValueError("successful TTFT result has invalid request counts")
            ttft_ms: float | None = finite_float(
                item.get("ttft_ms"),
                "TTFT ttft_ms",
            )
            if ttft_ms < 0:
                raise ValueError("TTFT latency must be non-negative")
        else:
            ttft_ms = None
        normalized.append(
            {
                "label": str(item.get("label") or input_length),
                "input_length": input_length,
                "output_length": positive_int(
                    item.get("output_length"), "TTFT output_length"
                ),
                "completed": completed,
                "failed": failed,
                "status": result_status,
                "ttft_ms": ttft_ms,
                "error": str(item.get("error") or ""),
            }
        )
    return sorted(normalized, key=lambda item: item["input_length"])


def build_record(
    *,
    project_root: Path,
    run_dir: Path,
    summary_path: Path | None,
    input_length: int | None,
    output_length: int | None,
    model: str | None,
    benchmark_config: str | None,
    decode_summary_path: Path | None = None,
    ttft_summary_path: Path | None = None,
    status: str = "success",
    decode_status: str = "not-run",
    ttft_status: str = "not-run",
) -> dict[str, Any]:
    if status not in {"success", "failed", "not-run"}:
        raise ValueError(f"invalid benchmark status: {status!r}")
    if decode_status not in {"success", "failed", "not-run"}:
        raise ValueError(f"invalid decode status: {decode_status!r}")
    if ttft_status not in {"success", "partial", "failed", "not-run"}:
        raise ValueError(f"invalid TTFT status: {ttft_status!r}")
    if status == "success" and summary_path is None:
        raise ValueError("a successful benchmark requires --summary")

    summary: dict[str, Any] = {}
    best: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    benchmark: dict[str, Any] = {}
    if status == "success":
        assert summary_path is not None
        summary = load_json(summary_path)
        loaded_best = summary.get("best")
        loaded_results = summary.get("results")
        if (
            not isinstance(loaded_best, dict)
            or not isinstance(loaded_results, list)
            or not loaded_results
            or not all(isinstance(item, dict) for item in loaded_results)
        ):
            raise ValueError(f"invalid benchmark summary: {summary_path}")
        best = loaded_best
        results = loaded_results

        failed = sum(int(item.get("failed", 0)) for item in results)
        if failed:
            raise ValueError(
                f"refusing to record a benchmark with {failed} failed requests"
            )

        best_filename = best.get("file")
        if isinstance(best_filename, str):
            detail_path = summary_path.parent / best_filename
            if detail_path.is_file():
                detail = load_json(detail_path)

        loaded_benchmark = summary.get("benchmark")
        if isinstance(loaded_benchmark, dict):
            benchmark = loaded_benchmark

    if not isinstance(benchmark, dict):
        benchmark = {}
    benchmark_config = normalize_benchmark_config(
        benchmark_config
        or benchmark.get("benchmark_config")
        or infer_legacy_benchmark_config(
            {"run_id": run_dir.name, "summary_path": str(summary_path)}
        )
    )
    if benchmark_config != "dp8" and decode_status != "not-run":
        raise ValueError("decode status can only be recorded on the dp8 record")

    if ttft_status in {"success", "partial"} or (
        ttft_status == "failed" and ttft_summary_path is not None
    ):
        if ttft_summary_path is None:
            raise ValueError("successful/partial TTFT benchmark requires --ttft-summary")
        single_request_ttft_results = prefill_ttft_results(
            load_json(ttft_summary_path),
            benchmark_config,
        )
        result_statuses = {
            result["status"] for result in single_request_ttft_results
        }
        if result_statuses == {"success"}:
            derived_ttft_status = "success"
        elif "success" in result_statuses:
            derived_ttft_status = "partial"
        else:
            derived_ttft_status = "failed"
        if ttft_status != derived_ttft_status:
            raise ValueError(
                "TTFT status does not match per-length results: "
                f"{ttft_status!r} != {derived_ttft_status!r}"
            )
    else:
        single_request_ttft_results = []

    if decode_status == "success":
        if decode_summary_path is None:
            raise ValueError("successful decode requires --decode-summary")
        decode_summary = load_json(decode_summary_path)
        (
            decode_window_p50_throughput,
            decode_peak_active_tpot_p50_ms,
        ) = decode_metrics(summary, benchmark_config, decode_summary)
        if decode_window_p50_throughput is None:
            raise ValueError("decode summary does not contain throughput")
    elif decode_status == "failed":
        decode_window_p50_throughput = -1.0
        decode_peak_active_tpot_p50_ms = None
    else:
        decode_window_p50_throughput = None
        decode_peak_active_tpot_p50_ms = None

    input_length = (
        input_length
        or benchmark.get("input_length")
        or infer_uniform_length(detail, "input_lens")
    )
    output_length = (
        output_length
        or benchmark.get("output_length")
        or infer_uniform_length(detail, "output_lens")
    )
    model = model or benchmark.get("model") or detail.get("model_id") or "unknown"
    input_length = positive_int(input_length, "input_length")
    output_length = positive_int(output_length, "output_length")

    metadata_path = run_dir / "run_metadata.json"
    metadata_exists = metadata_path.is_file()
    metadata = load_json(metadata_path) if metadata_exists else {}
    configured_machine_ip = os.environ.get("MACHINE_IP")
    if configured_machine_ip:
        machine_ip = normalized_ip_address(configured_machine_ip, "MACHINE_IP")
    elif "machine_ip" in metadata:
        machine_ip = normalized_ip_address(metadata["machine_ip"])
    elif metadata_exists:
        # Legacy run metadata predates machine IP capture. Do not guess which
        # machine produced an existing historical result.
        machine_ip = ""
    else:
        # Manual bench_all.sh runs do not create run_metadata.json, so capture
        # the address while the report for that run is being recorded.
        machine_ip = detect_machine_ip()
    run_id = run_dir.name
    started_at = metadata.get("started_at") or run_id_timestamp(run_id)
    if not isinstance(started_at, str):
        if summary_path is not None and summary_path.exists():
            started_at = iso_utc(
                datetime.fromtimestamp(summary_path.stat().st_mtime, tz=UTC)
            )
        else:
            started_at = iso_utc(datetime.now(tz=UTC))
    if summary_path is not None and summary_path.exists():
        fallback_completed_at = iso_utc(
            datetime.fromtimestamp(summary_path.stat().st_mtime, tz=UTC)
        )
    else:
        fallback_completed_at = iso_utc(datetime.now(tz=UTC))
    completed_at = benchmark_timestamp(detail.get("date")) or fallback_completed_at

    concurrency_results = []
    if status == "success":
        for item in sorted(results, key=lambda row: int(row["concurrency"])):
            concurrency_results.append(
                {
                    "concurrency": positive_int(item["concurrency"], "concurrency"),
                    "total_token_throughput": finite_float(
                        item["total_token_throughput"], "total_token_throughput"
                    ),
                    "request_throughput": finite_float(
                        item["request_throughput"], "request_throughput"
                    ),
                    "mean_ttft_ms": finite_float(
                        item["mean_ttft_ms"], "mean_ttft_ms"
                    ),
                    "p99_ttft_ms": finite_float(
                        item["p99_ttft_ms"], "p99_ttft_ms"
                    ),
                }
            )

    if status == "success":
        best_total_token_throughput: float | None = finite_float(
            best["total_token_throughput"], "best_total_token_throughput"
        )
        best_request_throughput: float | None = finite_float(
            best["request_throughput"], "best_request_throughput"
        )
        best_concurrency: int | None = positive_int(
            best["concurrency"], "best_concurrency"
        )
        mean_ttft_ms: float | None = finite_float(
            best["mean_ttft_ms"], "mean_ttft_ms"
        )
        p99_ttft_ms: float | None = finite_float(
            best["p99_ttft_ms"], "p99_ttft_ms"
        )
    elif status == "failed":
        best_total_token_throughput = -1.0
        best_request_throughput = -1.0
        best_concurrency = None
        mean_ttft_ms = None
        p99_ttft_ms = None
    else:
        best_total_token_throughput = None
        best_request_throughput = None
        best_concurrency = None
        mean_ttft_ms = None
        p99_ttft_ms = None

    return {
        "run_id": run_id,
        "benchmark_config": benchmark_config,
        "status": status,
        "decode_status": decode_status,
        "started_at": started_at,
        "completed_at": completed_at,
        "machine_ip": machine_ip,
        "model": str(model),
        "input_length": input_length,
        "output_length": output_length,
        "best_total_token_throughput": best_total_token_throughput,
        "best_request_throughput": best_request_throughput,
        "best_concurrency": best_concurrency,
        "mean_ttft_ms": mean_ttft_ms,
        "p99_ttft_ms": p99_ttft_ms,
        LEGACY_DECODE_THROUGHPUT_FIELD: None,
        LEGACY_DECODE_TPOT_FIELD: None,
        "decode_window_p50_throughput": decode_window_p50_throughput,
        "decode_peak_active_tpot_p50_ms": decode_peak_active_tpot_p50_ms,
        "prefill_ttft_status": ttft_status,
        "single_request_ttft_results": single_request_ttft_results,
        "torchtpu_vllm_revision": str(
            metadata.get("torchtpu_vllm_revision", "unknown")
        ),
        # Source-backed runs recorded a Git revision. Pip-backed runs have no
        # source checkout, so retain this legacy field as an empty value for
        # history/CSV compatibility and use the package version for display.
        "torch_tpu_revision": str(metadata.get("torch_tpu_revision", "")),
        "torch_tpu_version": str(metadata.get("torch_tpu_version", "unknown")),
        "summary_path": (
            relative_path(summary_path, project_root)
            if summary_path is not None
            else ""
        ),
        "ttft_summary_path": (
            relative_path(ttft_summary_path, project_root)
            if ttft_summary_path is not None
            else ""
        ),
        "concurrency_results": concurrency_results,
    }


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    history = load_json(path)
    schema_version = history.get("schema_version")
    if schema_version not in (1, 2, 3, 4, 5, 6, SCHEMA_VERSION):
        raise ValueError(f"unsupported history schema in {path}")
    runs = history.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"history runs must be a list in {path}")
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"history run {index} must be an object in {path}")
        run["machine_ip"] = normalized_ip_address(run.get("machine_ip"))
        run["benchmark_config"] = normalize_benchmark_config(
            run.get("benchmark_config") or infer_legacy_benchmark_config(run)
        )
        status = str(run.get("status") or "").strip().lower()
        if status not in {"success", "failed", "not-run"}:
            status = (
                "failed"
                if run.get("best_total_token_throughput") == -1
                else "success"
            )
        run["status"] = status
        legacy_decode_throughput = run.get(LEGACY_DECODE_THROUGHPUT_FIELD)
        if legacy_decode_throughput is None:
            legacy_decode_throughput = run.get("decode_peak_output_throughput")
        legacy_decode_tpot = run.get(LEGACY_DECODE_TPOT_FIELD)
        if legacy_decode_tpot is None:
            legacy_decode_tpot = run.get("decode_min_tpot_ms")
        run[LEGACY_DECODE_THROUGHPUT_FIELD] = optional_finite_float(
            legacy_decode_throughput,
            LEGACY_DECODE_THROUGHPUT_FIELD,
        )
        run[LEGACY_DECODE_TPOT_FIELD] = optional_finite_float(
            legacy_decode_tpot,
            LEGACY_DECODE_TPOT_FIELD,
        )
        run.pop("decode_peak_output_throughput", None)
        run.pop("decode_min_tpot_ms", None)
        run["decode_window_p50_throughput"] = optional_finite_float(
            run.get("decode_window_p50_throughput"),
            "decode_window_p50_throughput",
        )
        run["decode_peak_active_tpot_p50_ms"] = optional_finite_float(
            run.get("decode_peak_active_tpot_p50_ms"),
            "decode_peak_active_tpot_p50_ms",
        )
        decode_status = str(run.get("decode_status") or "").strip().lower()
        if decode_status not in {"success", "failed", "not-run"}:
            decode_throughput = run.get("decode_window_p50_throughput")
            legacy_decode_throughput = run.get(LEGACY_DECODE_THROUGHPUT_FIELD)
            if decode_throughput == -1:
                decode_status = "failed"
            elif (
                decode_throughput is not None
                or legacy_decode_throughput is not None
            ):
                decode_status = "success"
            else:
                decode_status = "not-run"
        run["decode_status"] = decode_status
        ttft_status = str(run.get("prefill_ttft_status") or "").strip().lower()
        ttft_results = run.get("single_request_ttft_results")
        if not isinstance(ttft_results, list):
            ttft_results = []
        for result in ttft_results:
            if not isinstance(result, dict):
                continue
            result_status = str(result.get("status") or "").strip().lower()
            if result_status not in {"success", "failed"}:
                result_status = (
                    "failed"
                    if int(result.get("failed", 0)) > 0
                    or result.get("ttft_ms") is None
                    else "success"
                )
            result["status"] = result_status
            result.setdefault("error", "")
        run["single_request_ttft_results"] = ttft_results
        if ttft_status not in {"success", "partial", "failed", "not-run"}:
            result_statuses = {
                result.get("status")
                for result in ttft_results
                if isinstance(result, dict)
            }
            if result_statuses == {"success"}:
                ttft_status = "success"
            elif "success" in result_statuses:
                ttft_status = "partial"
            elif result_statuses:
                ttft_status = "failed"
            else:
                ttft_status = "not-run"
        run["prefill_ttft_status"] = ttft_status
        run.setdefault("ttft_summary_path", "")
    return runs


def update_history(
    runs: list[dict[str, Any]], record: dict[str, Any]
) -> list[dict[str, Any]]:
    by_key = {
        (str(item["run_id"]), str(item["benchmark_config"])): item
        for item in runs
    }
    key = (str(record["run_id"]), str(record["benchmark_config"]))
    existing = by_key.get(key)
    if not record.get("machine_ip") and existing and existing.get("machine_ip"):
        record = {**record, "machine_ip": existing["machine_ip"]}
    if existing:
        decode_fields = (
            LEGACY_DECODE_THROUGHPUT_FIELD,
            LEGACY_DECODE_TPOT_FIELD,
            "decode_window_p50_throughput",
            "decode_peak_active_tpot_p50_ms",
        )
        record = {
            **record,
            **{
                field: existing.get(field)
                for field in decode_fields
                if record.get("decode_status") != "failed"
                and record.get(field) is None
                and existing.get(field) is not None
            },
        }
        if (
            record.get("decode_status") == "not-run"
            and existing.get("decode_status") in {"success", "failed"}
        ):
            record["decode_status"] = existing["decode_status"]
        if (
            record.get("prefill_ttft_status") == "not-run"
            and existing.get("prefill_ttft_status")
            in {"success", "partial", "failed"}
        ):
            record["prefill_ttft_status"] = existing["prefill_ttft_status"]
            record["single_request_ttft_results"] = existing.get(
                "single_request_ttft_results", []
            )
            record["ttft_summary_path"] = existing.get("ttft_summary_path", "")
    by_key[key] = record
    return sorted(
        by_key.values(),
        key=lambda item: (
            item["completed_at"],
            item["run_id"],
            item["benchmark_config"],
        ),
    )


def human_rate(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def display_time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone(UTC).strftime("%Y-%m-%d %H:%M")


def chart_svg(
    series: list[dict[str, Any]],
    x_labels: list[str],
    *,
    title: str,
    description: str = "Peak total token throughput across recent benchmark runs.",
    id_prefix: str = "throughput",
    width: int = 1000,
    height: int = 390,
    standalone: bool = True,
    value_suffix: str = " tok/s",
) -> str:
    points = [point for item in series for point in item["points"]]
    if not points or not x_labels:
        raise ValueError("cannot render an empty chart")

    left, right, top, bottom = 82, 28, 76, 66
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [finite_float(point["value"], "chart value") for point in points]
    upper = max(values) * 1.12
    if upper <= 0:
        upper = 1.0

    if len(x_labels) == 1:
        x_values = [left + plot_width / 2]
    else:
        x_values = [
            left + index * plot_width / (len(x_labels) - 1)
            for index in range(len(x_labels))
        ]

    prefix = '<?xml version="1.0" encoding="UTF-8"?>\n' if standalone else ""
    parts = [
        prefix,
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" ',
        f'role="img" aria-labelledby="{id_prefix}-title {id_prefix}-desc">',
        f'<title id="{id_prefix}-title">{html.escape(title)}</title>',
        f'<desc id="{id_prefix}-desc">{html.escape(description)}</desc>',
        "<style>",
        ".bg{fill:#fff}.grid{stroke:#d9e2ec;stroke-width:1}.axis{fill:#52606d;",
        "font:13px ui-monospace,SFMono-Regular,Consolas,monospace}",
        ".heading{fill:#102a43;font:600 19px system-ui,sans-serif}",
        ".line{fill:none;stroke-width:4;stroke-linejoin:round;stroke-linecap:round}",
        ".dot{fill:#fff;stroke-width:3}.latest{stroke-width:3}",
        ".legend{fill:#334e68;font:600 13px system-ui,sans-serif}",
        ".value{font:600 14px system-ui,sans-serif}",
        "</style>",
        f'<rect class="bg" width="{width}" height="{height}" rx="12"/>',
        f'<text class="heading" x="{left}" y="31">{html.escape(title)}</text>',
    ]

    legend_x = left
    for item in series:
        color = html.escape(str(item["color"]))
        label = html.escape(str(item["label"]))
        parts.append(
            f'<line x1="{legend_x}" y1="52" x2="{legend_x + 28}" y2="52" '
            f'stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
        )
        parts.append(
            f'<text class="legend" x="{legend_x + 37}" y="57">{label}</text>'
        )
        legend_x += 130

    for tick in range(5):
        ratio = tick / 4
        y = top + plot_height * (1 - ratio)
        tick_value = upper * ratio
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" '
            f'x2="{left + plot_width}" y2="{y:.2f}"/>'
        )
        parts.append(
            f'<text class="axis" text-anchor="end" x="{left - 11}" '
            f'y="{y + 4:.2f}">{html.escape(human_rate(tick_value))}</text>'
        )

    tick_count = min(6, len(x_labels))
    if tick_count == 1:
        tick_indices = [0]
    else:
        tick_indices = sorted(
            {
                round(index * (len(x_labels) - 1) / (tick_count - 1))
                for index in range(tick_count)
            }
        )
    for index in tick_indices:
        label = html.escape(str(x_labels[index]))
        anchor = (
            "end"
            if len(x_labels) > 1 and index == len(x_labels) - 1
            else "middle"
        )
        parts.append(
            f'<text class="axis" text-anchor="{anchor}" x="{x_values[index]:.2f}" '
            f'y="{top + plot_height + 28}">{label}</text>'
        )

    for series_index, item in enumerate(series):
        item_points = item["points"]
        color = html.escape(str(item["color"]))
        coordinates = []
        rendered_points = []
        for point in item_points:
            x_index = int(point["x_index"])
            if x_index < 0 or x_index >= len(x_values):
                raise ValueError(f"chart x_index out of range: {x_index}")
            value = finite_float(point["value"], "chart value")
            x = x_values[x_index]
            y = top + plot_height * (1 - value / upper)
            coordinates.append(f"{x:.2f},{y:.2f}")
            rendered_points.append((point, x, y, value))
        if len(coordinates) > 1:
            parts.append(
                f'<polyline class="line" stroke="{color}" '
                f'points="{" ".join(coordinates)}"/>'
            )
        for index, (point, x, y, _) in enumerate(rendered_points):
            latest = index == len(rendered_points) - 1
            fill = color if latest else "#fff"
            css_class = "dot latest" if latest else "dot"
            tooltip = html.escape(str(point["tooltip"]))
            parts.append(
                f'<circle class="{css_class}" fill="{fill}" stroke="{color}" '
                f'cx="{x:.2f}" cy="{y:.2f}" r="6">'
                f"<title>{tooltip}</title></circle>"
            )
        if rendered_points:
            _, latest_x, latest_y, latest_value = rendered_points[-1]
            label_y = latest_y - 13 if series_index % 2 == 0 else latest_y + 23
            label_y = min(top + plot_height - 5, max(top + 14, label_y))
            anchor = "end" if latest_x > width - 120 else "middle"
            parts.append(
                f'<text class="value" fill="{color}" text-anchor="{anchor}" '
                f'x="{latest_x:.2f}" y="{label_y:.2f}">'
                f"{latest_value:,.0f}{html.escape(value_suffix)}</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def empty_chart_svg(
    *,
    title: str,
    description: str,
    id_prefix: str,
    message: str = "No successful benchmark data yet.",
) -> str:
    return "".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>\n',
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 390" ',
            f'role="img" aria-labelledby="{id_prefix}-title {id_prefix}-desc">',
            f'<title id="{id_prefix}-title">{html.escape(title)}</title>',
            f'<desc id="{id_prefix}-desc">{html.escape(description)}</desc>',
            '<rect fill="#fff" width="1000" height="390" rx="12"/>',
            '<text fill="#102a43" font-family="system-ui,sans-serif" ',
            'font-size="19" font-weight="600" x="82" y="45">',
            f"{html.escape(title)}</text>",
            '<text fill="#52606d" font-family="system-ui,sans-serif" ',
            f'font-size="15" x="82" y="100">{html.escape(message)}</text>',
            "</svg>",
        ]
    )


def latest_runs_by_config(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for run in runs:
        latest[run["benchmark_config"]] = run
    return latest


def visible_history_runs(
    runs: list[dict[str, Any]], display_limit: int
) -> list[dict[str, Any]]:
    run_times: dict[str, str] = {}
    for run in runs:
        run_id = str(run["run_id"])
        run_times[run_id] = max(run_times.get(run_id, ""), run["completed_at"])
    visible_ids = {
        run_id
        for run_id, _ in sorted(run_times.items(), key=lambda item: (item[1], item[0]))[
            -display_limit:
        ]
    }
    return [run for run in runs if run["run_id"] in visible_ids]


def history_chart_data(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    run_times: dict[str, str] = {}
    for run in runs:
        run_id = str(run["run_id"])
        run_times[run_id] = max(run_times.get(run_id, ""), run["completed_at"])
    run_ids = [
        run_id
        for run_id, _ in sorted(run_times.items(), key=lambda item: (item[1], item[0]))
    ]
    index_by_run_id = {run_id: index for index, run_id in enumerate(run_ids)}
    x_labels = [display_time(run_times[run_id])[5:] for run_id in run_ids]
    series = []
    for config, style in BENCHMARK_CONFIGS.items():
        config_runs = [
            run
            for run in runs
            if run["benchmark_config"] == config
            and run.get("status", "success") == "success"
            and run.get("best_total_token_throughput") is not None
        ]
        if not config_runs:
            continue
        series.append(
            {
                "config": config,
                "label": style["label"],
                "color": style["color"],
                "points": [
                    {
                        "x_index": index_by_run_id[run["run_id"]],
                        "value": run["best_total_token_throughput"],
                        "tooltip": (
                            f"{style['label']} · {run['run_id']}: "
                            f"{run['best_total_token_throughput']:,.2f} tok/s "
                            f"at concurrency {run['best_concurrency']}"
                        ),
                    }
                    for run in config_runs
                ],
            }
        )
    return series, x_labels


def decode_history_chart_data(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    decode_runs = [
        run
        for run in runs
        if run["benchmark_config"] == "dp8"
        and run.get("decode_status", "success") == "success"
        and (
            run.get(LEGACY_DECODE_THROUGHPUT_FIELD) is not None
            or run.get("decode_window_p50_throughput") is not None
        )
    ]
    run_ids = [str(run["run_id"]) for run in decode_runs]
    index_by_run_id = {run_id: index for index, run_id in enumerate(run_ids)}
    x_labels = [display_time(run["completed_at"])[5:] for run in decode_runs]
    metric_series = (
        (
            "legacy",
            "Legacy decode peak output",
            "#dc6803",
            LEGACY_DECODE_THROUGHPUT_FIELD,
        ),
        (
            "peak-active-p50",
            "C256 peak-active window P50",
            "#039855",
            "decode_window_p50_throughput",
        ),
    )
    series = []
    for config, label, color, field in metric_series:
        points = [
            {
                "x_index": index_by_run_id[str(run["run_id"])],
                "value": run[field],
                "tooltip": (
                    f"{label} · {run['run_id']}: "
                    f"{run[field]:,.2f} tok/s"
                ),
            }
            for run in decode_runs
            if run.get(field) is not None
        ]
        if points:
            series.append(
                {
                    "config": config,
                    "label": label,
                    "color": color,
                    "points": points,
                }
            )
    return series, x_labels


def concurrency_chart_data(
    latest: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    concurrencies = sorted(
        {
            int(result["concurrency"])
            for run in latest.values()
            for result in run["concurrency_results"]
        }
    )
    index_by_concurrency = {
        concurrency: index for index, concurrency in enumerate(concurrencies)
    }
    series = []
    for config, style in BENCHMARK_CONFIGS.items():
        run = latest.get(config)
        if run is None or not run.get("concurrency_results"):
            continue
        series.append(
            {
                "config": config,
                "label": style["label"],
                "color": style["color"],
                "points": [
                    {
                        "x_index": index_by_concurrency[int(result["concurrency"])],
                        "value": result["total_token_throughput"],
                        "tooltip": (
                            f"{style['label']} concurrency {result['concurrency']}: "
                            f"{result['total_token_throughput']:,.2f} tok/s"
                        ),
                    }
                    for result in run["concurrency_results"]
                ],
            }
        )
    return series, [f"c{concurrency}" for concurrency in concurrencies]


def latest_ttft_runs_by_config(
    runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run.get("prefill_ttft_status", "not-run") != "not-run":
            latest[run["benchmark_config"]] = run
    return latest


def prefill_ttft_chart_data(
    latest: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    input_lengths = sorted(
        {
            int(result["input_length"])
            for run in latest.values()
            for result in run.get("single_request_ttft_results", [])
        }
    )
    index_by_length = {
        input_length: index for index, input_length in enumerate(input_lengths)
    }
    labels_by_length = {
        int(result["input_length"]): str(result["label"])
        for run in latest.values()
        for result in run.get("single_request_ttft_results", [])
    }
    series = []
    for config, style in BENCHMARK_CONFIGS.items():
        run = latest.get(config)
        if run is None or run.get("prefill_ttft_status") not in {
            "success",
            "partial",
        }:
            continue
        results = [
            result
            for result in run.get("single_request_ttft_results", [])
            if result.get("status", "success") == "success"
            and result.get("ttft_ms") is not None
        ]
        if not results:
            continue
        series.append(
            {
                "config": config,
                "label": style["label"],
                "color": style["color"],
                "points": [
                    {
                        "x_index": index_by_length[int(result["input_length"])],
                        "value": result["ttft_ms"],
                        "tooltip": (
                            f"{style['label']} {result['label']} · {run['run_id']}: "
                            f"{result['ttft_ms']:,.2f} ms"
                        ),
                    }
                    for result in results
                ],
            }
        )
    return series, [
        labels_by_length.get(input_length, str(input_length))
        for input_length in input_lengths
    ]


def render_history_json(runs: list[dict[str, Any]]) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": runs[-1]["completed_at"],
        "runs": runs,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def latest_run_payload(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "benchmark_config": run["benchmark_config"],
        "status": run.get("status", "success"),
        "decode_status": run.get("decode_status", "not-run"),
        "completed_at": run["completed_at"],
        "machine_ip": run["machine_ip"],
        "model": run["model"],
        "input_length": run["input_length"],
        "output_length": run["output_length"],
        "total_token_throughput": run["best_total_token_throughput"],
        "request_throughput": run["best_request_throughput"],
        "concurrency": run["best_concurrency"],
        "mean_ttft_ms": run["mean_ttft_ms"],
        "p99_ttft_ms": run["p99_ttft_ms"],
        LEGACY_DECODE_THROUGHPUT_FIELD: run.get(
            LEGACY_DECODE_THROUGHPUT_FIELD
        ),
        LEGACY_DECODE_TPOT_FIELD: run.get(LEGACY_DECODE_TPOT_FIELD),
        "decode_window_p50_throughput": run.get(
            "decode_window_p50_throughput"
        ),
        "decode_peak_active_tpot_p50_ms": run.get(
            "decode_peak_active_tpot_p50_ms"
        ),
        "prefill_ttft_status": run.get("prefill_ttft_status", "not-run"),
        "single_request_ttft_results": run.get(
            "single_request_ttft_results", []
        ),
        "torchtpu_vllm_revision": run["torchtpu_vllm_revision"],
        "torch_tpu_revision": run["torch_tpu_revision"],
        "torch_tpu_version": run["torch_tpu_version"],
    }


def render_latest_json(runs: list[dict[str, Any]]) -> str:
    latest = latest_runs_by_config(runs)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": max(run["completed_at"] for run in latest.values()),
        "benchmarks": {
            config: latest_run_payload(run) for config, run in sorted(latest.items())
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_csv(runs: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for run in runs:
        writer.writerow({field: run.get(field, "") for field in CSV_FIELDS})
    return output.getvalue()


def render_prefill_ttft_csv(runs: list[dict[str, Any]]) -> str:
    fields = (
        "run_id",
        "benchmark_config",
        "benchmark_status",
        "status",
        "completed_at",
        "input_length",
        "input_label",
        "output_length",
        "completed",
        "failed",
        "ttft_ms",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for run in runs:
        ttft_status = run.get("prefill_ttft_status", "not-run")
        results = run.get("single_request_ttft_results", [])
        if results:
            for result in results:
                writer.writerow(
                    {
                        "run_id": run["run_id"],
                        "benchmark_config": run["benchmark_config"],
                        "benchmark_status": ttft_status,
                        "status": result.get("status", "success"),
                        "completed_at": run["completed_at"],
                        "input_length": result["input_length"],
                        "input_label": result["label"],
                        "output_length": result["output_length"],
                        "completed": result["completed"],
                        "failed": result["failed"],
                        "ttft_ms": result["ttft_ms"],
                    }
                )
        elif ttft_status == "failed":
            writer.writerow(
                {
                    "run_id": run["run_id"],
                    "benchmark_config": run["benchmark_config"],
                    "benchmark_status": "failed",
                    "status": "failed",
                    "completed_at": run["completed_at"],
                }
            )
    return output.getvalue()


def report_table_runs(
    runs: list[dict[str, Any]], table_limit: int
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for run in runs:
        run_id = str(run["run_id"])
        row = grouped.setdefault(
            run_id,
            {
                "run_id": run_id,
                "started_at": run.get("started_at") or run["completed_at"],
                "torchtpu_vllm_revision": "unknown",
                "configs": {},
            },
        )
        started_at = run.get("started_at") or run["completed_at"]
        if started_at < row["started_at"]:
            row["started_at"] = started_at
        revision = str(run.get("torchtpu_vllm_revision") or "unknown")
        if revision != "unknown":
            row["torchtpu_vllm_revision"] = revision
        row["configs"][run["benchmark_config"]] = run

    ordered = sorted(
        grouped.values(),
        key=lambda row: (row["started_at"], row["run_id"]),
        reverse=True,
    )
    return ordered[:table_limit]


def table_metric(value: Any, *, decimals: int = 2) -> str:
    if value is None or value == "":
        return "—"
    return f"{finite_float(value, 'table metric'):,.{decimals}f}"


def render_readme_block(runs: list[dict[str, Any]], table_limit: int) -> str:
    latest = latest_runs_by_config(runs)
    latest_ttft = latest_ttft_runs_by_config(runs)
    latest_lines = []
    for config, style in BENCHMARK_CONFIGS.items():
        run = latest.get(config)
        if run is None:
            continue
        if run.get("status") == "failed":
            latest_lines.append(
                f"Latest {style['label']}: **failed "
                f"({run['best_total_token_throughput']:,.2f} total tok/s)** "
                f"(`{run['run_id']}`)."
            )
        elif run.get("status") == "not-run":
            latest_lines.append(
                f"Latest {style['label']}: **not run** (`{run['run_id']}`)."
            )
        else:
            latest_lines.append(
                f"Latest {style['label']}: "
                f"**{run['best_total_token_throughput']:,.2f} total tok/s** "
                f"at concurrency **{run['best_concurrency']}** "
                f"(`{run['run_id']}`)."
            )
    ttft_status_lines = []
    for config, style in BENCHMARK_CONFIGS.items():
        run = latest_ttft.get(config)
        if run is None:
            continue
        status = run.get("prefill_ttft_status", "not-run")
        sample_suffix = ""
        counts_to_labels: dict[int, list[str]] = {}
        if status in {"success", "partial"}:
            for result in run.get("single_request_ttft_results", []):
                samples = int(result.get("completed", 0))
                if result.get("status", "success") != "success":
                    samples = int(result.get("failed", 0))
                if samples > 0:
                    counts_to_labels.setdefault(samples, []).append(
                        str(result["label"])
                    )
        if len(counts_to_labels) == 1:
            samples = next(iter(counts_to_labels))
            sample_word = "sample" if samples == 1 else "samples"
            sample_suffix = f", **{samples} serial {sample_word}/length**"
        elif counts_to_labels:
            sample_groups = ", ".join(
                f"{'/'.join(labels)}={samples}"
                for samples, labels in counts_to_labels.items()
            )
            if sample_groups:
                sample_suffix = (
                    f", **serial samples/length: {sample_groups}**"
                )
        ttft_status_lines.append(
            f"Latest {style['label']} single-request TTFT: "
            f"**{status}**{sample_suffix} (`{run['run_id']}`)."
        )

    ttft_lengths = sorted(
        {
            int(result["input_length"])
            for run in runs
            for result in run.get("single_request_ttft_results", [])
        }
    )
    ttft_labels = {
        int(result["input_length"]): str(result["label"])
        for run in runs
        for result in run.get("single_request_ttft_results", [])
    }

    def ttft_cell(run: dict[str, Any] | None, input_length: int) -> str:
        if run is None:
            return "—"
        result = next(
            (
                item
                for item in run.get("single_request_ttft_results", [])
                if int(item["input_length"]) == input_length
            ),
            None,
        )
        if result is None:
            if run.get("prefill_ttft_status") == "failed":
                return "failed"
            return "—"
        if result.get("status", "success") == "failed":
            return "failed"
        return table_metric(result.get("ttft_ms"))

    ttft_headers = [
        header
        for input_length in ttft_lengths
        for header in (
            f"DP TTFT {ttft_labels.get(input_length, input_length)} (ms)",
            f"PCP TTFT {ttft_labels.get(input_length, input_length)} (ms)",
        )
    ]
    rows = []
    for grouped_run in report_table_runs(runs, table_limit):
        dp_run = grouped_run["configs"].get("dp8")
        pcp_run = grouped_run["configs"].get("pcp8")
        revision = grouped_run["torchtpu_vllm_revision"]
        revision_display = f"`{revision[:12]}`" if revision != "unknown" else "—"
        dp_prefill = dp_run.get("best_total_token_throughput") if dp_run else None
        pcp_prefill = (
            pcp_run.get("best_total_token_throughput") if pcp_run else None
        )
        dp_decode = (
            dp_run.get("decode_window_p50_throughput") if dp_run else None
        )
        dp_tpot_p50 = (
            dp_run.get("decode_peak_active_tpot_p50_ms") if dp_run else None
        )
        legacy_dp_decode = (
            dp_run.get(LEGACY_DECODE_THROUGHPUT_FIELD) if dp_run else None
        )
        legacy_dp_tpot = (
            dp_run.get(LEGACY_DECODE_TPOT_FIELD) if dp_run else None
        )
        decode_status = dp_run.get("decode_status") if dp_run else "not-run"
        if decode_status == "failed":
            dp_decode = -1.0
            dp_tpot_p50 = None
            decode_protocol = "failed"
        elif dp_decode is not None:
            decode_protocol = "C256 peak-active P50"
        elif legacy_dp_decode is not None:
            dp_decode = legacy_dp_decode
            dp_tpot_p50 = legacy_dp_tpot
            decode_protocol = "legacy peak/min"
        else:
            decode_protocol = "—"
        cells = [
            revision_display,
            display_time(grouped_run["started_at"]),
            table_metric(dp_prefill),
            table_metric(pcp_prefill),
            table_metric(dp_decode),
            table_metric(dp_tpot_p50),
            decode_protocol,
        ]
        for input_length in ttft_lengths:
            cells.extend(
                (
                    ttft_cell(dp_run, input_length),
                    ttft_cell(pcp_run, input_length),
                )
            )
        rows.append(f"| {' | '.join(cells)} |")

    history_headers = [
        "vllm-torchtpu commit",
        "Test time (UTC)",
        "DP peak prefill tok/s",
        "PCP peak prefill tok/s",
        "DP decode tok/s",
        "DP decode TPOT (ms)",
        "Decode protocol",
        *ttft_headers,
    ]
    history_alignment = [
        "---",
        "---",
        "---:",
        "---:",
        "---:",
        "---:",
        "---",
        *(["---:"] * len(ttft_headers)),
    ]

    return "\n".join(
        [
            README_START,
            "Latest DP8 vs PCP8 throughput by concurrency:",
            "",
            "![Latest DP8 vs PCP8 throughput by concurrency]"
            "(reports/throughput.svg)",
            "",
            "Latest DP8 vs PCP8 single-request prefill TTFT by input length:",
            "",
            "![Latest DP8 vs PCP8 single-request prefill TTFT]"
            "(reports/prefill_ttft.svg)",
            "",
            "Recent DP8 vs PCP8 peak throughput over time:",
            "",
            "![Recent DP8 vs PCP8 peak throughput over time]"
            "(reports/throughput_history.svg)",
            "",
            "Recent DP8 decode throughput over time:",
            "",
            "![Recent DP8 decode throughput over time]"
            "(reports/decode_throughput_history.svg)",
            "",
            *latest_lines,
            "",
            *ttft_status_lines,
            "",
            f"| {' | '.join(history_headers)} |",
            f"| {' | '.join(history_alignment)} |",
            *rows,
            "",
            "Failed benchmark groups are recorded as -1 tok/s in the table and "
            "JSON/CSV reports, while charts plot successful measurements only. "
            "The prefill charts compare DP8 and PCP8 throughput and track their "
            "recent peaks. The combined history table records each run's "
            "throughput and per-length median TTFT; missing measurements are "
            "shown as — and failed lengths as failed. The single-request TTFT "
            "chart uses concurrency 1, runs requests serially, and plots median "
            "latency to the first generated token across the completed samples. "
            "The decode chart keeps "
            "legacy peak-output and current peak-active P50 statistics in "
            "separate series; see "
            "[`reports/latest.json`](reports/latest.json) for the newest peaks and "
            "[`reports/throughput_history.json`](reports/throughput_history.json) "
            "for the full history.",
            README_END,
        ]
    )


def update_readme(path: Path, block: str) -> None:
    content = path.read_text(encoding="utf-8")
    if content.count(README_START) != 1 or content.count(README_END) != 1:
        raise ValueError(f"README report markers are missing or duplicated in {path}")
    before, remainder = content.split(README_START, 1)
    _, after = remainder.split(README_END, 1)
    atomic_write(path, before + block + after)


def main() -> None:
    args = parse_args()
    if args.display_limit <= 0 or args.table_limit <= 0:
        raise SystemExit("display limit and table limit must be positive")

    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    summary_path = args.summary.resolve() if args.summary is not None else None
    decode_status = args.decode_status
    if decode_status is None:
        decode_status = "success" if args.decode_summary is not None else "not-run"
    ttft_status = args.ttft_status
    if ttft_status is None:
        ttft_status = "success" if args.ttft_summary is not None else "not-run"
    reports_dir = project_root / "reports"
    history_path = reports_dir / "throughput_history.json"
    latest_path = reports_dir / "latest.json"
    csv_path = reports_dir / "throughput_history.csv"
    svg_path = reports_dir / "throughput.svg"
    history_svg_path = reports_dir / "throughput_history.svg"
    decode_history_svg_path = reports_dir / "decode_throughput_history.svg"
    prefill_ttft_svg_path = reports_dir / "prefill_ttft.svg"
    prefill_ttft_csv_path = reports_dir / "prefill_ttft_history.csv"
    readme_path = project_root / "README.md"
    lock_path = project_root / ".state" / "benchmark_report.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        record = build_record(
            project_root=project_root,
            run_dir=run_dir,
            summary_path=summary_path,
            input_length=args.input_length,
            output_length=args.output_length,
            model=args.model,
            benchmark_config=args.benchmark_config,
            decode_summary_path=(
                args.decode_summary.resolve()
                if args.decode_summary is not None
                else None
            ),
            ttft_summary_path=(
                args.ttft_summary.resolve()
                if args.ttft_summary is not None
                else None
            ),
            status=args.status,
            decode_status=decode_status,
            ttft_status=ttft_status,
        )
        runs = update_history(load_history(history_path), record)
        latest = latest_runs_by_config(runs)
        homepage_series, homepage_labels = concurrency_chart_data(latest)
        visible_runs = visible_history_runs(runs, args.display_limit)
        history_series, history_labels = history_chart_data(visible_runs)
        visible_run_count = len({run["run_id"] for run in visible_runs})
        decode_history_series, decode_history_labels = decode_history_chart_data(
            visible_runs
        )
        decode_run_count = len(decode_history_labels)
        latest_ttft = latest_ttft_runs_by_config(runs)
        prefill_ttft_series, prefill_ttft_labels = prefill_ttft_chart_data(
            latest_ttft
        )

        atomic_write(history_path, render_history_json(runs))
        atomic_write(latest_path, render_latest_json(runs))
        atomic_write(csv_path, render_csv(runs))
        atomic_write(prefill_ttft_csv_path, render_prefill_ttft_csv(runs))
        if homepage_series and homepage_labels:
            homepage_svg = chart_svg(
                homepage_series,
                homepage_labels,
                title="Latest DP8 vs PCP8 throughput by concurrency",
                description=(
                    "Total token throughput at each tested concurrency for the "
                    "latest DP8 and PCP8 benchmark attempts with successful data."
                ),
            )
        else:
            homepage_svg = empty_chart_svg(
                title="Latest DP8 vs PCP8 throughput by concurrency",
                description=(
                    "No successful DP8 or PCP8 concurrency measurements are "
                    "available for the latest benchmark attempts."
                ),
                id_prefix="throughput",
            )
        atomic_write(svg_path, homepage_svg)
        if history_series and history_labels:
            history_svg = chart_svg(
                history_series,
                history_labels,
                title=(
                    "DP8 vs PCP8 peak throughput over time — "
                    f"last {visible_run_count} runs"
                ),
                description=(
                    "Peak total token throughput for recent DP8 and PCP8 "
                    "benchmark runs, ordered by completion time."
                ),
                id_prefix="history",
            )
        else:
            history_svg = empty_chart_svg(
                title=(
                    "DP8 vs PCP8 peak throughput over time — "
                    f"last {visible_run_count} runs"
                ),
                description="No successful prefill measurements are available.",
                id_prefix="history",
            )
        atomic_write(history_svg_path, history_svg)
        decode_history_title = (
            f"DP8 decode throughput over time — last {decode_run_count} runs"
        )
        decode_history_description = (
            "Legacy peak-output throughput and current C256 peak-active "
            "window P50 throughput are separate series because their "
            "statistics are not directly comparable."
        )
        if decode_history_series:
            decode_history_svg = chart_svg(
                decode_history_series,
                decode_history_labels,
                title=decode_history_title,
                description=decode_history_description,
                id_prefix="decode-history",
            )
        else:
            decode_history_svg = empty_chart_svg(
                title=decode_history_title,
                description=decode_history_description,
                id_prefix="decode-history",
            )
        atomic_write(decode_history_svg_path, decode_history_svg)
        if prefill_ttft_series and prefill_ttft_labels:
            prefill_ttft_svg = chart_svg(
                prefill_ttft_series,
                prefill_ttft_labels,
                title="DP8 vs PCP8 single-request prefill TTFT",
                description=(
                    "Latency to the first generated token at concurrency one "
                    "for the latest DP8 and PCP8 TTFT benchmark attempts."
                ),
                id_prefix="prefill-ttft",
                value_suffix=" ms",
            )
        else:
            prefill_ttft_svg = empty_chart_svg(
                title="DP8 vs PCP8 single-request prefill TTFT",
                description="No successful single-request TTFT data is available.",
                id_prefix="prefill-ttft",
            )
        atomic_write(prefill_ttft_svg_path, prefill_ttft_svg)
        update_readme(
            readme_path,
            render_readme_block(runs, args.table_limit),
        )

    throughput = record["best_total_token_throughput"]
    throughput_display = (
        "not run" if throughput is None else f"{throughput:,.2f} tok/s"
    )
    print(
        f"Recorded {config_label(record['benchmark_config'])} "
        f"{record['status']} result: {throughput_display} "
        f"(run={record['run_id']})"
    )
    print(f"Latest throughput chart: {svg_path}")
    print(f"Throughput history chart: {history_svg_path}")
    print(f"Decode throughput history chart: {decode_history_svg_path}")
    print(f"Single-request prefill TTFT chart: {prefill_ttft_svg_path}")


if __name__ == "__main__":
    main()
