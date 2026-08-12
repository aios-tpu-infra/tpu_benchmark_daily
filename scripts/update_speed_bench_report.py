#!/usr/bin/env python3

"""Record semantic mixed-length DP8 and PCP8 prefill benchmarks."""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
README_START = "<!-- SPEED_BENCH_REPORT_START -->"
README_END = "<!-- SPEED_BENCH_REPORT_END -->"
BENCHMARK_CONFIGS = {
    "dp8": "DP8",
    "pcp8": "PCP8",
}
README_CONCURRENCIES = (8, 64)
CSV_FIELDS = (
    "run_id",
    "benchmark_config",
    "status",
    "mode",
    "started_at",
    "completed_at",
    "machine_ip",
    "model",
    "num_prompts",
    "min_input_tokens",
    "max_input_tokens",
    "total_input_tokens",
    "dataset_sha256",
    "source_revision",
    "throughput_status",
    "throughput_concurrency",
    "input_token_throughput",
    "total_token_throughput",
    "request_throughput",
    "throughput_median_ttft_ms",
    "throughput_p90_ttft_ms",
    "throughput_p99_ttft_ms",
    "serial_ttft_status",
    "serial_median_ttft_ms",
    "serial_p90_ttft_ms",
    "serial_p99_ttft_ms",
    "torchtpu_vllm_revision",
    "torch_tpu_version",
    "summary_path",
)


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


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def component_metric(component: dict[str, Any], field: str) -> Any:
    return component.get(field) if component.get("status") == "success" else None


def summary_concurrency_results(
    throughput: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_results = throughput.get("results")
    if raw_results is None:
        raw_results = [throughput]
    if not isinstance(raw_results, list) or not all(
        isinstance(item, dict) for item in raw_results
    ):
        raise ValueError("invalid SPEED-Bench concurrency results")

    results: list[dict[str, Any]] = []
    seen: set[int] = set()
    for component in raw_results:
        assert isinstance(component, dict)
        concurrency_value = component.get("configured_max_concurrency")
        if concurrency_value is None:
            continue
        concurrency = int(concurrency_value)
        if concurrency <= 0 or concurrency in seen:
            raise ValueError(
                f"invalid or duplicate SPEED-Bench concurrency: {concurrency}"
            )
        seen.add(concurrency)
        status = str(component.get("status", "failed"))
        if status not in {"success", "failed"}:
            raise ValueError(f"invalid concurrency result status: {status!r}")
        results.append(
            {
                "concurrency": concurrency,
                "status": status,
                "input_token_throughput": component_metric(
                    component, "input_token_throughput"
                ),
                "total_token_throughput": component_metric(
                    component, "total_token_throughput"
                ),
                "request_throughput": component_metric(
                    component, "request_throughput"
                ),
                "ttft_p50_ms": component_metric(component, "median_ttft_ms"),
                "ttft_p90_ms": component_metric(component, "p90_ttft_ms"),
                "ttft_p99_ms": component_metric(component, "p99_ttft_ms"),
            }
        )
    return sorted(results, key=lambda item: item["concurrency"])


def record_concurrency_results(run: dict[str, Any]) -> list[dict[str, Any]]:
    raw_results = run.get("concurrency_results")
    if isinstance(raw_results, list) and all(
        isinstance(item, dict) for item in raw_results
    ):
        return sorted(raw_results, key=lambda item: int(item["concurrency"]))
    concurrency = run.get("throughput_concurrency")
    if concurrency is None:
        return []
    return [
        {
            "concurrency": int(concurrency),
            "status": str(run.get("throughput_status", "failed")),
            "input_token_throughput": run.get("input_token_throughput"),
            "total_token_throughput": run.get("total_token_throughput"),
            "request_throughput": run.get("request_throughput"),
            "ttft_p50_ms": run.get("throughput_median_ttft_ms"),
            "ttft_p90_ms": run.get("throughput_p90_ttft_ms"),
            "ttft_p99_ms": run.get("throughput_p99_ttft_ms"),
        }
    ]


def normalize_benchmark_config(
    value: Any, *, legacy_default: bool = False
) -> str:
    config = str(value or "").strip().lower()
    if not config and legacy_default:
        return "dp8"
    if config not in BENCHMARK_CONFIGS:
        supported = ", ".join(BENCHMARK_CONFIGS)
        raise ValueError(
            f"benchmark_config must be one of {supported}, got {value!r}"
        )
    return config


def config_label(config: str) -> str:
    return BENCHMARK_CONFIGS[config]


def build_record(
    project_root: Path, run_dir: Path, summary_path: Path, model: str
) -> dict[str, Any]:
    summary = load_json(summary_path)
    if int(summary.get("schema_version", 0)) != 1:
        raise ValueError("unsupported SPEED-Bench summary schema")
    status = str(summary.get("status", ""))
    if status not in {"success", "partial", "failed"}:
        raise ValueError(f"invalid SPEED-Bench summary status: {status!r}")
    benchmark = summary.get("benchmark")
    throughput = summary.get("throughput")
    serial_ttft = summary.get("serial_ttft", {"status": "not-run"})
    if not all(isinstance(item, dict) for item in (benchmark, throughput, serial_ttft)):
        raise ValueError("invalid SPEED-Bench summary structure")
    assert isinstance(benchmark, dict)
    assert isinstance(throughput, dict)
    assert isinstance(serial_ttft, dict)
    benchmark_config = normalize_benchmark_config(
        benchmark.get("benchmark_config")
    )
    dataset_sha256 = str(benchmark.get("dataset_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", dataset_sha256) is None:
        raise ValueError("dataset_sha256 must be a lowercase SHA-256 digest")
    metadata_path = run_dir / "run_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.is_file() else {}
    completed_at = str(
        metadata.get(
            "completed_at", datetime.now(UTC).isoformat(timespec="seconds")
        )
    )
    concurrency_results = summary_concurrency_results(throughput)
    primary_result = max(
        concurrency_results,
        key=lambda item: item["concurrency"],
        default={},
    )
    return {
        "run_id": str(metadata.get("run_id", run_dir.name)),
        "status": status,
        "mode": str(summary.get("mode")),
        "started_at": str(metadata.get("started_at", completed_at)),
        "completed_at": completed_at,
        "machine_ip": str(metadata.get("machine_ip", "")),
        "model": model,
        "benchmark_config": benchmark_config,
        "source": str(benchmark.get("source", "")),
        "source_revision": str(benchmark.get("source_revision", "")),
        "dataset_sha256": dataset_sha256,
        "num_prompts": int(benchmark.get("num_prompts", 0)),
        "output_length": int(benchmark.get("output_length", 0)),
        "min_input_tokens": int(benchmark.get("min_input_tokens", 0)),
        "max_input_tokens": int(benchmark.get("max_input_tokens", 0)),
        "mean_input_tokens": float(benchmark.get("mean_input_tokens", 0)),
        "total_input_tokens": int(benchmark.get("total_input_tokens", 0)),
        "throughput_status": str(throughput.get("status", "failed")),
        "concurrency_results": concurrency_results,
        # Keep the highest-concurrency result in the legacy flat fields so
        # existing report readers continue to receive one representative row.
        "throughput_concurrency": primary_result.get("concurrency"),
        "input_token_throughput": primary_result.get("input_token_throughput"),
        "total_token_throughput": primary_result.get("total_token_throughput"),
        "request_throughput": primary_result.get("request_throughput"),
        "throughput_median_ttft_ms": primary_result.get("ttft_p50_ms"),
        "throughput_p90_ttft_ms": primary_result.get("ttft_p90_ms"),
        "throughput_p99_ttft_ms": primary_result.get("ttft_p99_ms"),
        "serial_ttft_status": str(serial_ttft.get("status", "failed")),
        "serial_mean_ttft_ms": component_metric(serial_ttft, "mean_ttft_ms"),
        "serial_median_ttft_ms": component_metric(serial_ttft, "median_ttft_ms"),
        "serial_p90_ttft_ms": component_metric(serial_ttft, "p90_ttft_ms"),
        "serial_p99_ttft_ms": component_metric(serial_ttft, "p99_ttft_ms"),
        "serial_ttft_observations": (
            serial_ttft.get("observations", [])
            if serial_ttft.get("status") == "success"
            else []
        ),
        "torchtpu_vllm_revision": str(
            metadata.get("torchtpu_vllm_revision", "unknown")
        ),
        "torch_tpu_version": str(metadata.get("torch_tpu_version", "unknown")),
        "summary_path": relative_path(summary_path, project_root),
    }


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    history = load_json(path)
    if int(history.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported SPEED-Bench history schema in {path}")
    runs = history.get("runs")
    if not isinstance(runs, list) or not all(isinstance(item, dict) for item in runs):
        raise ValueError(f"invalid SPEED-Bench history in {path}")
    return runs


def update_history(
    runs: list[dict[str, Any]], record: dict[str, Any]
) -> list[dict[str, Any]]:
    record_key = (
        str(record["run_id"]),
        normalize_benchmark_config(record.get("benchmark_config")),
    )
    by_run_and_config: dict[tuple[str, str], dict[str, Any]] = {}
    for item in runs:
        item_config = normalize_benchmark_config(
            item.get("benchmark_config"), legacy_default=True
        )
        item_key = (
            str(item["run_id"]),
            item_config,
        )
        if item_key == record_key or item.get("summary_path") == record.get(
            "summary_path"
        ):
            continue
        by_run_and_config[item_key] = {
            **item,
            "benchmark_config": item_config,
        }
    by_run_and_config[record_key] = record
    return sorted(
        by_run_and_config.values(),
        key=lambda item: (
            item["completed_at"],
            item["run_id"],
            normalize_benchmark_config(
                item.get("benchmark_config"), legacy_default=True
            ),
        ),
    )


def render_history(runs: list[dict[str, Any]]) -> str:
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, "runs": runs},
        indent=2,
        sort_keys=True,
    ) + "\n"


def render_latest(runs: list[dict[str, Any]]) -> str:
    latest = runs[-1] if runs else None
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, "benchmark": latest},
        indent=2,
        sort_keys=True,
    ) + "\n"


def render_csv(runs: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=CSV_FIELDS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for run in runs:
        concurrency_results = record_concurrency_results(run)
        if not concurrency_results:
            writer.writerow(run)
            continue
        for result in concurrency_results:
            writer.writerow(
                {
                    **run,
                    "throughput_status": result.get("status"),
                    "throughput_concurrency": result.get("concurrency"),
                    "input_token_throughput": result.get(
                        "input_token_throughput"
                    ),
                    "total_token_throughput": result.get(
                        "total_token_throughput"
                    ),
                    "request_throughput": result.get("request_throughput"),
                    "throughput_median_ttft_ms": result.get("ttft_p50_ms"),
                    "throughput_p90_ttft_ms": result.get("ttft_p90_ms"),
                    "throughput_p99_ttft_ms": result.get("ttft_p99_ms"),
                }
            )
    return output.getvalue()


def display_time(value: str) -> str:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            UTC
        ).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def display_metric(value: Any, *, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.{digits}f}{suffix}"


def readme_commit_rows(
    runs: list[dict[str, Any]], table_limit: int
) -> list[dict[str, Any]]:
    """Group the latest result for each config under one commit row."""
    grouped: dict[str, dict[str, Any]] = {}
    for run in runs:
        revision = str(run.get("torchtpu_vllm_revision") or "unknown")
        # Unknown legacy revisions must not all collapse into one row.
        group_key = revision if revision != "unknown" else f"run:{run['run_id']}"
        row = grouped.setdefault(
            group_key,
            {
                "revision": revision,
                "completed_at": str(run["completed_at"]),
                "configs": {},
            },
        )
        completed_at = str(run["completed_at"])
        if completed_at > row["completed_at"]:
            row["completed_at"] = completed_at
        config = normalize_benchmark_config(
            run.get("benchmark_config"), legacy_default=True
        )
        previous = row["configs"].get(config)
        if previous is None or completed_at >= str(previous["completed_at"]):
            row["configs"][config] = run
    return sorted(
        grouped.values(),
        key=lambda row: (row["completed_at"], row["revision"]),
        reverse=True,
    )[:table_limit]


def readme_result_cell(run: dict[str, Any] | None, concurrency: int) -> str:
    if run is None:
        return "—"
    result = next(
        (
            item
            for item in record_concurrency_results(run)
            if int(item["concurrency"]) == concurrency
        ),
        None,
    )
    if result is None:
        return "—"
    if result.get("status") != "success":
        return "**failed**"
    throughput = display_metric(result.get("input_token_throughput"))
    percentiles = "/".join(
        display_metric(result.get(field))
        for field in ("ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms")
    )
    return (
        f"**{throughput} tok/s**<br>"
        f"P50/P90/P99: {percentiles} ms"
    )


def render_readme_block(runs: list[dict[str, Any]], table_limit: int) -> str:
    if not runs:
        return "No semantic mixed-length benchmark runs have been recorded."
    latest = runs[-1]
    latest_config = normalize_benchmark_config(
        latest.get("benchmark_config"), legacy_default=True
    )
    latest_results = record_concurrency_results(latest)
    latest_metrics = []
    for result in latest_results:
        concurrency = int(result["concurrency"])
        if result.get("status") != "success":
            latest_metrics.append(f"C{concurrency} **failed**")
            continue
        input_throughput = display_metric(
            result.get("input_token_throughput"), suffix=" input tok/s"
        )
        latest_metrics.append(
            f"C{concurrency} **{input_throughput}**, "
            "TTFT P50/P90/P99 "
            f"**{display_metric(result.get('ttft_p50_ms'))}/"
            f"{display_metric(result.get('ttft_p90_ms'))}/"
            f"{display_metric(result.get('ttft_p99_ms'))} ms**"
        )
    latest_summary = "; ".join(latest_metrics) or "no concurrency results"
    concurrency_labels = ", ".join(
        f"C{int(result['concurrency'])}" for result in latest_results
    )
    lines = [
        (
            f"Latest {config_label(latest_config)} semantic mixed-length result: "
            f"{latest_summary} (`{latest['run_id']}`)."
        ),
        "",
        (
            "The latest recorded dataset contains "
            f"**{latest['num_prompts']}** requests from "
            f"NVIDIA SPEED-Bench, ranging from **{latest['min_input_tokens']:,}** "
            f"to **{latest['max_input_tokens']:,}** input tokens "
            f"(SHA-256 `{latest['dataset_sha256'][:12]}…`). Each {concurrency_labels} "
            "serving run reports both throughput and load TTFT."
        ),
        "",
        (
            "Each result cell shows **input tok/s** followed by "
            "TTFT **P50/P90/P99** in milliseconds."
        ),
        "",
        (
            "| vllm-torchtpu commit | Test time (UTC) | "
            "DP C8 | DP C64 | PCP C8 | PCP C64 |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in readme_commit_rows(runs, table_limit):
        revision = str(row["revision"])
        revision_label = f"`{revision[:12]}`" if revision != "unknown" else "—"
        cells = [revision_label, display_time(str(row["completed_at"]))]
        for config in ("dp8", "pcp8"):
            for concurrency in README_CONCURRENCIES:
                cells.append(
                    readme_result_cell(row["configs"].get(config), concurrency)
                )
        lines.append(f"| {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            (
                "Full machine-readable history is stored in "
                "[`reports/speed_bench_history.json`](reports/speed_bench_history.json) "
                "and [`reports/speed_bench_history.csv`](reports/speed_bench_history.csv)."
            ),
        ]
    )
    return "\n".join(lines)


def update_readme(path: Path, block: str) -> None:
    content = path.read_text(encoding="utf-8")
    replacement = f"{README_START}\n{block}\n{README_END}"
    if README_START in content and README_END in content:
        before, remainder = content.split(README_START, 1)
        _, after = remainder.split(README_END, 1)
        updated = before + replacement + after
    else:
        section = f"## Real variable-length prefill benchmark\n\n{replacement}\n\n"
        anchor = "## Layout\n"
        if anchor in content:
            updated = content.replace(anchor, section + anchor, 1)
        else:
            updated = content.rstrip() + "\n\n" + section
    atomic_write(path, updated)


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=script_root)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--table-limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    summary_path = args.summary.resolve()
    reports_dir = project_root / "reports"
    history_path = reports_dir / "speed_bench_history.json"
    latest_path = reports_dir / "speed_bench_latest.json"
    csv_path = reports_dir / "speed_bench_history.csv"
    readme_path = project_root / "README.md"
    lock_path = project_root / ".state" / "report.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    record = build_record(project_root, run_dir, summary_path, args.model)

    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runs = update_history(load_history(history_path), record)
        atomic_write(history_path, render_history(runs))
        atomic_write(latest_path, render_latest(runs))
        atomic_write(csv_path, render_csv(runs))
        update_readme(readme_path, render_readme_block(runs, args.table_limit))

    print(
        f"Recorded {config_label(record['benchmark_config'])} "
        "SPEED-Bench mixed workload: "
        f"{record['status']} (run={record['run_id']})"
    )
    print(f"SPEED-Bench history: {history_path}")


if __name__ == "__main__":
    main()
