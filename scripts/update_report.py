#!/usr/bin/env python3

"""Record a successful benchmark and regenerate the local throughput report."""

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


SCHEMA_VERSION = 2
README_START = "<!-- BENCHMARK_REPORT_START -->"
README_END = "<!-- BENCHMARK_REPORT_END -->"
BENCHMARK_CONFIGS = {
    "dp8": {"label": "DP8", "color": "#1570ef"},
    "pcp8": {"label": "PCP8", "color": "#7a5af8"},
}
CSV_FIELDS = (
    "run_id",
    "benchmark_config",
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
    "torchtpu_vllm_revision",
    "torch_tpu_revision",
    "torch_tpu_version",
    "summary_path",
)


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Append one successful benchmark and regenerate reports."
    )
    parser.add_argument("--project-root", type=Path, default=script_root)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
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


def build_record(
    *,
    project_root: Path,
    run_dir: Path,
    summary_path: Path,
    input_length: int | None,
    output_length: int | None,
    model: str | None,
    benchmark_config: str | None,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    best = summary.get("best")
    results = summary.get("results")
    if not isinstance(best, dict) or not isinstance(results, list) or not results:
        raise ValueError(f"invalid benchmark summary: {summary_path}")

    failed = sum(int(item.get("failed", 0)) for item in results)
    if failed:
        raise ValueError(f"refusing to record a benchmark with {failed} failed requests")

    best_filename = best.get("file")
    detail: dict[str, Any] = {}
    if isinstance(best_filename, str):
        detail_path = summary_path.parent / best_filename
        if detail_path.is_file():
            detail = load_json(detail_path)

    benchmark = summary.get("benchmark")
    if not isinstance(benchmark, dict):
        benchmark = {}
    benchmark_config = normalize_benchmark_config(
        benchmark_config
        or benchmark.get("benchmark_config")
        or infer_legacy_benchmark_config(
            {"run_id": run_dir.name, "summary_path": str(summary_path)}
        )
    )

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
        started_at = iso_utc(
            datetime.fromtimestamp(summary_path.stat().st_mtime, tz=UTC)
        )
    completed_at = benchmark_timestamp(detail.get("date")) or iso_utc(
        datetime.fromtimestamp(summary_path.stat().st_mtime, tz=UTC)
    )

    concurrency_results = []
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
                "mean_ttft_ms": finite_float(item["mean_ttft_ms"], "mean_ttft_ms"),
                "p99_ttft_ms": finite_float(item["p99_ttft_ms"], "p99_ttft_ms"),
            }
        )

    return {
        "run_id": run_id,
        "benchmark_config": benchmark_config,
        "started_at": started_at,
        "completed_at": completed_at,
        "machine_ip": machine_ip,
        "model": str(model),
        "input_length": input_length,
        "output_length": output_length,
        "best_total_token_throughput": finite_float(
            best["total_token_throughput"], "best_total_token_throughput"
        ),
        "best_request_throughput": finite_float(
            best["request_throughput"], "best_request_throughput"
        ),
        "best_concurrency": positive_int(best["concurrency"], "best_concurrency"),
        "mean_ttft_ms": finite_float(best["mean_ttft_ms"], "mean_ttft_ms"),
        "p99_ttft_ms": finite_float(best["p99_ttft_ms"], "p99_ttft_ms"),
        "torchtpu_vllm_revision": str(
            metadata.get("torchtpu_vllm_revision", "unknown")
        ),
        # Source-backed runs recorded a Git revision. Pip-backed runs have no
        # source checkout, so retain this legacy field as an empty value for
        # history/CSV compatibility and use the package version for display.
        "torch_tpu_revision": str(metadata.get("torch_tpu_revision", "")),
        "torch_tpu_version": str(metadata.get("torch_tpu_version", "unknown")),
        "summary_path": relative_path(summary_path, project_root),
        "concurrency_results": concurrency_results,
    }


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    history = load_json(path)
    schema_version = history.get("schema_version")
    if schema_version not in (1, SCHEMA_VERSION):
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
        parts.append(
            f'<text class="axis" text-anchor="middle" x="{x_values[index]:.2f}" '
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
                f"{latest_value:,.0f} tok/s</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


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
        config_runs = [run for run in runs if run["benchmark_config"] == config]
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
        if run is None:
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
        writer.writerow({field: run[field] for field in CSV_FIELDS})
    return output.getvalue()


def render_html(runs: list[dict[str, Any]], display_limit: int) -> str:
    visible_runs = visible_history_runs(runs, display_limit)
    latest = latest_runs_by_config(runs)
    most_recent = runs[-1]
    trend_series, trend_labels = history_chart_data(visible_runs)
    concurrency_series, concurrency_labels = concurrency_chart_data(latest)
    visible_run_count = len({run["run_id"] for run in visible_runs})

    trend_svg = chart_svg(
        trend_series,
        trend_labels,
        title=f"Peak total token throughput — last {visible_run_count} runs",
        id_prefix="trend",
        standalone=False,
    )
    concurrency_svg = chart_svg(
        concurrency_series,
        concurrency_labels,
        title="Latest DP8 vs PCP8 throughput by concurrency",
        description=(
            "Total token throughput at each tested concurrency for the latest "
            "successful DP8 and PCP8 benchmarks."
        ),
        id_prefix="concurrency",
        standalone=False,
    )

    cards = []
    for config, style in BENCHMARK_CONFIGS.items():
        run = latest.get(config)
        if run is None:
            continue
        config_runs = [item for item in runs if item["benchmark_config"] == config]
        previous = config_runs[-2] if len(config_runs) > 1 else None
        delta = None
        if previous and previous["best_total_token_throughput"]:
            delta = 100 * (
                run["best_total_token_throughput"]
                / previous["best_total_token_throughput"]
                - 1
            )
        delta_text = (
            "first recorded run" if delta is None else f"{delta:+.2f}% vs previous"
        )
        delta_class = (
            "neutral" if delta is None else ("positive" if delta >= 0 else "negative")
        )
        cards.append(
            '<div class="card">'
            f'<div class="label">Latest {html.escape(str(style["label"]))} peak</div>'
            f'<div class="metric">{run["best_total_token_throughput"]:,.2f}</div>'
            f'<div class="note">c{run["best_concurrency"]} · '
            f'{run["best_request_throughput"]:,.3f} req/s · '
            f'<span class="{delta_class}">{html.escape(delta_text)}</span></div>'
            "</div>"
        )
    cards.append(
        '<div class="card"><div class="label">Recorded measurements</div>'
        f'<div class="metric">{len(runs)}</div><div class="note">'
        f'{len({run["run_id"] for run in runs})} benchmark runs</div></div>'
    )

    rows = []
    for run in reversed(visible_runs):
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(run['run_id'])}</code></td>"
            f"<td><strong>{html.escape(config_label(run['benchmark_config']))}</strong></td>"
            f"<td>{html.escape(display_time(run['completed_at']))} UTC</td>"
            f"<td class=number>{run['best_total_token_throughput']:,.2f}</td>"
            f"<td class=number>{run['best_concurrency']}</td>"
            f"<td class=number>{run['best_request_throughput']:,.3f}</td>"
            f"<td class=number>{run['p99_ttft_ms']:,.1f}</td>"
            f"<td><code>{html.escape(run['torch_tpu_version'])}</code></td>"
            "</tr>"
        )

    generated_at = html.escape(str(max(run["completed_at"] for run in runs)))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TPU benchmark throughput</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; background: #f5f7fa; color: #102a43; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 36px 22px 64px; }}
    h1 {{ margin: 0 0 6px; font-size: clamp(28px, 4vw, 44px); }}
    .subtitle {{ color: #627d98; margin: 0 0 28px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}
    .card, .panel {{ background: white; border: 1px solid #d9e2ec; border-radius: 14px; box-shadow: 0 4px 18px #102a430d; }}
    .card {{ padding: 20px; }}
    .label {{ color: #627d98; font-size: 13px; text-transform: uppercase; letter-spacing: .05em; }}
    .metric {{ margin-top: 8px; font-size: 29px; font-weight: 750; }}
    .note {{ margin-top: 7px; color: #486581; font-size: 14px; }}
    .positive {{ color: #087443; }} .negative {{ color: #b42318; }} .neutral {{ color: #486581; }}
    .panel {{ margin-top: 18px; padding: 16px; overflow-x: auto; }}
    .panel svg {{ display: block; width: 100%; min-width: 680px; height: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 900px; font-size: 14px; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid #e6edf3; text-align: left; }}
    th {{ color: #486581; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .number {{ text-align: right; font-variant-numeric: tabular-nums; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
    footer {{ margin-top: 24px; color: #829ab1; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <h1>TPU benchmark throughput</h1>
  <p class="subtitle">Qwen3.5-397B-A17B-FP8 · dummy weights · input {most_recent['input_length']} / output {most_recent['output_length']} · DP8 vs PCP8</p>
  <section class="cards">{''.join(cards)}</section>
  <section class="panel">{trend_svg}</section>
  <section class="panel">{concurrency_svg}</section>
  <section class="panel">
    <table>
      <thead><tr><th>Run</th><th>Config</th><th>Completed</th><th class=number>Peak tok/s</th><th class=number>Concurrency</th><th class=number>Req/s</th><th class=number>p99 TTFT ms</th><th>torch_tpu</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </section>
  <footer>Generated from <code>throughput_history.json</code>. Latest data timestamp: {generated_at}.</footer>
</main>
</body>
</html>
"""


def render_readme_block(runs: list[dict[str, Any]], table_limit: int) -> str:
    latest = latest_runs_by_config(runs)
    latest_lines = []
    for config, style in BENCHMARK_CONFIGS.items():
        run = latest.get(config)
        if run is None:
            continue
        latest_lines.append(
            f"Latest successful {style['label']}: "
            f"**{run['best_total_token_throughput']:,.2f} total tok/s** "
            f"at concurrency **{run['best_concurrency']}** "
            f"(`{run['run_id']}`)."
        )
    rows = []
    for run in reversed(runs[-table_limit:]):
        rows.append(
            f"| {display_time(run['completed_at'])} | "
            f"{config_label(run['benchmark_config'])} | "
            f"{run['best_total_token_throughput']:,.2f} | "
            f"{run['best_concurrency']} | "
            f"{run['best_request_throughput']:,.3f} | "
            f"{run['p99_ttft_ms']:,.1f} |"
        )

    return "\n".join(
        [
            README_START,
            "[![Recent peak throughput](reports/throughput.svg)](reports/index.html)",
            "",
            *latest_lines,
            "",
            "| Completed (UTC) | Config | Peak total tok/s | Best concurrency | Requests/s | p99 TTFT (ms) |",
            "|---|---|---:|---:|---:|---:|",
            *rows,
            "",
            "The chart compares the latest successful DP8 and PCP8 throughput "
            "across concurrency levels; see "
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
    summary_path = args.summary.resolve()
    reports_dir = project_root / "reports"
    history_path = reports_dir / "throughput_history.json"
    latest_path = reports_dir / "latest.json"
    csv_path = reports_dir / "throughput_history.csv"
    svg_path = reports_dir / "throughput.svg"
    html_path = reports_dir / "index.html"
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
        )
        runs = update_history(load_history(history_path), record)
        latest = latest_runs_by_config(runs)
        homepage_series, homepage_labels = concurrency_chart_data(latest)

        atomic_write(history_path, render_history_json(runs))
        atomic_write(latest_path, render_latest_json(runs))
        atomic_write(csv_path, render_csv(runs))
        atomic_write(
            svg_path,
            chart_svg(
                homepage_series,
                homepage_labels,
                title="Latest DP8 vs PCP8 throughput by concurrency",
                description=(
                    "Total token throughput at each tested concurrency for the "
                    "latest successful DP8 and PCP8 benchmarks."
                ),
            ),
        )
        atomic_write(html_path, render_html(runs, args.display_limit))
        update_readme(
            readme_path,
            render_readme_block(runs, args.table_limit),
        )

    print(
        f"Recorded {config_label(record['benchmark_config'])} peak throughput: "
        f"{record['best_total_token_throughput']:,.2f} tok/s "
        f"(run={record['run_id']}, concurrency={record['best_concurrency']})"
    )
    print(f"Throughput dashboard: {html_path}")


if __name__ == "__main__":
    main()
