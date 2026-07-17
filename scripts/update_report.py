#!/usr/bin/env python3

"""Record a successful benchmark and regenerate the local throughput report."""

from __future__ import annotations

import argparse
import csv
import fcntl
import html
import io
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
README_START = "<!-- BENCHMARK_REPORT_START -->"
README_END = "<!-- BENCHMARK_REPORT_END -->"
CSV_FIELDS = (
    "run_id",
    "started_at",
    "completed_at",
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
    try:
        parsed = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
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


def build_record(
    *,
    project_root: Path,
    run_dir: Path,
    summary_path: Path,
    input_length: int | None,
    output_length: int | None,
    model: str | None,
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
    metadata = load_json(metadata_path) if metadata_path.is_file() else {}
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
        "started_at": started_at,
        "completed_at": completed_at,
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
        "torch_tpu_revision": str(metadata.get("torch_tpu_revision", "unknown")),
        "torch_tpu_version": str(metadata.get("torch_tpu_version", "unknown")),
        "summary_path": relative_path(summary_path, project_root),
        "concurrency_results": concurrency_results,
    }


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    history = load_json(path)
    if history.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported history schema in {path}")
    runs = history.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"history runs must be a list in {path}")
    return runs


def update_history(
    runs: list[dict[str, Any]], record: dict[str, Any]
) -> list[dict[str, Any]]:
    by_run_id = {str(item["run_id"]): item for item in runs}
    by_run_id[str(record["run_id"])] = record
    return sorted(
        by_run_id.values(), key=lambda item: (item["completed_at"], item["run_id"])
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
    points: list[dict[str, Any]],
    *,
    title: str,
    description: str = "Peak total token throughput across recent benchmark runs.",
    id_prefix: str = "throughput",
    width: int = 1000,
    height: int = 390,
    standalone: bool = True,
) -> str:
    if not points:
        raise ValueError("cannot render an empty chart")

    left, right, top, bottom = 82, 28, 56, 66
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [finite_float(point["value"], "chart value") for point in points]
    upper = max(values) * 1.12
    if upper <= 0:
        upper = 1.0

    if len(points) == 1:
        x_values = [left + plot_width / 2]
    else:
        x_values = [
            left + index * plot_width / (len(points) - 1)
            for index in range(len(points))
        ]
    y_values = [top + plot_height * (1 - value / upper) for value in values]

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
        ".line{fill:none;stroke:#1570ef;stroke-width:4;stroke-linejoin:round;",
        "stroke-linecap:round}.dot{fill:#fff;stroke:#1570ef;stroke-width:3}",
        ".latest{fill:#f79009;stroke:#b54708;stroke-width:3}",
        ".value{fill:#102a43;font:600 14px system-ui,sans-serif}",
        "</style>",
        f'<rect class="bg" width="{width}" height="{height}" rx="12"/>',
        f'<text class="heading" x="{left}" y="31">{html.escape(title)}</text>',
    ]

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

    coordinates = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in zip(x_values, y_values, strict=True)
    )
    if len(points) > 1:
        parts.append(f'<polyline class="line" points="{coordinates}"/>')

    tick_count = min(6, len(points))
    if tick_count == 1:
        tick_indices = [0]
    else:
        tick_indices = sorted(
            {
                round(index * (len(points) - 1) / (tick_count - 1))
                for index in range(tick_count)
            }
        )
    for index in tick_indices:
        label = html.escape(str(points[index]["label"]))
        parts.append(
            f'<text class="axis" text-anchor="middle" x="{x_values[index]:.2f}" '
            f'y="{top + plot_height + 28}">{label}</text>'
        )

    for index, (point, x, y) in enumerate(
        zip(points, x_values, y_values, strict=True)
    ):
        css_class = "dot latest" if index == len(points) - 1 else "dot"
        tooltip = html.escape(str(point["tooltip"]))
        parts.append(
            f'<circle class="{css_class}" cx="{x:.2f}" cy="{y:.2f}" r="6">'
            f"<title>{tooltip}</title></circle>"
        )

    latest_value = values[-1]
    latest_y = max(top + 15, y_values[-1] - 14)
    latest_anchor = "end" if x_values[-1] > width - 120 else "middle"
    parts.append(
        f'<text class="value" text-anchor="{latest_anchor}" x="{x_values[-1]:.2f}" '
        f'y="{latest_y:.2f}">{latest_value:,.0f} tok/s</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def history_chart_points(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": display_time(run["completed_at"])[5:],
            "value": run["best_total_token_throughput"],
            "tooltip": (
                f"{run['run_id']}: {run['best_total_token_throughput']:,.2f} tok/s "
                f"at concurrency {run['best_concurrency']}"
            ),
        }
        for run in runs
    ]


def concurrency_chart_points(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "label": f"c{result['concurrency']}",
            "value": result["total_token_throughput"],
            "tooltip": (
                f"concurrency {result['concurrency']}: "
                f"{result['total_token_throughput']:,.2f} tok/s"
            ),
        }
        for result in run["concurrency_results"]
    ]


def render_history_json(runs: list[dict[str, Any]]) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": runs[-1]["completed_at"],
        "runs": runs,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_latest_json(run: dict[str, Any]) -> str:
    payload = {
        "run_id": run["run_id"],
        "completed_at": run["completed_at"],
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
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_csv(runs: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for run in runs:
        writer.writerow({field: run[field] for field in CSV_FIELDS})
    return output.getvalue()


def render_html(runs: list[dict[str, Any]], display_limit: int) -> str:
    visible_runs = runs[-display_limit:]
    latest = visible_runs[-1]
    previous = visible_runs[-2] if len(visible_runs) > 1 else None
    delta = None
    if previous and previous["best_total_token_throughput"]:
        delta = 100 * (
            latest["best_total_token_throughput"]
            / previous["best_total_token_throughput"]
            - 1
        )
    delta_text = "first recorded run" if delta is None else f"{delta:+.2f}% vs previous"
    delta_class = "neutral" if delta is None else ("positive" if delta >= 0 else "negative")

    trend_svg = chart_svg(
        history_chart_points(visible_runs),
        title=f"Peak total token throughput — last {len(visible_runs)} runs",
        id_prefix="trend",
        standalone=False,
    )
    concurrency_svg = chart_svg(
        concurrency_chart_points(latest),
        title=f"Latest run by concurrency — {latest['run_id']}",
        description="Total token throughput at each tested concurrency for the latest run.",
        id_prefix="concurrency",
        standalone=False,
    )

    rows = []
    for run in reversed(visible_runs):
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(run['run_id'])}</code></td>"
            f"<td>{html.escape(display_time(run['completed_at']))} UTC</td>"
            f"<td class=number>{run['best_total_token_throughput']:,.2f}</td>"
            f"<td class=number>{run['best_concurrency']}</td>"
            f"<td class=number>{run['best_request_throughput']:,.3f}</td>"
            f"<td class=number>{run['p99_ttft_ms']:,.1f}</td>"
            f"<td><code>{html.escape(run['torch_tpu_revision'][:12])}</code></td>"
            "</tr>"
        )

    generated_at = html.escape(str(runs[-1]["completed_at"]))
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
  <p class="subtitle">Qwen3.5-397B-A17B-FP8 · dummy weights · input {latest['input_length']} / output {latest['output_length']}</p>
  <section class="cards">
    <div class="card"><div class="label">Latest peak throughput</div><div class="metric">{latest['best_total_token_throughput']:,.2f}</div><div class="note">total tok/s</div></div>
    <div class="card"><div class="label">Best concurrency</div><div class="metric">{latest['best_concurrency']}</div><div class="note">{latest['best_request_throughput']:,.3f} requests/s</div></div>
    <div class="card"><div class="label">Change</div><div class="metric {delta_class}">{html.escape(delta_text)}</div><div class="note">successful runs only</div></div>
    <div class="card"><div class="label">Recorded runs</div><div class="metric">{len(runs)}</div><div class="note">showing latest {len(visible_runs)}</div></div>
  </section>
  <section class="panel">{trend_svg}</section>
  <section class="panel">{concurrency_svg}</section>
  <section class="panel">
    <table>
      <thead><tr><th>Run</th><th>Completed</th><th class=number>Peak tok/s</th><th class=number>Concurrency</th><th class=number>Req/s</th><th class=number>p99 TTFT ms</th><th>torch_tpu</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </section>
  <footer>Generated from <code>throughput_history.json</code>. Latest data timestamp: {generated_at}.</footer>
</main>
</body>
</html>
"""


def render_readme_block(runs: list[dict[str, Any]], table_limit: int) -> str:
    latest = runs[-1]
    rows = []
    for run in reversed(runs[-table_limit:]):
        rows.append(
            f"| {display_time(run['completed_at'])} | "
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
            "Latest successful run: "
            f"**{latest['best_total_token_throughput']:,.2f} total tok/s** "
            f"at concurrency **{latest['best_concurrency']}** "
            f"(`{latest['run_id']}`).",
            "",
            "| Completed (UTC) | Peak total tok/s | Best concurrency | Requests/s | p99 TTFT (ms) |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "The chart shows successful runs only; see "
            "[`reports/latest.json`](reports/latest.json) for the newest peak and "
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
    readme_path = project_root / "README"
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
        )
        runs = update_history(load_history(history_path), record)
        visible_runs = runs[-args.display_limit :]

        atomic_write(history_path, render_history_json(runs))
        atomic_write(latest_path, render_latest_json(runs[-1]))
        atomic_write(csv_path, render_csv(runs))
        atomic_write(
            svg_path,
            chart_svg(
                history_chart_points(visible_runs),
                title=f"Peak total token throughput — last {len(visible_runs)} runs",
            ),
        )
        atomic_write(html_path, render_html(runs, args.display_limit))
        update_readme(
            readme_path,
            render_readme_block(runs, args.table_limit),
        )

    print(
        "Recorded peak throughput: "
        f"{record['best_total_token_throughput']:,.2f} tok/s "
        f"(run={record['run_id']}, concurrency={record['best_concurrency']})"
    )
    print(f"Throughput dashboard: {html_path}")


if __name__ == "__main__":
    main()
