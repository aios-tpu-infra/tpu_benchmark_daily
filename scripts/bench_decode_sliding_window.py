#!/usr/bin/env python3
"""Measure full-overlap decode throughput and TPOT with sliding windows.

Every request in one benchmark uses the same natural-language-derived token
prompt and the same decode length. The workload dimensions are intentionally
limited to prompt length, decode length, and concurrency so future daily cases
can reuse this script without inheriting unrelated benchmark modes.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import csv
import dataclasses
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import statistics
import threading
import time
from typing import Any, Iterator

import requests


PROMPT_TEXT = (
    "The benchmark uses a long, ordinary English passage to exercise prefill "
    "before measuring decode throughput. The passage discusses software "
    "systems, scheduling decisions, distributed execution, and careful "
    "measurement methodology in plain language. Each paragraph is stable and "
    "deterministic so every request receives exactly the same input. The goal "
    "is to keep the prompt natural enough for tokenizer behavior to resemble "
    "real text while preserving a controlled token length. "
)


@dataclasses.dataclass(frozen=True)
class RequestResult:
    request_id: int
    started_s: float
    finished_s: float
    token_times_s: list[float]
    error: str | None


@dataclasses.dataclass(frozen=True)
class RoundAnalysis:
    summary: dict[str, Any]
    windows: list[dict[str, Any]]
    request_tpots: list[dict[str, Any]]


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return value


def nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative number")
    return value


def repeat_and_truncate_token_ids(
    source_token_ids: list[int], target_tokens: int
) -> list[int]:
    if not source_token_ids:
        raise ValueError("fixed prompt text produced no tokens")
    repeats = math.ceil(target_tokens / len(source_token_ids))
    return (source_token_ids * repeats)[:target_tokens]


def build_prompt_token_ids(
    *, tokenizer_dir: Path, target_tokens: int
) -> tuple[list[int], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=False,
        local_files_only=True,
    )
    source_token_ids = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    prompt_token_ids = repeat_and_truncate_token_ids(
        source_token_ids, target_tokens
    )
    return prompt_token_ids, len(source_token_ids)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]


def metric_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "avg": statistics.mean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "stddev": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
    }


def build_request_tpot_rows(
    *,
    round_index: int,
    results: list[RequestResult],
    decode_tokens: int,
    batch_start_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        received_tokens = len(result.token_times_s)
        if (
            result.error is not None
            or received_tokens != decode_tokens
            or received_tokens < 2
        ):
            continue
        first_token_s = result.token_times_s[0]
        last_token_s = result.token_times_s[-1]
        decode_duration_s = last_token_s - first_token_s
        rows.append(
            {
                "round": round_index,
                "request_id": result.request_id,
                "received_tokens": received_tokens,
                "first_token_after_batch_start_s": first_token_s - batch_start_s,
                "last_token_after_batch_start_s": last_token_s - batch_start_s,
                "decode_duration_s": decode_duration_s,
                "tpot_ms": decode_duration_s / (received_tokens - 1) * 1000,
            }
        )
    return rows


def token_timestamps_from_choice(
    choice: dict[str, Any], received_s: float
) -> list[float]:
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list):
        raise ValueError("stream chunk is missing token_ids")
    if not all(isinstance(token_id, int) for token_id in token_ids):
        raise ValueError("stream chunk contains a non-integer token ID")
    return [received_s] * len(token_ids)


def iter_sse_data(response: requests.Response) -> Iterator[str]:
    """Yield SSE data without reading a potentially huge event byte by byte."""
    response.raw.decode_content = True
    while raw_line := response.raw.readline():
        if not raw_line.startswith(b"data: "):
            continue
        yield raw_line[len(b"data: ") :].rstrip(b"\r\n").decode("utf-8")


def streamed_completion(
    *,
    request_id: int,
    endpoint: str,
    request_body: bytes,
    start_barrier: threading.Barrier,
    timeout_s: float,
) -> RequestResult:
    try:
        start_barrier.wait(timeout=60)
    except threading.BrokenBarrierError:
        now = time.perf_counter()
        return RequestResult(request_id, now, now, [], "start barrier failed")

    started_s = time.perf_counter()
    token_times_s: list[float] = []
    error = None
    try:
        with requests.post(
            endpoint,
            data=request_body,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
            stream=True,
            timeout=(10.0, timeout_s),
        ) as response:
            if response.status_code != 200:
                error = f"HTTP {response.status_code}: {response.text[:500]}"
            else:
                for data in iter_sse_data(response):
                    received_s = time.perf_counter()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    token_times_s.extend(
                        token_timestamps_from_choice(choice, received_s)
                    )
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)

    return RequestResult(
        request_id=request_id,
        started_s=started_s,
        finished_s=time.perf_counter(),
        token_times_s=token_times_s,
        error=error,
    )


def run_round(
    *,
    endpoint: str,
    request_body: bytes,
    concurrency: int,
    timeout_s: float,
) -> list[RequestResult]:
    barrier = threading.Barrier(concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                streamed_completion,
                request_id=request_id,
                endpoint=endpoint,
                request_body=request_body,
                start_barrier=barrier,
                timeout_s=timeout_s,
            )
            for request_id in range(concurrency)
        ]
        results = [
            future.result()
            for future in concurrent.futures.as_completed(futures)
        ]
    return sorted(results, key=lambda result: result.request_id)


def analyze_round(
    *,
    round_index: int,
    results: list[RequestResult],
    concurrency: int,
    decode_tokens: int,
    window_s: float,
    step_s: float,
) -> RoundAnalysis:
    failures = [
        {
            "request_id": result.request_id,
            "received_tokens": len(result.token_times_s),
            "error": result.error or (
                f"expected {decode_tokens} tokens, received {len(result.token_times_s)}"
            ),
        }
        for result in results
        if result.error is not None or len(result.token_times_s) != decode_tokens
    ]
    batch_start_s = min((result.started_s for result in results), default=0.0)
    request_tpots = build_request_tpot_rows(
        round_index=round_index,
        results=results,
        decode_tokens=decode_tokens,
        batch_start_s=batch_start_s,
    )
    summary: dict[str, Any] = {
        "round": round_index,
        "concurrency": concurrency,
        "successful_requests": len(results) - len(failures),
        "failed_requests": len(failures),
        "full_overlap_start_after_batch_start_s": None,
        "full_overlap_end_after_batch_start_s": None,
        "full_overlap_duration_s": 0.0,
        "window_count": 0,
        "window_throughput_tok_s": metric_stats([]),
        "request_tpot_ms": metric_stats(
            [float(row["tpot_ms"]) for row in request_tpots]
        ),
        "valid": False,
        "invalid_reason": "",
        "errors": failures,
    }
    if failures or len(results) != concurrency:
        summary["invalid_reason"] = "incomplete_requests"
        return RoundAnalysis(summary, [], request_tpots)

    full_start_s = max(result.token_times_s[1] for result in results)
    full_end_s = min(result.token_times_s[-1] for result in results)
    full_duration_s = max(0.0, full_end_s - full_start_s)
    summary.update(
        {
            "full_overlap_start_after_batch_start_s": full_start_s - batch_start_s,
            "full_overlap_end_after_batch_start_s": full_end_s - batch_start_s,
            "full_overlap_duration_s": full_duration_s,
        }
    )
    if full_duration_s < window_s:
        summary["invalid_reason"] = "full_overlap_shorter_than_window"
        return RoundAnalysis(summary, [], request_tpots)

    all_token_times = sorted(
        timestamp for result in results for timestamp in result.token_times_s
    )
    windows: list[dict[str, Any]] = []
    cursor_s = full_start_s
    while cursor_s + window_s <= full_end_s + 1e-9:
        end_s = cursor_s + window_s
        token_left = bisect.bisect_left(all_token_times, cursor_s)
        token_right = bisect.bisect_left(all_token_times, end_s)
        token_count = token_right - token_left
        windows.append(
            {
                "round": round_index,
                "window": len(windows) + 1,
                "start_after_batch_start_s": cursor_s - batch_start_s,
                "end_after_batch_start_s": end_s - batch_start_s,
                "active_requests": concurrency,
                "token_count": token_count,
                "throughput_tok_s": token_count / window_s,
            }
        )
        cursor_s += step_s

    throughputs = [float(window["throughput_tok_s"]) for window in windows]
    summary.update(
        {
            "window_count": len(windows),
            "window_throughput_tok_s": metric_stats(throughputs),
            "valid": bool(windows),
            "invalid_reason": "" if windows else "no_samples",
        }
    )
    return RoundAnalysis(summary, windows, request_tpots)


def write_raw_requests(
    path: Path,
    *,
    round_index: int,
    results: list[RequestResult],
) -> None:
    batch_start_s = min((result.started_s for result in results), default=0.0)
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            record = {
                "round": round_index,
                "request_id": result.request_id,
                "started_after_batch_start_s": result.started_s - batch_start_s,
                "finished_after_batch_start_s": result.finished_s - batch_start_s,
                "received_tokens": len(result.token_times_s),
                "token_times_after_batch_start_s": [
                    timestamp - batch_start_s for timestamp in result.token_times_s
                ],
                "error": result.error,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_request_tpot_csv(
    path: Path, request_tpots: list[dict[str, Any]]
) -> None:
    fieldnames = [
        "round",
        "request_id",
        "received_tokens",
        "first_token_after_batch_start_s",
        "last_token_after_batch_start_s",
        "decode_duration_s",
        "tpot_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(request_tpots)


def write_rounds_csv(path: Path, rounds: list[dict[str, Any]]) -> None:
    fieldnames = [
        "round",
        "valid",
        "successful_requests",
        "failed_requests",
        "full_overlap_duration_s",
        "window_count",
        "throughput_count",
        "throughput_avg_tok_s",
        "throughput_min_tok_s",
        "throughput_max_tok_s",
        "throughput_stddev_tok_s",
        "throughput_p90_tok_s",
        "throughput_p99_tok_s",
        "request_tpot_count",
        "request_tpot_avg_ms",
        "request_tpot_min_ms",
        "request_tpot_max_ms",
        "request_tpot_stddev_ms",
        "request_tpot_p50_ms",
        "request_tpot_p90_ms",
        "request_tpot_p99_ms",
        "invalid_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rounds:
            throughput = row["window_throughput_tok_s"]
            request_tpot = row["request_tpot_ms"]
            writer.writerow(
                {
                    "round": row["round"],
                    "valid": row["valid"],
                    "successful_requests": row["successful_requests"],
                    "failed_requests": row["failed_requests"],
                    "full_overlap_duration_s": row["full_overlap_duration_s"],
                    "window_count": row["window_count"],
                    "throughput_count": throughput["count"],
                    "throughput_avg_tok_s": throughput["avg"],
                    "throughput_min_tok_s": throughput["min"],
                    "throughput_max_tok_s": throughput["max"],
                    "throughput_stddev_tok_s": throughput["stddev"],
                    "throughput_p90_tok_s": throughput["p90"],
                    "throughput_p99_tok_s": throughput["p99"],
                    "request_tpot_count": request_tpot["count"],
                    "request_tpot_avg_ms": request_tpot["avg"],
                    "request_tpot_min_ms": request_tpot["min"],
                    "request_tpot_max_ms": request_tpot["max"],
                    "request_tpot_stddev_ms": request_tpot["stddev"],
                    "request_tpot_p50_ms": request_tpot["p50"],
                    "request_tpot_p90_ms": request_tpot["p90"],
                    "request_tpot_p99_ms": request_tpot["p99"],
                    "invalid_reason": row["invalid_reason"],
                }
            )


def write_timeline_csv(path: Path, windows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "round",
        "window",
        "start_after_batch_start_s",
        "end_after_batch_start_s",
        "active_requests",
        "token_count",
        "throughput_tok_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(windows)


def format_metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    benchmark = summary["benchmark"]
    aggregate = summary["aggregate"]
    throughput = aggregate["window_throughput_tok_s"]
    request_tpot = aggregate["request_tpot_ms"]
    valid_rounds = summary["valid_rounds"]
    requested_rounds = summary["requested_rounds"]
    lines = [
        "# Decode 滑动窗口 Daily Benchmark",
        "",
        "## TL;DR",
        "",
        (
            f"本次使用 C{benchmark['concurrency']}、{benchmark['prefill_tokens']} token "
            f"prefill 和 {benchmark['decode_tokens']} token decode；"
            f"有效轮次 {valid_rounds}/{requested_rounds}。"
        ),
        "",
        "## 配置",
        "",
        f"- 模型：`{benchmark['model']}`",
        f"- 并发：`{benchmark['concurrency']}`",
        f"- Prefill：`{benchmark['prefill_tokens']}` tokens",
        f"- Decode：`{benchmark['decode_tokens']}` tokens",
        "- Prompt：固定英文自然语言文本 tokenize 后重复并截断，所有请求相同。",
        f"- Tokenizer：`{benchmark['tokenizer_dir']}`",
        f"- 滑动窗口：`{benchmark['window_s']}` 秒，步长 `{benchmark['step_s']}` 秒",
        "- 边界：最慢请求第 2 个输出 token 至最快请求最后一个输出 token。",
        (
            "- Request TPOT：每条成功请求使用完整输出的首尾 token 时间跨度，"
            "除以输出 token 间隔数；不依赖吞吐窗口是否有效。"
        ),
        "",
        "## 聚合指标",
        "",
        "|指标|count|avg|min|max|stddev|p90|p99|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            "|Decode throughput (tok/s)|"
            f"{throughput['count']}|{format_metric(throughput['avg'])}|"
            f"{format_metric(throughput['min'])}|{format_metric(throughput['max'])}|"
            f"{format_metric(throughput['stddev'])}|{format_metric(throughput['p90'])}|"
            f"{format_metric(throughput['p99'])}|"
        ),
        (
            "|Request TPOT (ms)|"
            f"{request_tpot['count']}|{format_metric(request_tpot['avg'])}|"
            f"{format_metric(request_tpot['min'])}|"
            f"{format_metric(request_tpot['max'])}|"
            f"{format_metric(request_tpot['stddev'])}|"
            f"{format_metric(request_tpot['p90'])}|"
            f"{format_metric(request_tpot['p99'])}|"
        ),
        "",
        "## 复现",
        "",
        "```bash",
        "python scripts/bench_decode_sliding_window.py \\",
        f"  --base-url {benchmark['base_url']} \\",
        f"  --model {benchmark['model']} \\",
        "  --output-dir <OUTPUT_DIR> \\",
        f"  --concurrency {benchmark['concurrency']} \\",
        f"  --prefill-tokens {benchmark['prefill_tokens']} \\",
        f"  --decode-tokens {benchmark['decode_tokens']} \\",
        f"  --tokenizer-dir {benchmark['tokenizer_dir']} \\",
        f"  --rounds {requested_rounds} \\",
        f"  --window-seconds {benchmark['window_s']} \\",
        f"  --step-seconds {benchmark['step_s']}",
        "```",
        "",
        "## 轮次",
        "",
        (
            "|round|valid|requests|full overlap (s)|windows|"
            "throughput avg (tok/s)|Request TPOT avg (ms)|reason|"
        ),
        "|---:|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["results"]:
        lines.append(
            f"|{row['round']}|{row['valid']}|"
            f"{row['successful_requests']}/{benchmark['concurrency']}|"
            f"{format_metric(row['full_overlap_duration_s'])}|{row['window_count']}|"
            f"{format_metric(row['window_throughput_tok_s']['avg'])}|"
            f"{format_metric(row['request_tpot_ms']['avg'])}|{row['invalid_reason']}|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run identical concurrent prefill/decode requests and measure only "
            "their full-overlap decode interval with sliding windows."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18100")
    parser.add_argument("--model", default="Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=positive_int, default=16)
    parser.add_argument("--prefill-tokens", type=positive_int, default=65536)
    parser.add_argument("--decode-tokens", type=positive_int, default=1024)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--window-seconds", type=positive_float, default=10.0)
    parser.add_argument("--step-seconds", type=positive_float, default=1.0)
    parser.add_argument(
        "--request-timeout-seconds", type=positive_float, default=3600.0
    )
    parser.add_argument("--cooldown-seconds", type=nonnegative_float, default=2.0)
    args = parser.parse_args()
    if args.decode_tokens < 2:
        parser.error(
            "--decode-tokens must be at least 2 for the full-overlap boundary"
        )
    return args


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    health_response = requests.get(f"{base_url}/health", timeout=10)
    health_response.raise_for_status()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_requests.jsonl"
    raw_path.write_text("", encoding="utf-8")

    prompt, source_token_count = build_prompt_token_ids(
        tokenizer_dir=args.tokenizer_dir,
        target_tokens=args.prefill_tokens,
    )
    print(
        "[prompt] mode=fixed_natural_language "
        f"source_tokens={source_token_count} "
        f"request_tokens={len(prompt)}",
        flush=True,
    )
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.decode_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    request_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    endpoint = f"{base_url}/v1/completions"

    round_summaries: list[dict[str, Any]] = []
    all_windows: list[dict[str, Any]] = []
    all_request_tpots: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        print(
            f"[decode] round={round_index}/{args.rounds} "
            f"C={args.concurrency} P={args.prefill_tokens} D={args.decode_tokens}",
            flush=True,
        )
        results = run_round(
            endpoint=endpoint,
            request_body=request_body,
            concurrency=args.concurrency,
            timeout_s=args.request_timeout_seconds,
        )
        write_raw_requests(raw_path, round_index=round_index, results=results)
        analysis = analyze_round(
            round_index=round_index,
            results=results,
            concurrency=args.concurrency,
            decode_tokens=args.decode_tokens,
            window_s=args.window_seconds,
            step_s=args.step_seconds,
        )
        round_summaries.append(analysis.summary)
        all_windows.extend(analysis.windows)
        all_request_tpots.extend(analysis.request_tpots)
        print(
            f"[round-summary] valid={analysis.summary['valid']} "
            f"windows={analysis.summary['window_count']} "
            f"throughput_avg={analysis.summary['window_throughput_tok_s']['avg']} "
            f"request_tpot_avg_ms={analysis.summary['request_tpot_ms']['avg']}",
            flush=True,
        )
        if round_index < args.rounds and args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)

    valid_rounds = [row for row in round_summaries if row["valid"]]
    all_throughputs = [
        float(window["throughput_tok_s"]) for window in all_windows
    ]
    best_round = max(
        valid_rounds,
        key=lambda row: float(row["window_throughput_tok_s"]["avg"]),
        default=None,
    )
    summary = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "benchmark": {
            "base_url": base_url,
            "model": args.model,
            "input_length": args.prefill_tokens,
            "output_length": args.decode_tokens,
            "concurrency": args.concurrency,
            "prefill_tokens": args.prefill_tokens,
            "decode_tokens": args.decode_tokens,
            "prompt_mode": "fixed_natural_language_token_ids",
            "prompt_source_text": PROMPT_TEXT,
            "prompt_source_tokens": source_token_count,
            "tokenizer_dir": str(args.tokenizer_dir),
            "identical_prompt_for_all_requests": True,
            "window_s": args.window_seconds,
            "step_s": args.step_seconds,
            "boundary": "slowest_second_token_to_fastest_last_token",
        },
        "best": (
            {
                "completed": best_round["successful_requests"],
                "concurrency": args.concurrency,
                "failed": best_round["failed_requests"],
                "file": "rounds.csv",
                "mean_tpot_ms": best_round["request_tpot_ms"]["avg"],
                "output_throughput": best_round["window_throughput_tok_s"]["avg"],
                "p99_tpot_ms": best_round["request_tpot_ms"]["p99"],
                "round": best_round["round"],
            }
            if best_round is not None
            else None
        ),
        "requested_rounds": args.rounds,
        "valid_rounds": len(valid_rounds),
        "aggregate": {
            "window_throughput_tok_s": metric_stats(all_throughputs),
            "request_tpot_ms": metric_stats(
                [float(row["tpot_ms"]) for row in all_request_tpots]
            ),
            "round_avg_throughput_tok_s": metric_stats(
                [float(row["window_throughput_tok_s"]["avg"]) for row in valid_rounds]
            ),
            "round_avg_request_tpot_ms": metric_stats(
                [
                    float(row["request_tpot_ms"]["avg"])
                    for row in round_summaries
                    if row["request_tpot_ms"]["avg"] is not None
                ]
            ),
        },
        "results": round_summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_rounds_csv(args.output_dir / "rounds.csv", round_summaries)
    write_timeline_csv(args.output_dir / "timeline.csv", all_windows)
    write_request_tpot_csv(
        args.output_dir / "request_tpot.csv", all_request_tpots
    )
    write_markdown(args.output_dir / "summary.md", summary)

    if len(valid_rounds) != args.rounds:
        raise SystemExit(
            f"decode benchmark has {len(valid_rounds)}/{args.rounds} valid rounds"
        )
    aggregate = summary["aggregate"]
    print(
        "[done] "
        f"throughput_avg={aggregate['window_throughput_tok_s']['avg']:.3f} tok/s "
        f"request_tpot_avg={aggregate['request_tpot_ms']['avg']:.3f} ms "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
