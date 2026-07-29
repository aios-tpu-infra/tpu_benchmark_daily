#!/usr/bin/env python3

"""Expand one captured vLLM TTFT sample into fixed serial test-only samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PER_REQUEST_FIELDS = (
    "input_lens",
    "output_lens",
    "ttfts",
    "itls",
    "errors",
    "generated_texts",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def expand_fixture(data: dict[str, Any], samples: int) -> dict[str, Any]:
    if int(data.get("completed", 0)) != 1 or int(data.get("failed", 0)) != 0:
        raise ValueError("TTFT source fixture must contain one successful request")

    expanded = dict(data)
    for field in PER_REQUEST_FIELDS:
        values = data.get(field)
        if values is None:
            continue
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"source fixture field {field!r} must contain one item")
        expanded[field] = values * samples

    input_lens = expanded.get("input_lens")
    output_lens = expanded.get("output_lens")
    if not isinstance(input_lens, list) or not isinstance(output_lens, list):
        raise ValueError("source fixture must contain input_lens and output_lens")

    expanded["num_prompts"] = samples
    expanded["completed"] = samples
    expanded["failed"] = 0
    expanded["total_input_tokens"] = sum(int(value) for value in input_lens)
    expanded["total_output_tokens"] = sum(int(value) for value in output_lens)
    if "duration" in data:
        duration = float(data["duration"]) * samples
        expanded["duration"] = duration
        expanded["request_throughput"] = samples / duration
        expanded["total_token_throughput"] = (
            expanded["total_input_tokens"] + expanded["total_output_tokens"]
        ) / duration
    expanded["fixture_note"] = (
        f"{data.get('fixture_note', '')} "
        f"TEST_ONLY repeats the captured sample {samples} times serially."
    ).strip()
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--samples", type=positive_int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expanded = expand_fixture(load_json(args.source), args.samples)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(
        json.dumps(expanded, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
