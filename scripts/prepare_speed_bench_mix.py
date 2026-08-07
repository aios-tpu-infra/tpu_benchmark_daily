#!/usr/bin/env python3

"""Build a deterministic, variable-length prompt mix from public SPEED-Bench."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence


SUBSETS = (
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
)
PLACEHOLDER = (
    "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
)
PADDING_RE = re.compile(r"(?:Answer now please\.(?:\r?\n|$))+", re.MULTILINE)
SOURCE_REVISION = "487aa718444e816458d1a0a52bfce7a454285cf4"


@dataclass(frozen=True)
class Candidate:
    prompt: str
    prompt_sha256: str
    question_id: str
    subset: str
    category: str
    sub_category: str
    source: str
    source_id: str
    raw_prompt_tokens: int
    input_tokens: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_prompt(turns: Iterable[Any]) -> str | None:
    prompt = "\n".join(str(turn) for turn in turns)
    if PLACEHOLDER in prompt:
        return None
    prompt = PADDING_RE.sub("", prompt).rstrip()
    return prompt or None


def _quantile_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    return [round((length - 1) * (index + 1) / (count + 1)) for index in range(count)]


def select_balanced(candidates: Sequence[Candidate], count: int) -> list[Candidate]:
    """Select stable token-length quantiles while balancing SPEED categories."""
    if count <= 0:
        raise ValueError("count must be positive")
    if len(candidates) < count:
        raise ValueError(f"need {count} candidates, found {len(candidates)}")

    by_category: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_category[candidate.category].append(candidate)
    categories = sorted(by_category)
    if not categories:
        raise ValueError("no candidate categories found")

    allocation = {category: count // len(categories) for category in categories}
    for category in categories[: count % len(categories)]:
        allocation[category] += 1

    selected: list[Candidate] = []
    selected_hashes: set[str] = set()
    for category in categories:
        group = sorted(
            by_category[category],
            key=lambda item: (item.input_tokens, item.question_id, item.prompt_sha256),
        )
        requested = allocation[category]
        if requested > len(group):
            requested = len(group)
        for index in _quantile_indices(len(group), requested):
            candidate = group[index]
            if candidate.prompt_sha256 not in selected_hashes:
                selected.append(candidate)
                selected_hashes.add(candidate.prompt_sha256)

    if len(selected) < count:
        remaining = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.prompt_sha256 not in selected_hashes
            ),
            key=lambda item: (item.input_tokens, item.question_id, item.prompt_sha256),
        )
        needed = count - len(selected)
        for index in _quantile_indices(len(remaining), needed):
            candidate = remaining[index]
            if candidate.prompt_sha256 not in selected_hashes:
                selected.append(candidate)
                selected_hashes.add(candidate.prompt_sha256)

    if len(selected) != count:
        raise ValueError(f"selected {len(selected)} prompts, expected {count}")
    return sorted(
        selected,
        key=lambda item: (item.input_tokens, item.question_id, item.prompt_sha256),
    )


def select_candidates(
    candidates: Sequence[Candidate], count: int | None
) -> list[Candidate]:
    """Select a sample, or return every eligible prompt in stable order."""
    if count is None:
        return sorted(
            candidates,
            key=lambda item: (
                item.input_tokens,
                item.question_id,
                item.prompt_sha256,
            ),
        )
    return select_balanced(candidates, count)


def select_random(
    candidates: Sequence[Candidate], count: int, seed: int
) -> list[Candidate]:
    """Uniformly sample candidates in an input-order-independent way."""
    if count <= 0:
        raise ValueError("count must be positive")
    if len(candidates) < count:
        raise ValueError(f"need {count} candidates, found {len(candidates)}")
    ordered = sorted(
        candidates,
        key=lambda item: (item.subset, item.question_id, item.prompt_sha256),
    )
    selected = random.Random(seed).sample(ordered, count)
    return sorted(
        selected,
        key=lambda item: (
            item.subset,
            item.input_tokens,
            item.question_id,
            item.prompt_sha256,
        ),
    )


def build_candidates(
    rows: Iterable[dict[str, Any]],
    subset: str,
    tokenizer: Any,
    min_input_tokens: int,
    max_input_tokens: int,
    seen_hashes: set[str],
) -> tuple[list[Candidate], dict[str, int]]:
    counts = {
        "rows": 0,
        "placeholders": 0,
        "empty_after_cleanup": 0,
        "duplicates": 0,
        "outside_token_range": 0,
        "eligible": 0,
    }
    candidates: list[Candidate] = []
    for row in rows:
        counts["rows"] += 1
        prompt = clean_prompt(row.get("turns") or [])
        if prompt is None:
            original = "\n".join(str(turn) for turn in row.get("turns") or [])
            if PLACEHOLDER in original:
                counts["placeholders"] += 1
            else:
                counts["empty_after_cleanup"] += 1
            continue

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_hash in seen_hashes:
            counts["duplicates"] += 1
            continue

        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        input_tokens = len(tokenizer(rendered).input_ids)
        if not min_input_tokens <= input_tokens <= max_input_tokens:
            counts["outside_token_range"] += 1
            continue

        raw_prompt_tokens = len(tokenizer(prompt).input_ids)
        seen_hashes.add(prompt_hash)
        candidates.append(
            Candidate(
                prompt=prompt,
                prompt_sha256=prompt_hash,
                question_id=str(row.get("question_id", "")),
                subset=subset,
                category=str(row.get("category", "")),
                sub_category=str(row.get("sub_category", "")),
                source=str(row.get("source", "")),
                source_id=str(row.get("src_id", "")),
                raw_prompt_tokens=raw_prompt_tokens,
                input_tokens=input_tokens,
            )
        )
        counts["eligible"] += 1
    return candidates, counts


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_root / "datasets" / "speed_bench",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "datasets" / "speed_bench_mix",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--samples-per-subset", type=int, default=4)
    selection_group.add_argument(
        "--all-eligible",
        action="store_true",
        help="include every cleaned, deduplicated prompt in the token range",
    )
    selection_group.add_argument(
        "--random-sample-total",
        type=int,
        help="uniformly sample this many prompts across all eligible subsets",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--min-input-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=258048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        not args.all_eligible
        and args.random_sample_total is None
        and args.samples_per_subset <= 0
    ):
        raise SystemExit("--samples-per-subset must be positive")
    if args.random_sample_total is not None and args.random_sample_total <= 0:
        raise SystemExit("--random-sample-total must be positive")
    if args.output_tokens <= 0:
        raise SystemExit("--output-tokens must be positive")
    if args.min_input_tokens <= 0 or args.max_input_tokens < args.min_input_tokens:
        raise SystemExit("invalid input token range")

    try:
        import pyarrow.parquet as parquet
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "pyarrow and transformers are required; run this script with the project venv"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    output_records: list[dict[str, Any]] = []
    selected_candidates: list[Candidate] = []
    input_files: list[dict[str, Any]] = []
    subset_stats: dict[str, Any] = {}
    seen_hashes: set[str] = set()

    for subset in SUBSETS:
        paths = sorted((args.source_dir / subset).glob("test-*.parquet"))
        if len(paths) != 1:
            raise SystemExit(
                f"expected one parquet file for {subset}, found {len(paths)}"
            )
        path = paths[0]
        input_files.append(
            {
                "path": str(path.relative_to(args.source_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        candidates, counts = build_candidates(
            parquet.read_table(path).to_pylist(),
            subset,
            tokenizer,
            args.min_input_tokens,
            args.max_input_tokens,
            seen_hashes,
        )
        selected = select_candidates(
            candidates,
            (
                None
                if args.all_eligible or args.random_sample_total is not None
                else args.samples_per_subset
            ),
        )
        subset_stats[subset] = counts
        selected_candidates.extend(selected)

    if args.random_sample_total is not None:
        selected_candidates = select_random(
            selected_candidates,
            args.random_sample_total,
            args.random_seed,
        )

    for subset in SUBSETS:
        subset_selection = [
            item for item in selected_candidates if item.subset == subset
        ]
        subset_stats[subset] = {
            **subset_stats[subset],
            "selected": len(subset_selection),
            "selected_input_tokens": [
                item.input_tokens for item in subset_selection
            ],
        }
    for candidate in selected_candidates:
        record = asdict(candidate)
        record["output_tokens"] = args.output_tokens
        output_records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "requests.jsonl.gz"
    dataset_digest = hashlib.sha256()
    uncompressed_bytes = 0
    with archive_path.open("wb") as archive_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=archive_handle,
            mtime=0,
        ) as handle:
            for record in output_records:
                line = (
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                ).encode("utf-8")
                dataset_digest.update(line)
                uncompressed_bytes += len(line)
                handle.write(line)

    legacy_output_path = args.output_dir / "requests.jsonl"
    if legacy_output_path.exists():
        legacy_output_path.unlink()

    dataset_sha256 = dataset_digest.hexdigest()
    artifact_sha256 = sha256_file(archive_path)
    input_lengths = [int(record["input_tokens"]) for record in output_records]
    manifest = {
        "schema_version": 1,
        "source": "nvidia/SPEED-Bench",
        "source_revision": SOURCE_REVISION,
        "source_files": input_files,
        "tokenizer": args.model_dir.name,
        "selection": {
            "subsets": list(SUBSETS),
            "mode": (
                "all_eligible"
                if args.all_eligible
                else (
                    "random_sample"
                    if args.random_sample_total is not None
                    else "balanced_sample"
                )
            ),
            "samples_per_subset": (
                None
                if args.all_eligible or args.random_sample_total is not None
                else args.samples_per_subset
            ),
            "random_sample_total": args.random_sample_total,
            "random_seed": (
                args.random_seed
                if args.random_sample_total is not None
                else None
            ),
            "category_balanced": (
                not args.all_eligible and args.random_sample_total is None
            ),
            "length_selection": (
                "all eligible prompts sorted by input length"
                if args.all_eligible
                else (
                    "uniform random sample across all eligible prompts"
                    if args.random_sample_total is not None
                    else "even quantiles within each category"
                )
            ),
            "placeholder_rows": "filtered",
            "padding_removed": "Answer now please.",
            "deduplication": "global SHA-256 of cleaned prompt",
            "min_input_tokens": args.min_input_tokens,
            "max_input_tokens": args.max_input_tokens,
            "output_tokens": args.output_tokens,
        },
        "dataset": {
            "path": archive_path.name,
            "format": "jsonl+gzip",
            "sha256": dataset_sha256,
            "artifact_sha256": artifact_sha256,
            "uncompressed_bytes": uncompressed_bytes,
            "compressed_bytes": archive_path.stat().st_size,
            "requests": len(output_records),
            "total_input_tokens": sum(input_lengths),
            "min_input_tokens": min(input_lengths),
            "max_input_tokens": max(input_lengths),
            "input_tokens": input_lengths,
        },
        "subsets": subset_stats,
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {len(output_records)} requests to {archive_path}")
    print(
        "Input tokens: "
        f"total={sum(input_lengths)}, min={min(input_lengths)}, "
        f"max={max(input_lengths)}"
    )
    print(f"Dataset SHA-256: {dataset_sha256}")
    print(f"Artifact SHA-256: {artifact_sha256}")


if __name__ == "__main__":
    main()
