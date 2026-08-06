# SPEED-Bench semantic mixed-length workload

`requests.jsonl.gz` is a deterministic 1,000-request derivative of the public
`nvidia/SPEED-Bench` snapshot at revision
`487aa718444e816458d1a0a52bfce7a454285cf4`. After building the 4,194 eligible
requests from the 1K, 2K, 8K, 16K, and 32K throughput subsets, it uniformly
samples 1,000 requests across the combined pool with seed 42.

The preparation step filters placeholder rows, removes the repeated
`Answer now please.` length padding, deduplicates cleaned prompts, and measures
the final input lengths with the Qwen3.5-397B-A17B-FP8 chat template. The
manifest pins source-file hashes, the uncompressed dataset hash, the compressed
artifact hash, and all exact input lengths used by result validation. The
compressed artifact keeps the 46,043,978-byte JSONL dataset compact; the
14,168,852-byte gzip file is stored directly in Git and remains beneath the
normal 100 MiB per-file limit. The runner verifies and decompresses it into the
run directory before invoking vLLM.

To reproduce the checked-in files from a raw snapshot at
`datasets/speed_bench/`:

```bash
.venv/bin/python scripts/prepare_speed_bench_mix.py \
  --model-dir /home/xiaohao_yxh/workspace/models/Qwen3.5-397B-A17B-FP8 \
  --random-sample-total 1000 \
  --random-seed 42
```

The raw snapshot and its license/README remain the source of truth for usage
terms and attribution; they are not required by the daily runner.
