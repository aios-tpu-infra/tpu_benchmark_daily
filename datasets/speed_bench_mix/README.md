# SPEED-Bench semantic mixed-length workload

`requests.jsonl` is a deterministic 20-request derivative of the public
`nvidia/SPEED-Bench` snapshot at revision
`487aa718444e816458d1a0a52bfce7a454285cf4`. It contains four requests from
each of the 1K, 2K, 8K, 16K, and 32K throughput subsets, balanced across the
source categories and spread across token-length quantiles.

The preparation step filters placeholder rows, removes the repeated
`Answer now please.` length padding, deduplicates cleaned prompts, and measures
the final input lengths with the Qwen3.5-397B-A17B-FP8 chat template. The
manifest pins source-file hashes, the generated dataset hash, and all exact
input lengths used by result validation.

To reproduce the checked-in files from a raw snapshot at
`datasets/speed_bench/`:

```bash
.venv/bin/python scripts/prepare_speed_bench_mix.py \
  --model-dir /home/xiaohao_yxh/workspace/models/Qwen3.5-397B-A17B-FP8
```

The raw snapshot and its license/README remain the source of truth for usage
terms and attribution; they are not required by the daily runner.
