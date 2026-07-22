# TPU daily benchmark

This project runs a reproducible Qwen3.5-397B-A17B-FP8 throughput benchmark on
TorchTPU/vLLM. Model weights are replaced with vLLM dummy weights; the checked-in
`models/` directory contains only an offline configuration/tokenizer snapshot.

## Recent benchmark throughput

<!-- BENCHMARK_REPORT_START -->
[![Recent peak throughput](reports/throughput.svg)](reports/index.html)

Latest successful run: **46,240.26 total tok/s** at concurrency **16** (`20260722T014057Z`).

| Completed (UTC) | Peak total tok/s | Best concurrency | Requests/s | p99 TTFT (ms) |
|---|---:|---:|---:|---:|
| 2026-07-22 02:28 | 46,240.26 | 16 | 5.644 | 3,738.4 |
| 2026-07-21 18:39 | 40,378.43 | 64 | 4.928 | 12,381.2 |
| 2026-07-20 18:16 | 43,690.58 | 64 | 5.333 | 11,976.0 |
| 2026-07-19 18:16 | 44,436.44 | 64 | 5.424 | 11,778.4 |
| 2026-07-18 18:17 | 44,397.93 | 64 | 5.419 | 11,780.6 |
| 2026-07-18 00:01 | 44,371.29 | 64 | 5.416 | 11,793.6 |
| 2026-07-17 09:18 | 49,360.61 | 64 | 6.025 | 10,597.1 |
| 2026-07-17 08:58 | 49,372.29 | 64 | 6.026 | 10,591.7 |
| 2026-07-17 08:05 | 49,381.83 | 64 | 6.027 | 10,594.9 |

The chart shows successful runs only; see [`reports/latest.json`](reports/latest.json) for the newest peak and [`reports/throughput_history.json`](reports/throughput_history.json) for the full history.
<!-- BENCHMARK_REPORT_END -->

## Layout

- `third_party/torchtpu-vllm/`: `vllm-project/vllm-torchtpu` Git submodule,
  refreshed from `origin/main` (the local path is retained for compatibility).
- `models/`: offline model metadata; no checkpoint weights.
- `scripts/run.sh`: starts the vLLM server with `--load-format dummy`.
- `scripts/bench_all.sh`: benchmarks input length 8192 at concurrency 1–64.
- `scripts/update_environment.sh`: updates `vllm-torchtpu`, installs its
  compatible `torch_tpu` wheel from Google Artifact Registry with pip, then
  synchronizes the rest of the project `.venv`.
- `scripts/daily_benchmark.sh`: complete locked cron workflow.
- `reports/`: durable peak-throughput history, README chart, and static dashboard.
- `runs/`: timestamped logs, environment snapshots, and benchmark JSON files.

## First preparation

The machine needs `git`, `uv`, the Google Cloud CLI, and Python 3.12. SSH access
to GitHub is required for the `vllm-torchtpu` submodule. The active gcloud user
must have read access to the private `torch-tpu` Artifact Registry. Authenticate
that user before the first run:

```bash
gcloud auth login
gcloud auth list --filter=status:ACTIVE
```

The installer deliberately uses `gcloud auth print-access-token` instead of
Application Default Credentials because these can represent different users or
permissions. For non-gcloud automation, `TORCH_TPU_ACCESS_TOKEN` can provide a
short-lived token explicitly.

Run:

```bash
scripts/daily_benchmark.sh --prepare-only
```

Each invocation fetches the latest `vllm-torchtpu/main`, reads its exact
compatible `torch` and `torch-tpu` pins, and runs pip against the private
`torch-tpu` virtual registry. Installing the exact `torch` pin first is
intentional: `torch-tpu` alone has a broad dependency constraint that can select
a newer ABI-incompatible PyTorch build. The downloaded `torch-tpu` wheel is
force-reinstalled so a previous source-built wheel cannot remain in `.venv`.
The remaining dependencies are then synchronized with `uv`.

## Manual full run

```bash
scripts/daily_benchmark.sh
```

Before updating or building, the full workflow stops an existing vLLM API
server listening on `PORT` (18100 by default), including its worker process
group. A non-vLLM process on that port is never killed and causes the job to
fail safely. `--prepare-only` leaves any running service untouched.

The newly started server is stopped after the benchmark by default. Use
`--keep-server-running` only for interactive debugging.

After every successful full benchmark, the runner records the highest
`total_token_throughput`, regenerates the chart and dashboard, then commits
`README.md` and `reports/` and pushes that commit directly to `origin/main`. The
GitHub repository homepage therefore shows the latest curve without a separate
web service. Set `PUBLISH_REPORTS=0` to disable commit and push for a local-only
run.

The most recent local peak is available by itself in `reports/latest.json`.
Open the full local dashboard directly with a browser, or serve it locally:

```bash
python3 -m http.server 8000 --directory reports
```

Then visit `http://127.0.0.1:8000/`. Automatic publication uses the repository's
configured Git SSH credentials. It refuses to run when `main` differs from the
remote, the index is not empty, or unrelated project files are modified. The
`vllm-torchtpu` submodule pointer may be modified by its daily update, but it is
never included in the generated-report commit.

## Example crontab

Run every day at 02:00 UTC:

```cron
0 2 * * * /bin/bash /mnt/data/xiaohao/workspace/tpu_benchmark_daily/scripts/daily_benchmark.sh
```

The runner uses absolute project paths internally, takes an exclusive `flock`,
and writes all output beneath `runs/<UTC timestamp>/`. The exact
`vllm-torchtpu` revision, pip-installed `torch_tpu` version, and machine IP are
saved in each run's `run_metadata.json`. Set `MACHINE_IP` to override automatic
primary-address detection when the machine has multiple network interfaces.
