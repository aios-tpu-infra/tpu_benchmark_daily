# TPU daily benchmark

## TL;DR

本项目每日顺序执行三组 Qwen3.5-397B-A17B-FP8 benchmark：真实权重
TP1/DP8/EP8 C256 decode、dummy-weight DP8 prefill 和 dummy-weight PCP8
prefill。Decode 使用 C256/P65536/D1024、独立请求前缀、三轮 10 秒滑窗，
按实际 peak-active plateau 统计；prefill 保持原有口径，避免破坏历史趋势。

三组服务统一使用项目内 `models/Qwen3.5-397B-A17B-FP8`。Decode 从该目录
加载完整权重；DP8/PCP8 prefill 对同一目录使用 `--load-format dummy`。

## Recent benchmark throughput

<!-- BENCHMARK_REPORT_START -->
Latest DP8 vs PCP8 throughput by concurrency:

![Latest DP8 vs PCP8 throughput by concurrency](reports/throughput.svg)

Recent DP8 vs PCP8 peak throughput over time:

![Recent DP8 vs PCP8 peak throughput over time](reports/throughput_history.svg)

Recent DP8 decode throughput over time:

![Recent DP8 decode throughput over time](reports/decode_throughput_history.svg)

Latest DP8: **50,460.53 total tok/s** at concurrency **32** (`20260727T072636Z`).
Latest PCP8: **44,029.38 total tok/s** at concurrency **16** (`20260727T072636Z`).

| vllm-torchtpu commit | Test time (UTC) | DP peak prefill tok/s | PCP peak prefill tok/s | DP decode tok/s | DP decode TPOT (ms) | Decode protocol |
|---|---|---:|---:|---:|---:|---|
| `c908cf61549d` | 2026-07-27 07:26 | 50,460.53 | 44,029.38 | -1.00 | — | failed |
| `0df7f8451750` | 2026-07-25 18:00 | 52,548.29 | 43,961.81 | 3,937.43 | 46.68 | C256 peak-active P50 |
| `d4c736067443` | 2026-07-24 18:00 | 52,008.96 | 43,102.49 | 3,956.50 | 46.45 | C256 peak-active P50 |
| `d87b06bee7c4` | 2026-07-24 00:32 | 51,494.71 | 34,121.60 | 632.24 | 18.41 | legacy peak/min |
| `a03d8effc78a` | 2026-07-23 08:34 | 51,458.93 | 34,291.21 | 637.32 | 18.72 | legacy peak/min |
| `a03d8effc78a` | 2026-07-23 06:45 | 51,476.87 | 34,276.38 | 637.69 | 20.51 | legacy peak/min |
| — | 2026-07-22 08:15 | 48,359.05 | — | — | — | — |
| `db5ae0ab3941` | 2026-07-22 05:07 | — | 34,296.71 | — | — | — |
| `db5ae0ab3941` | 2026-07-22 01:40 | 46,240.26 | — | — | — | — |
| `db0149493e41` | 2026-07-21 18:00 | 40,378.43 | — | — | — | — |

Failed benchmark groups are recorded as -1 tok/s in the table and JSON/CSV reports, while charts plot successful measurements only. The prefill charts compare DP8 and PCP8 throughput and track their recent peaks. The decode chart keeps legacy peak-output and current peak-active P50 statistics in separate series; see [`reports/latest.json`](reports/latest.json) for the newest peaks and [`reports/throughput_history.json`](reports/throughput_history.json) for the full history.
<!-- BENCHMARK_REPORT_END -->

## Layout

- `third_party/torchtpu-vllm/`: `vllm-project/vllm-torchtpu` Git submodule,
  refreshed from `origin/main` (the local path is retained for compatibility).
- `models/`: offline model metadata and locally provisioned checkpoint weights;
  checkpoint files are excluded from Git.
- `scripts/start_dp_decode_server.sh`: starts the real-weight TP1/DP8/EP8
  C256 decode service with unified pool, 4352-token pages, GMU 0.96, async
  scheduling, GDN v3, prefix cache disabled, and the same compile shapes as
  the current standalone C256 test.
- `scripts/start_dp_server.sh`: starts the DP8/PCP1 vLLM server with dummy weights.
- `scripts/start_pcp_server.sh`: starts the DP1/PCP8 vLLM server with dummy weights.
- `scripts/bench_all.sh`: benchmarks input length 8192 at concurrency 8–256 for
  the configuration selected by `BENCHMARK_CONFIG`.
- `scripts/update_environment.sh`: updates `vllm-torchtpu`, installs its
  compatible `torch_tpu` wheel from Google Artifact Registry with pip, then
  synchronizes the rest of the project `.venv`.
- `scripts/daily_benchmark.sh`: complete locked cron workflow.
- `reports/`: durable peak-throughput history and generated SVG charts.
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

完整 daily run 要求 `models/Qwen3.5-397B-A17B-FP8` 中存在真实模型权重。
权重文件由环境在本地提供并被 Git 忽略；`--prepare-only` 和三个启动脚本
只执行与 DP8/PCP8 一致的 config/tokenizer 元数据检查，decode 加载权重时
仍会由模型加载器验证权重是否完整。

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

To run only one benchmark group, use `--only`:

```bash
scripts/daily_benchmark.sh --only dp-decode
scripts/daily_benchmark.sh --only dp-prefill
scripts/daily_benchmark.sh --only pcp-prefill
```

Omitting `--only` preserves the full three-group workflow. A selective run
updates the environment in the same way as a full run, but validates and starts
only the service required by the selected benchmark. DP8/PCP8 prefill-only
runs use `--load-format dummy`, so they do not require real checkpoint weights
in the shared model directory.

Every selected benchmark group is reported and published unless
`PUBLISH_REPORTS=0`, including decode-only runs. A failed group records
throughput `-1`, does not prevent later selected groups from running, and still
participates in the final report publication. The runner returns a nonzero exit
status after publication when any selected group failed. With
`--keep-server-running`, the server for a fully successful selective run is kept.

Before updating or building, the full workflow stops an existing vLLM API
server listening on `PORT` (18100 by default), including its worker process
group. A non-vLLM process on that port is never killed and causes the job to
fail safely. `--prepare-only` leaves any running service untouched.

The runner first starts the real-weight DP8 C256 decode service, runs one
C8/P65536/D32 smoke process followed by three independent
C256/P65536/D1024 processes, and stops it. It then starts the existing
dummy-weight DP8 and PCP8 services one at a time for their prefill suites. The
complete run therefore contains three benchmark groups: DP8 C256 decode, DP8
prefill, and PCP8 prefill. Each group is isolated so a startup or benchmark
failure is recorded before the runner advances to the next group. Servers are
stopped after the benchmark by default.
Use `--keep-server-running` only for interactive debugging; when successful,
it keeps the final PCP8 server alive.

All three services use `models/Qwen3.5-397B-A17B-FP8` as their model
directory. The decode service uses the real checkpoint from that directory;
DP8 and PCP8 prefill continue to pass `--load-format dummy`.

Decode 每条请求使用不同但可重复生成的自然语言前缀和独立 `cache_salt`，因此
不依赖 prefix cache 的跨请求复用；client 通过 `X-data-parallel-rank` 将
256 条请求按 `request_id % 8` 精确均分为每个 DP rank 32 条。请求直接并发
提交，不使用服务端 admission barrier。客户端通过 `requests.post(json=...)`
发起 streaming 请求，并从 cumulative `usage.completion_tokens` 展开 token
时间线，与当前 tpu-misc C256 记录使用相同计数口径。

若 256 条请求没有形成完整 10 秒重叠区间，各轮
`run_<N>/summary.json` 中的 `active_requests_max` 和
`timeline_valid_full_concurrency_decode` 会记录实际峰值并发，主吞吐取该峰值
并发平台窗口的 P50；主 TPOT 统计相同 peak-active 时间范围内的 token 间隔，
逐请求全生命周期 TPOT 作为补充。三轮结果均为独立 Python 进程，最终
`aggregate.json`/`aggregate.csv` 对每轮指标计算 `count`、`avg`、`min`、
`max`、sample `stddev`、`p90` 和 `p99`；daily 报告使用三轮
peak-active window throughput P50 的平均值，以及三轮 peak-active TPOT P50
的平均值。

三组测试的结果使用同级目录；两个 prefill 目录在根部提供 `summary.json`，
decode 目录保存 smoke、三轮独立 summary 及跨轮 aggregate：

```text
runs/<UTC timestamp>/results/
├── dp8_decode_c256/
│   ├── smoke/
│   │   └── summary.json
│   ├── run_1/
│   │   ├── summary.json
│   │   ├── timeline.csv
│   │   ├── request_tpot.csv
│   │   └── raw_requests.jsonl
│   ├── run_2/
│   │   └── ...
│   ├── run_3/
│   │   └── ...
│   ├── aggregate.json
│   └── aggregate.csv
├── dp8/
│   ├── summary.json
│   └── vllm_dp8_tp1_len8192_c*.json
└── pcp8/
    ├── summary.json
    └── vllm_pcp8_tp1_len8192_c*.json
```

Decode、DP8 prefill 和 PCP8 prefill 的原始摘要相互独立；报告生成器显式读取
`dp8_decode_c256/aggregate.json`，不会再把 decode 原始结果嵌套进
`dp8/summary.json`。

The three benchmark groups share one run ID and one workflow start timestamp.
The homepage combines them into one table row and uses that shared start time
for `Test time (UTC)`.

After all selected benchmark groups finish, the runner records the highest
`total_token_throughput` separately for successful DP8 and PCP8 groups and `-1`
for failed groups, regenerates the concurrency and time-series SVG charts, then
commits `README.md` and `reports/` once and pushes that commit directly to
`origin/main`. Failed values remain visible in the table and JSON/CSV reports,
while trend charts plot successful measurements only. Set `PUBLISH_REPORTS=0`
to disable commit and push for a local-only run.

The most recent local DP8 and PCP8 peaks are available in `reports/latest.json`.
The generated images are `reports/throughput.svg`,
`reports/throughput_history.svg`, and
`reports/decode_throughput_history.svg`; every successful report update
replaces all three files atomically. Decode history renders the legacy
peak-output metric and the current C256 peak-active window P50 metric as
separate series because the two statistics are not directly comparable.
Automatic publication uses the repository's configured Git SSH credentials. It
refuses to run when `main` differs from the remote, the index is not empty, or
unrelated project files are modified. The `vllm-torchtpu` submodule pointer may
be modified by its daily update, but it is never included in the generated-report
commit.

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
