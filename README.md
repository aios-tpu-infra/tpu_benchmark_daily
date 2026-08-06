# TPU daily benchmark

## TL;DR

本项目每日顺序执行三组 Qwen3.5-397B-A17B-FP8 真实权重 benchmark：
TP1/DP8/EP8 C256 decode、DP8 prefill 和 PCP8 prefill。Decode 使用
C256/P65536/D1024、独立请求前缀、三轮 10 秒滑窗，按实际 peak-active
plateau 统计。DP8 和 PCP8 prefill 服务还分别测试并发度 1、输入长度
8K/16K/32K/64K/128K/252K、输出长度 1 的 TTFT；每档串行执行
16 条 measured requests，并展示 median TTFT。

DP8 prefill 还会额外运行一组固定的真实语义变长请求：从公开的 NVIDIA
SPEED-Bench 快照中确定性选取 20 条、移除人为长度填充后，输入长度覆盖
1,038–32,982 tokens。该 workload 分别记录并发度 8 的 input/total token
吞吐，以及并发度 1、固定 DP rank 的 median/P90/P99 TTFT。PCP 的变长路径
目前存在已知实现问题，因此这组测试暂时只在 DP8 上运行。

三组服务统一从项目内 `models/Qwen3.5-397B-A17B-FP8` 加载完整 checkpoint，
不再使用 `--load-format dummy`。由于真实权重与此前 dummy-weight 结果不可
直接比较，报告历史已从首轮真实权重测试 `20260728T012922Z` 重新开始。

## Recent benchmark throughput

<!-- BENCHMARK_REPORT_START -->
Latest DP8 vs PCP8 throughput by concurrency:

![Latest DP8 vs PCP8 throughput by concurrency](reports/throughput.svg)

Latest DP8 vs PCP8 single-request prefill TTFT by input length:

![Latest DP8 vs PCP8 single-request prefill TTFT](reports/prefill_ttft.svg)

Recent DP8 vs PCP8 peak throughput over time:

![Recent DP8 vs PCP8 peak throughput over time](reports/throughput_history.svg)

Recent DP8 decode throughput over time:

![Recent DP8 decode throughput over time](reports/decode_throughput_history.svg)

Latest DP8: **52,337.11 total tok/s** at concurrency **32** (`20260806T013022Z`).
Latest PCP8: **48,507.94 total tok/s** at concurrency **16** (`20260806T040306Z`).

Latest DP8 single-request TTFT: **success**, **16 serial samples/length** (`20260806T013022Z`).
Latest PCP8 single-request TTFT: **success**, **16 serial samples/length** (`20260806T040306Z`).

| vllm-torchtpu commit | Test time (UTC) | DP peak prefill tok/s | PCP peak prefill tok/s | DP decode tok/s | DP decode TPOT (ms) | Decode protocol | DP TTFT 8K (ms) | PCP TTFT 8K (ms) | DP TTFT 16K (ms) | PCP TTFT 16K (ms) | DP TTFT 32K (ms) | PCP TTFT 32K (ms) | DP TTFT 64K (ms) | PCP TTFT 64K (ms) | DP TTFT 128K (ms) | PCP TTFT 128K (ms) | DP TTFT 252K (ms) | PCP TTFT 252K (ms) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f53d6300e29f` | 2026-08-06 04:03 | — | 48,507.94 | — | — | — | — | 631.84 | — | 712.95 | — | 918.13 | — | 1,981.19 | — | 4,482.04 | — | 10,764.83 |
| `f53d6300e29f` | 2026-08-06 04:02 | — | -1.00 | — | — | — | — | failed | — | failed | — | failed | — | failed | — | failed | — | failed |
| `f53d6300e29f` | 2026-08-06 02:43 | — | 48,474.69 | — | — | — | — | 633.27 | — | 702.56 | — | 926.37 | — | 1,962.64 | — | 4,447.49 | — | 10,728.04 |
| `f53d6300e29f` | 2026-08-06 01:30 | 52,337.11 | — | — | — | — | 1,350.46 | — | 2,770.51 | — | 5,812.90 | — | 12,695.16 | — | 29,747.25 | — | 75,101.33 | — |
| `f53d6300e29f` | 2026-08-06 00:15 | 49,536.95 | — | — | — | — | 1,478.83 | — | 3,020.83 | — | 6,323.41 | — | 13,736.61 | — | 31,791.95 | — | 79,113.26 | — |
| `2060adda4744` | 2026-08-05 16:00 | 49,042.41 | 45,165.61 | 4,866.90 | 42.67 | C256 peak-active P50 | 1,477.95 | 691.67 | 3,022.87 | 765.35 | 6,320.08 | 972.52 | 13,718.80 | 2,078.42 | 31,795.93 | 4,668.80 | 79,095.38 | 11,120.77 |
| `da465492d4df` | 2026-08-04 16:00 | 48,809.66 | 45,543.51 | 4,881.97 | 42.57 | C256 peak-active P50 | 1,476.04 | 685.02 | 3,021.77 | 756.65 | 6,319.50 | 961.64 | 13,723.81 | 2,054.56 | 31,760.88 | 4,637.09 | 78,998.81 | 11,100.22 |
| `12d064c247ef` | 2026-08-04 03:24 | 49,106.59 | 45,548.65 | 4,882.83 | 42.52 | C256 peak-active P50 | 1,482.61 | 688.09 | 3,024.47 | 760.03 | 6,325.74 | 962.01 | 13,741.21 | 2,055.09 | 31,789.74 | 4,620.75 | 79,046.59 | 11,058.31 |
| `30bcc7cfde65` | 2026-08-03 16:00 | 49,711.87 | 84,808.34 | 4,680.27 | 44.36 | C256 peak-active P50 | 1,477.13 | 366.58 | 3,013.63 | 438.92 | 6,270.04 | 618.41 | 13,537.86 | 1,373.85 | 31,034.65 | 3,278.30 | 76,212.01 | 8,412.47 |
| `9fbb71b8f700` | 2026-08-03 08:06 | — | — | 4,690.53 | 44.31 | C256 peak-active P50 | — | — | — | — | — | — | — | — | — | — | — | — |

Failed benchmark groups are recorded as -1 tok/s in the table and JSON/CSV reports, while charts plot successful measurements only. The prefill charts compare DP8 and PCP8 throughput and track their recent peaks. The combined history table records each run's throughput and per-length median TTFT; missing measurements are shown as — and failed lengths as failed. The single-request TTFT chart uses concurrency 1, runs requests serially, and plots median latency to the first generated token across the completed samples. The decode chart keeps legacy peak-output and current peak-active P50 statistics in separate series; see [`reports/latest.json`](reports/latest.json) for the newest peaks and [`reports/throughput_history.json`](reports/throughput_history.json) for the full history.
<!-- BENCHMARK_REPORT_END -->

## Real variable-length prefill benchmark

<!-- SPEED_BENCH_REPORT_START -->
Latest PCP8 semantic mixed-length result: **29,026.30 input tok/s**, serial TTFT **739.72 ms median** (`20260806T072451Z`).

The fixed dataset contains **20** requests from NVIDIA SPEED-Bench, ranging from **1,038** to **32,982** input tokens (SHA-256 `865ccc4fdc3e…`). Throughput uses concurrency 8; TTFT uses concurrency 1.

| Prefill mode | Dataset SHA-256 | vllm-torchtpu commit | Test time (UTC) | Status | Input tok/s (C8) | Total tok/s (C8) | Serial TTFT median (ms) | P90 (ms) | P99 (ms) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| **PCP8** | `865ccc4fdc3e` | `884a3154ca91` | 2026-08-06 07:26 | success | 29,026.30 | 29,028.78 | 739.72 | 912.64 | 1,513.70 |
| **DP8** | `865ccc4fdc3e` | `f53d6300e29f` | 2026-08-06 06:25 | success | 18,611.18 | 18,612.76 | 1,348.76 | 5,729.08 | 6,263.70 |

Full machine-readable history is stored in [`reports/speed_bench_history.json`](reports/speed_bench_history.json) and [`reports/speed_bench_history.csv`](reports/speed_bench_history.csv).
<!-- SPEED_BENCH_REPORT_END -->

## Layout

- `third_party/torchtpu-vllm/`: `vllm-project/vllm-torchtpu` Git submodule,
  refreshed from `origin/main` (the local path is retained for compatibility).
- `models/`: offline model metadata and locally provisioned checkpoint weights;
  checkpoint files are excluded from Git.
- `scripts/start_dp_decode_server.sh`: starts the real-weight TP1/DP8/EP8
  C256 decode service with unified pool, auto-derived block size, GMU
  0.932285943, async scheduling, GDN v3, prefix cache disabled, and the same
  compile shapes as the current standalone C256 test；服务由
  `vllm-service-launch` 以 `role=decode` 托管。
- `scripts/start_dp_server.sh`: starts the real-weight DP8/PCP1 vLLM server
  with unified pool and an auto-derived block size；服务由
  `vllm-service-launch` 以 `role=prefill` 托管。
- `scripts/start_pcp_server.sh`: starts the real-weight DP1/PCP8 vLLM server
  with unified pool and an auto-derived block size；服务由
  `vllm-service-launch` 以 `role=prefill` 托管。
- `scripts/bench_all.sh`: benchmarks input length 8192 at concurrency 8–256 for
  the configuration selected by `BENCHMARK_CONFIG`.
- `scripts/bench_prefill_ttft.sh`: benchmarks 16 serial requests at concurrency
  1 for each input length from 8K–252K and writes an independent TTFT summary.
- `scripts/prepare_speed_bench_mix.py`: deterministically builds the checked-in
  20-request semantic mixed-length dataset from a raw SPEED-Bench snapshot.
- `scripts/bench_speed_bench_mix.sh`: runs the DP8 semantic mixed-length
  throughput and serial-TTFT measurements and validates the raw vLLM results.
- `scripts/update_environment.sh`: updates `vllm-torchtpu`, installs its
  compatible `torch_tpu` wheel from Google Artifact Registry with pip, then
  synchronizes the rest of the project `.venv`.
- `scripts/daily_benchmark.sh`: complete locked cron workflow.
- `reports/`: durable peak-throughput history and generated SVG charts.
- `runs/`: timestamped logs, environment snapshots, and benchmark JSON files.

## First preparation

机器需要安装 `git`、`uv`、Google Cloud CLI、Python 3.12，以及
`vllm-service-launch` 和对应的 systemd service。拉取 `vllm-torchtpu`
submodule 需要 GitHub SSH 权限；当前 gcloud 用户必须具有私有 `torch-tpu`
Artifact Registry 的读取权限。首次运行前先完成认证：

```bash
gcloud auth login
gcloud auth list --filter=status:ACTIVE
```

在 benchmark 主机安装 launcher：

```bash
sudo /path/to/tpu-misc/pd_disagg/observability/install.sh
sudo systemd-tmpfiles --create vllm-metrics-targets.conf
sudo systemctl daemon-reload
vllm-service-launch --help
```

完整 daily run 要求 `models/Qwen3.5-397B-A17B-FP8` 中存在真实模型权重。
权重文件由环境在本地提供并被 Git 忽略；`--prepare-only` 只执行
config/tokenizer 元数据检查，三个启动脚本加载服务时会由模型加载器验证
权重是否完整。

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

By default the runner tests the latest commit on `vllm-torchtpu/main`. To test
an exact commit instead, pass its Git commit ID:

```bash
scripts/daily_benchmark.sh --commit 0123456789abcdef0123456789abcdef01234567
```

`--commit` works with full runs, `--prepare-only`, and `--only`. A full
40-character commit ID is recommended; a 7–40 character short ID also works
when Git can resolve it locally or fetch it from the remote. The selected exact
revision is recorded in `run_metadata.json` and the generated report as usual.
Omitting `--commit` preserves the scheduled-job behavior of fetching the latest
`origin/main`.

To run only one benchmark group, use `--only`:

```bash
scripts/daily_benchmark.sh --only dp-decode
scripts/daily_benchmark.sh --only dp-prefill
scripts/daily_benchmark.sh --only pcp-prefill
```

DP8 and PCP8 prefill run both the 8K throughput sweep and the multi-length
single-request TTFT sweep by default. Use `--prefill-mode` to select only one:

```bash
# Run only the 8K concurrency/throughput sweep.
scripts/daily_benchmark.sh --only dp-prefill --prefill-mode throughput
scripts/daily_benchmark.sh --only pcp-prefill --prefill-mode throughput

# Run only the 8K–252K single-request TTFT sweep.
scripts/daily_benchmark.sh --only dp-prefill --prefill-mode ttft
scripts/daily_benchmark.sh --only pcp-prefill --prefill-mode ttft
```

`--prefill-mode all` is the default and preserves the existing behavior. The
mode applies to every selected prefill group, so a full run with
`--prefill-mode throughput` still runs DP decode but limits both DP8 and PCP8
prefill to their throughput sweeps. An explicitly supplied `--prefill-mode` is
rejected with `--only dp-decode` because that selection contains no prefill
benchmark.

Use `--prefill-workload` to choose the request set independently from the
measurement type:

```bash
# Run only the existing fixed 8K/multi-length-random prefill measurements.
scripts/daily_benchmark.sh --only dp-prefill --prefill-workload synthetic

# Run only the 20-request semantic mixed-length workload.
scripts/daily_benchmark.sh --only dp-prefill --prefill-workload speed-bench

# Combine workload and metric selectors.
scripts/daily_benchmark.sh --only dp-prefill \
  --prefill-workload speed-bench --prefill-mode throughput
```

`--prefill-workload all` is the default: DP8 runs both synthetic and semantic
workloads, while PCP8 runs only the synthetic workload until its variable-length
path is fixed. The semantic dataset is committed at
`datasets/speed_bench_mix/requests.jsonl`; its manifest records the source
snapshot revision, source-file hashes, final dataset hash, and exact token
lengths. Its results are kept separate from the fixed 8K history in
`reports/speed_bench_history.json` and `reports/speed_bench_history.csv`.

Use fixture-backed test mode to exercise result extraction and report rendering
without updating the environment, starting a server, sending requests, changing
durable reports, or publishing:

```bash
scripts/daily_benchmark.sh --test-only
```

The isolated preview is written beneath
`.state/test-only-preview/<timestamp>/project/`.
The fixture values are for parser and presentation validation only and must not
be treated as publishable benchmark measurements. The server scripts and both
fixed-prefill benchmark scripts, plus `bench_speed_bench_mix.sh`, also accept
`--test-only` directly. This mode covers
the DP8/PCP8 prefill paths; the C256 decode benchmark remains real-run only.
Set `TTFT_TEST_ONLY_FAILED_LENGTHS` to a space-separated subset of the configured
input lengths to preview partial-failure rendering. For example, this marks only
252K as failed while retaining the other fixture-backed DP8 measurements:

```bash
TTFT_TEST_ONLY_FAILED_LENGTHS=258048 \
  scripts/daily_benchmark.sh --test-only --only dp-prefill
```

Omitting `--only` preserves the full three-group workflow. A selective run
updates the environment in the same way as a full run, but validates and starts
only the service required by the selected benchmark. A prefill measurement
excluded by `--prefill-mode` is recorded as `not-run`, not as a failure. All
selections load the real checkpoint and therefore require complete weights in
the shared model directory.

Every selected benchmark group is reported and published unless
`PUBLISH_REPORTS=0`, including decode-only runs. A failed group records
throughput `-1`, does not prevent later selected groups from running, and still
participates in the final report publication. The runner returns a nonzero exit
status after publication when any selected group failed. TTFT input lengths are
handled independently: successful lengths remain visible, failed lengths are
shown as `failed`, and the TTFT chart plots only successful points. A partial
TTFT sweep does not stop the remaining lengths or the following PCP8 group, but
is counted as a failure in the final process status after reports are published.
With
`--keep-server-running`, the server for a fully successful selective run is kept.

更新或构建前，完整工作流会按稳定的 launcher service ID 停止上一次残留的
daily benchmark 服务。若其他进程占用 `PORT`（默认 18100），工作流不会主动
终止该进程，而是让 launcher 安全地返回启动失败。`--prepare-only` 不会修改
任何运行中的服务。

The runner first starts the real-weight DP8 C256 decode service, runs one
C8/P65536/D32 smoke process followed by three independent
C256/P65536/D1024 processes, and stops it. It then starts the real-weight DP8
and PCP8 services one at a time for their prefill suites. The complete run
therefore contains three benchmark groups: DP8 C256 decode, DP8 prefill, and
PCP8 prefill. Each group is isolated so a startup or benchmark failure is
recorded before the runner advances to the next group. The two prefill services
run their existing 8K concurrency sweep first and their single-request TTFT
sweep second, using 16 serial measured requests per input length. Servers are
stopped after the benchmark by default.
Use `--keep-server-running` only for interactive debugging; when successful,
it keeps the final PCP8 server alive.

The DP8 and PCP8 prefill launchers default to `--max-model-len 262144`.
The longest TTFT input is 258048 tokens (252K), leaving room for the one-token
output within the model's 262144-token context limit.

All three services use `models/Qwen3.5-397B-A17B-FP8` as their model
directory and load the real checkpoint from that directory with vLLM's default
automatic load format.

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
│   ├── single_request_ttft/
│   │   ├── summary.json
│   │   └── vllm_dp8_single_request_ttft_len*.json
│   └── vllm_dp8_tp1_len8192_c*.json
└── pcp8/
    ├── summary.json
    ├── single_request_ttft/
    │   ├── summary.json
    │   └── vllm_pcp8_single_request_ttft_len*.json
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
`reports/decode_throughput_history.svg`. The latest length-vs-TTFT comparison is
`reports/prefill_ttft.svg`, with long-form history in
`reports/prefill_ttft_history.csv`; every successful report update replaces the
generated files atomically. Decode history renders the legacy
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
