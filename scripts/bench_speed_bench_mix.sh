#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-397B-A17B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"
HOST="${BENCH_HOST:-127.0.0.1}"
PORT="${PORT:-18100}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-dp8}"
SPEED_BENCH_MODE="${SPEED_BENCH_MODE:-all}"
DATASET_DIR="${SPEED_BENCH_DATASET_DIR:-$PROJECT_ROOT/datasets/speed_bench_mix}"
TEST_ONLY="${TEST_ONLY:-0}"
FIXTURE_ROOT="${FIXTURE_ROOT:-$PROJECT_ROOT/tests/fixtures/speed_bench_mix}"

RUN_DIR=
while (( $# > 0 )); do
  case "$1" in
    --test-only)
      TEST_ONLY=1
      ;;
    *)
      if [[ -z "$RUN_DIR" && "$1" != -* ]]; then
        RUN_DIR=$1
      else
        echo "ERROR: unknown argument '$1'." >&2
        exit 2
      fi
      ;;
  esac
  shift
done
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$PROJECT_ROOT/runs/manual-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)

if [[ "$BENCHMARK_CONFIG" != dp8 ]]; then
  echo "ERROR: the SPEED-Bench mixed workload currently supports DP8 only." >&2
  exit 2
fi
case "$SPEED_BENCH_MODE" in
  all|throughput|ttft) ;;
  *)
    echo "ERROR: SPEED_BENCH_MODE must be all, throughput, or ttft." >&2
    exit 2
    ;;
esac
if [[ "$TEST_ONLY" != 0 && "$TEST_ONLY" != 1 ]]; then
  echo "ERROR: TEST_ONLY must be 0 or 1." >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT == 0 )); then
  echo "ERROR: PORT must be a positive integer, got '$PORT'." >&2
  exit 2
fi

MANIFEST_PATH="$DATASET_DIR/manifest.json"
DATASET_PATH="$DATASET_DIR/requests.jsonl"
RESULT_DIR="$RUN_DIR/results/dp8/speed_bench_mix"
THROUGHPUT_RESULT="$RESULT_DIR/throughput.json"
TTFT_RESULT="$RESULT_DIR/ttft.json"
SUMMARY_PATH="$RESULT_DIR/summary.json"
mkdir -p "$RESULT_DIR"

if [[ ! -f "$MANIFEST_PATH" || ! -f "$DATASET_PATH" ]]; then
  echo "ERROR: prepared SPEED-Bench dataset is missing beneath $DATASET_DIR." >&2
  exit 1
fi

if (( TEST_ONLY )); then
  AGGREGATE_PYTHON="${TEST_ONLY_PYTHON:-python3.12}"
else
  AGGREGATE_PYTHON="$VENV_DIR/bin/python"
fi
if ! command -v "$AGGREGATE_PYTHON" >/dev/null 2>&1; then
  echo "ERROR: Python is missing: $AGGREGATE_PYTHON" >&2
  exit 1
fi

throughput_status=failed
ttft_status=failed

if (( TEST_ONLY )); then
  if [[ "$SPEED_BENCH_MODE" == all || "$SPEED_BENCH_MODE" == throughput ]]; then
    cp "$FIXTURE_ROOT/throughput.json" "$THROUGHPUT_RESULT"
    throughput_status=success
    echo "TEST_ONLY: replayed SPEED-Bench throughput fixture."
  fi
  if [[ "$SPEED_BENCH_MODE" == all || "$SPEED_BENCH_MODE" == ttft ]]; then
    cp "$FIXTURE_ROOT/ttft.json" "$TTFT_RESULT"
    ttft_status=success
    echo "TEST_ONLY: replayed SPEED-Bench serial TTFT fixture."
  fi
else
  if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
    echo "ERROR: vLLM CLI is missing: $VENV_DIR/bin/vllm" >&2
    exit 1
  fi
  if [[ ! -f "$MODEL_DIR/tokenizer.json" ]]; then
    echo "ERROR: local tokenizer metadata is missing: $MODEL_DIR" >&2
    exit 1
  fi

  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1

  common_args=(
    --backend openai
    --host "$HOST"
    --port "$PORT"
    --endpoint /v1/completions
    --model "$SERVED_MODEL_NAME"
    --tokenizer "$MODEL_DIR"
    --dataset-name custom
    --dataset-path "$DATASET_PATH"
    --custom-output-len 1
    --no-oversample
    --request-rate inf
    --ignore-eos
    --temperature 0
    --seed 42
    --percentile-metrics ttft,e2el
    --metric-percentiles 50,90,99
  )

  if ! curl -fsS --max-time 5 "http://$HOST:$PORT/health" >/dev/null; then
    echo "ERROR: DP8 service is unavailable; SPEED-Bench components will be recorded as failed." >&2
  else
    echo "Warming up DP8 with the first 1K semantic request..."
    if ! "$VENV_DIR/bin/vllm" bench serve \
        "${common_args[@]}" \
        --num-prompts 1 \
        --disable-shuffle \
        --max-concurrency 1 \
        --header X-data-parallel-rank=0 \
        --result-dir "$RESULT_DIR/warmup" \
        --label speed_bench_mix_warmup; then
      echo "WARNING: SPEED-Bench warm-up failed; measured components will still be attempted." >&2
    fi

    if [[ "$SPEED_BENCH_MODE" == all || "$SPEED_BENCH_MODE" == throughput ]]; then
      echo "Running DP8 SPEED-Bench mixed-length throughput (20 requests, concurrency 8)..."
      if "$VENV_DIR/bin/vllm" bench serve \
          "${common_args[@]}" \
          --num-prompts 20 \
          --max-concurrency 8 \
          --save-result \
          --save-detailed \
          --result-dir "$RESULT_DIR" \
          --result-filename throughput.json \
          --label speed_bench_mix_dp8_throughput; then
        throughput_status=success
      else
        echo "ERROR: SPEED-Bench mixed-length throughput failed." >&2
      fi
    fi

    if [[ "$SPEED_BENCH_MODE" == all || "$SPEED_BENCH_MODE" == ttft ]]; then
      echo "Running DP8 SPEED-Bench mixed-length serial TTFT (20 requests)..."
      if "$VENV_DIR/bin/vllm" bench serve \
          "${common_args[@]}" \
          --num-prompts 20 \
          --max-concurrency 1 \
          --header X-data-parallel-rank=0 \
          --save-result \
          --save-detailed \
          --result-dir "$RESULT_DIR" \
          --result-filename ttft.json \
          --label speed_bench_mix_dp8_serial_ttft; then
        ttft_status=success
      else
        echo "ERROR: SPEED-Bench mixed-length serial TTFT failed." >&2
      fi
    fi
  fi
fi

aggregate_args=(
  --manifest "$MANIFEST_PATH"
  --mode "$SPEED_BENCH_MODE"
  --benchmark-config dp8
  --output-length 1
  --throughput-status "$throughput_status"
  --ttft-status "$ttft_status"
  --output "$SUMMARY_PATH"
)
if [[ -f "$THROUGHPUT_RESULT" ]]; then
  aggregate_args+=(--throughput-result "$THROUGHPUT_RESULT")
fi
if [[ -f "$TTFT_RESULT" ]]; then
  aggregate_args+=(--ttft-result "$TTFT_RESULT")
fi
"$AGGREGATE_PYTHON" "$SCRIPT_DIR/aggregate_speed_bench_mix.py" \
  "${aggregate_args[@]}"

summary_status=$(
  "$AGGREGATE_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
    "$SUMMARY_PATH"
)
if [[ "$summary_status" != success ]]; then
  echo "ERROR: SPEED-Bench mixed workload completed with status $summary_status." >&2
  exit 1
fi
echo "SPEED-Bench mixed workload completed successfully: $SUMMARY_PATH"
