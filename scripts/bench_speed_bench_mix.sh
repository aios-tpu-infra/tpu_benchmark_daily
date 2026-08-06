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
if [[ -n "${SPEED_BENCH_CONCURRENCIES:-}" ]]; then
  CONCURRENCY_SPEC=$SPEED_BENCH_CONCURRENCIES
elif [[ -n "${SPEED_BENCH_THROUGHPUT_CONCURRENCY:-}" ]]; then
  # Compatibility for callers that selected one concurrency before the sweep
  # was introduced.
  CONCURRENCY_SPEC=$SPEED_BENCH_THROUGHPUT_CONCURRENCY
else
  CONCURRENCY_SPEC="8 64"
fi

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

case "$BENCHMARK_CONFIG" in
  dp8) CONFIG_LABEL=DP8 ;;
  pcp8) CONFIG_LABEL=PCP8 ;;
  *)
    echo "ERROR: BENCHMARK_CONFIG must be dp8 or pcp8." >&2
    exit 2
    ;;
esac
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
CONCURRENCY_SPEC=${CONCURRENCY_SPEC//,/ }
read -r -a THROUGHPUT_CONCURRENCIES <<< "$CONCURRENCY_SPEC"
if (( ${#THROUGHPUT_CONCURRENCIES[@]} == 0 )); then
  echo "ERROR: SPEED_BENCH_CONCURRENCIES must not be empty." >&2
  exit 2
fi
declare -A SEEN_CONCURRENCIES=()
for concurrency in "${THROUGHPUT_CONCURRENCIES[@]}"; do
  if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid SPEED_BENCH_CONCURRENCIES value '$concurrency'." >&2
    exit 2
  fi
  if [[ -n "${SEEN_CONCURRENCIES[$concurrency]:-}" ]]; then
    echo "ERROR: duplicate SPEED_BENCH_CONCURRENCIES value '$concurrency'." >&2
    exit 2
  fi
  SEEN_CONCURRENCIES[$concurrency]=1
done

if (( TEST_ONLY )); then
  MANIFEST_PATH="$FIXTURE_ROOT/manifest.json"
else
  MANIFEST_PATH="$DATASET_DIR/manifest.json"
fi
RESULT_DIR="$RUN_DIR/results/$BENCHMARK_CONFIG/speed_bench_mix"
SUMMARY_PATH="$RESULT_DIR/summary.json"
mkdir -p "$RESULT_DIR"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "ERROR: prepared SPEED-Bench manifest is missing: $MANIFEST_PATH." >&2
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

if ! NUM_PROMPTS=$(
  "$AGGREGATE_PYTHON" -c \
    'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8"))["dataset"]; count=int(data["requests"]); assert count == len(data["input_tokens"]); print(count)' \
    "$MANIFEST_PATH"
); then
  echo "ERROR: failed to read a consistent request count from $MANIFEST_PATH." >&2
  exit 1
fi
if [[ ! "$NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: invalid SPEED-Bench request count: '$NUM_PROMPTS'." >&2
  exit 1
fi

mapfile -t DATASET_METADATA < <(
  "$AGGREGATE_PYTHON" -c \
    'import json,pathlib,sys; manifest=pathlib.Path(sys.argv[1]); data=json.load(open(manifest, encoding="utf-8"))["dataset"]; path=pathlib.Path(data["path"]); print(path if path.is_absolute() else manifest.parent / path); print(data["sha256"]); print(data.get("artifact_sha256", data["sha256"]))' \
    "$MANIFEST_PATH"
)
if (( ${#DATASET_METADATA[@]} != 3 )); then
  echo "ERROR: failed to read dataset metadata from $MANIFEST_PATH." >&2
  exit 1
fi
DATASET_ARTIFACT=${DATASET_METADATA[0]}
DATASET_SHA256=${DATASET_METADATA[1]}
ARTIFACT_SHA256=${DATASET_METADATA[2]}
if [[ ! -f "$DATASET_ARTIFACT" ]]; then
  echo "ERROR: prepared SPEED-Bench dataset is missing: $DATASET_ARTIFACT." >&2
  exit 1
fi
ACTUAL_ARTIFACT_SHA256=$(sha256sum "$DATASET_ARTIFACT" | awk '{print $1}')
if [[ "$ACTUAL_ARTIFACT_SHA256" != "$ARTIFACT_SHA256" ]]; then
  echo "ERROR: SPEED-Bench artifact SHA-256 mismatch." >&2
  exit 1
fi

if [[ "$DATASET_ARTIFACT" == *.gz ]]; then
  DATASET_PATH="$RUN_DIR/speed_bench_mix_requests.jsonl"
  TEMP_DATASET_PATH="$RUN_DIR/.speed_bench_mix_requests.jsonl.tmp"
  if ! gzip -dc "$DATASET_ARTIFACT" >"$TEMP_DATASET_PATH"; then
    rm -f -- "$TEMP_DATASET_PATH"
    echo "ERROR: failed to decompress $DATASET_ARTIFACT." >&2
    exit 1
  fi
  ACTUAL_DATASET_SHA256=$(sha256sum "$TEMP_DATASET_PATH" | awk '{print $1}')
  if [[ "$ACTUAL_DATASET_SHA256" != "$DATASET_SHA256" ]]; then
    rm -f -- "$TEMP_DATASET_PATH"
    echo "ERROR: SPEED-Bench content SHA-256 mismatch after decompression." >&2
    exit 1
  fi
  mv -- "$TEMP_DATASET_PATH" "$DATASET_PATH"
else
  DATASET_PATH="$DATASET_ARTIFACT"
fi

declare -A throughput_statuses=()
for concurrency in "${THROUGHPUT_CONCURRENCIES[@]}"; do
  throughput_statuses[$concurrency]=failed
done

if (( TEST_ONLY )); then
  for concurrency in "${THROUGHPUT_CONCURRENCIES[@]}"; do
    throughput_result="$RESULT_DIR/throughput_c${concurrency}.json"
    "$AGGREGATE_PYTHON" - \
        "$FIXTURE_ROOT/throughput.json" \
        "$throughput_result" \
        "$concurrency" <<'PY'
import json
import pathlib
import sys

source, destination, concurrency = sys.argv[1:]
data = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
data["max_concurrency"] = int(concurrency)
pathlib.Path(destination).write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
    throughput_statuses[$concurrency]=success
    echo "TEST_ONLY: replayed SPEED-Bench fixture at concurrency $concurrency."
  done
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
    echo "ERROR: $CONFIG_LABEL service is unavailable; SPEED-Bench components will be recorded as failed." >&2
  else
    echo "Warming up $CONFIG_LABEL with the first semantic request..."
    if ! "$VENV_DIR/bin/vllm" bench serve \
        "${common_args[@]}" \
        --num-prompts 1 \
        --disable-shuffle \
        --max-concurrency 1 \
        --header X-data-parallel-rank=0 \
        --result-dir "$RESULT_DIR/warmup" \
        --label "speed_bench_mix_${BENCHMARK_CONFIG}_warmup"; then
      echo "WARNING: SPEED-Bench warm-up failed; measured components will still be attempted." >&2
    fi

    for concurrency in "${THROUGHPUT_CONCURRENCIES[@]}"; do
      throughput_result="$RESULT_DIR/throughput_c${concurrency}.json"
      echo "Running $CONFIG_LABEL SPEED-Bench mixed-length workload ($NUM_PROMPTS requests, concurrency $concurrency)..."
      if "$VENV_DIR/bin/vllm" bench serve \
          "${common_args[@]}" \
          --num-prompts "$NUM_PROMPTS" \
          --max-concurrency "$concurrency" \
          --save-result \
          --save-detailed \
          --result-dir "$RESULT_DIR" \
          --result-filename "$(basename -- "$throughput_result")" \
          --label "speed_bench_mix_${BENCHMARK_CONFIG}_c${concurrency}"; then
        throughput_statuses[$concurrency]=success
      else
        echo "ERROR: SPEED-Bench concurrency $concurrency failed." >&2
      fi
    done
  fi
fi

aggregate_args=(
  --manifest "$MANIFEST_PATH"
  --mode "$SPEED_BENCH_MODE"
  --benchmark-config "$BENCHMARK_CONFIG"
  --output-length 1
  --output "$SUMMARY_PATH"
)
for concurrency in "${THROUGHPUT_CONCURRENCIES[@]}"; do
  throughput_result="$RESULT_DIR/throughput_c${concurrency}.json"
  aggregate_args+=(
    --concurrency-status "$concurrency=${throughput_statuses[$concurrency]}"
  )
  if [[ -f "$throughput_result" ]]; then
    aggregate_args+=(--concurrency-result "$concurrency=$throughput_result")
  fi
done
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
