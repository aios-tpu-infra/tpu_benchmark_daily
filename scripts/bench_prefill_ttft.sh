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
TTFT_INPUT_LENGTHS="${TTFT_INPUT_LENGTHS:-8192 16384 32768 65536 131072 258048}"
TTFT_OUTPUT_LEN="${TTFT_OUTPUT_LEN:-1}"
TTFT_NUM_PROMPTS="${TTFT_NUM_PROMPTS:-16}"
TTFT_REDUCED_INPUT_LENGTHS="${TTFT_REDUCED_INPUT_LENGTHS:-65536 131072 258048}"
TTFT_REDUCED_NUM_PROMPTS="${TTFT_REDUCED_NUM_PROMPTS:-4}"
TEST_ONLY="${TEST_ONLY:-0}"
FIXTURE_ROOT="${FIXTURE_ROOT:-$PROJECT_ROOT/tests/fixtures/prefill_ttft}"
TTFT_TEST_ONLY_FAILED_LENGTHS="${TTFT_TEST_ONLY_FAILED_LENGTHS:-}"

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

if [[ ! "$BENCHMARK_CONFIG" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
  echo "ERROR: invalid BENCHMARK_CONFIG '$BENCHMARK_CONFIG'." >&2
  exit 2
fi
if [[ "$TEST_ONLY" != 0 && "$TEST_ONLY" != 1 ]]; then
  echo "ERROR: TEST_ONLY must be 0 or 1." >&2
  exit 2
fi
for value_name in PORT TTFT_OUTPUT_LEN TTFT_NUM_PROMPTS TTFT_REDUCED_NUM_PROMPTS; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value == 0 )); then
    echo "ERROR: $value_name must be a positive integer, got '$value'." >&2
    exit 2
  fi
done

read -r -a input_lengths <<< "$TTFT_INPUT_LENGTHS"
if (( ${#input_lengths[@]} == 0 )); then
  echo "ERROR: TTFT_INPUT_LENGTHS must not be empty." >&2
  exit 2
fi
for input_length in "${input_lengths[@]}"; do
  if [[ ! "$input_length" =~ ^[0-9]+$ ]] || (( input_length == 0 )); then
    echo "ERROR: invalid TTFT input length '$input_length'." >&2
    exit 2
  fi
done
read -r -a reduced_input_lengths <<< "$TTFT_REDUCED_INPUT_LENGTHS"
for input_length in "${reduced_input_lengths[@]}"; do
  if [[ ! "$input_length" =~ ^[0-9]+$ ]] || (( input_length == 0 )); then
    echo "ERROR: invalid reduced-sample TTFT input length '$input_length'." >&2
    exit 2
  fi
done
read -r -a test_only_failed_lengths <<< "$TTFT_TEST_ONLY_FAILED_LENGTHS"
for input_length in "${test_only_failed_lengths[@]}"; do
  if [[ ! "$input_length" =~ ^[0-9]+$ ]] ||
      [[ " ${input_lengths[*]} " != *" $input_length "* ]]; then
    echo "ERROR: invalid test-only failed TTFT length '$input_length'." >&2
    exit 2
  fi
done

RESULT_DIR="$RUN_DIR/results/$BENCHMARK_CONFIG/single_request_ttft"
mkdir -p "$RESULT_DIR"
result_prefix="vllm_${BENCHMARK_CONFIG}_single_request_ttft"

if (( ! TEST_ONLY )); then
  if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
    echo "ERROR: vLLM CLI is missing: $VENV_DIR/bin/vllm" >&2
    exit 1
  fi
  if [[ ! -f "$MODEL_DIR/tokenizer.json" ]]; then
    echo "ERROR: local tokenizer metadata is missing: $MODEL_DIR" >&2
    exit 1
  fi
  AGGREGATE_PYTHON="$VENV_DIR/bin/python"
else
  AGGREGATE_PYTHON="${TEST_ONLY_PYTHON:-python3.12}"
  if ! command -v "$AGGREGATE_PYTHON" >/dev/null 2>&1; then
    echo "ERROR: TEST_ONLY Python is missing: $AGGREGATE_PYTHON" >&2
    exit 1
  fi
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

samples_by_length_args=()
for input_length in "${input_lengths[@]}"; do
  num_prompts=$TTFT_NUM_PROMPTS
  if [[ " ${reduced_input_lengths[*]} " == *" $input_length "* ]]; then
    num_prompts=$TTFT_REDUCED_NUM_PROMPTS
  fi
  samples_by_length_args+=(--samples-by-length "$input_length=$num_prompts")
  result_filename="${result_prefix}_len${input_length}.json"
  result_path="$RESULT_DIR/$result_filename"

  if (( TEST_ONLY )); then
    fixture_path="$FIXTURE_ROOT/$BENCHMARK_CONFIG/$result_filename"
    if [[ ! -f "$fixture_path" ]]; then
      echo "ERROR: TTFT fixture is missing: $fixture_path" >&2
      exit 1
    fi
    replay_args=()
    if [[ " ${test_only_failed_lengths[*]} " == *" $input_length "* ]]; then
      replay_args+=(--failed)
    fi
    "$AGGREGATE_PYTHON" "$SCRIPT_DIR/replay_prefill_ttft_fixture.py" \
      --source "$fixture_path" \
      --destination "$result_path" \
      --samples "$num_prompts" \
      "${replay_args[@]}"
    echo "TEST_ONLY: replayed $fixture_path as $num_prompts serial requests"
    continue
  fi

  header_args=()
  if [[ "$BENCHMARK_CONFIG" == "dp8" ]]; then
    header_args=(--header X-data-parallel-rank=0)
  fi
  if ! curl -fsS --max-time 5 \
      "http://$HOST:$PORT/health" >/dev/null; then
    echo "ERROR: service is unavailable; recording length $input_length as failed." >&2
    continue
  fi
  echo "Running $BENCHMARK_CONFIG single-request TTFT at input length $input_length..."
  if ! "$VENV_DIR/bin/vllm" bench serve \
    --backend openai \
    --host "$HOST" \
    --port "$PORT" \
    --endpoint /v1/completions \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len "$input_length" \
    --random-output-len "$TTFT_OUTPUT_LEN" \
    --random-range-ratio 0 \
    --num-prompts "$num_prompts" \
    --request-rate inf \
    --max-concurrency 1 \
    --ignore-eos \
    --temperature 0 \
    --seed 42 \
    --percentile-metrics ttft,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --result-dir "$RESULT_DIR" \
    --result-filename "$result_filename" \
    --label "${result_prefix}_len${input_length}" \
    "${header_args[@]}"; then
    echo "ERROR: TTFT request length $input_length failed; continuing." >&2
  fi
done

"$AGGREGATE_PYTHON" "$SCRIPT_DIR/aggregate_prefill_ttft.py" \
  --result-dir "$RESULT_DIR" \
  --benchmark-config "$BENCHMARK_CONFIG" \
  --input-lengths "${input_lengths[@]}" \
  --output-length "$TTFT_OUTPUT_LEN" \
  --samples-per-length "$TTFT_NUM_PROMPTS" \
  "${samples_by_length_args[@]}"

echo "Single-request TTFT benchmark completed: $RESULT_DIR/summary.json"
