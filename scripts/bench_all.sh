#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-397B-A17B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"
HOST="${BENCH_HOST:-127.0.0.1}"
PORT="${PORT:-18100}"
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-dp8}"
PUBLISH_REPORTS="${PUBLISH_REPORTS:-1}"

RUN_DIR="${1:-}"
if (( $# > 0 )); then
  shift
fi
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$PROJECT_ROOT/runs/manual-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)
if [[ ! "$BENCHMARK_CONFIG" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
  echo "ERROR: invalid BENCHMARK_CONFIG '$BENCHMARK_CONFIG'." >&2
  exit 2
fi

RESULT_PREFIX="vllm_${BENCHMARK_CONFIG}_tp1_len${INPUT_LEN}"
RESULT_DIR="$RUN_DIR/results/$BENCHMARK_CONFIG"
mkdir -p "$RESULT_DIR"

if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
  echo "ERROR: vLLM CLI is missing: $VENV_DIR/bin/vllm" >&2
  echo "Run scripts/update_environment.sh first." >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/tokenizer.json" ]]; then
  echo "ERROR: local tokenizer metadata is missing: $MODEL_DIR" >&2
  exit 1
fi
if [[ "$PUBLISH_REPORTS" != 0 && "$PUBLISH_REPORTS" != 1 ]]; then
  echo "ERROR: PUBLISH_REPORTS must be 0 or 1." >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

read -r -a concurrencies <<< "${CONCURRENCIES:-1 2 4 8 16 32 64}"
if (( ${#concurrencies[@]} == 0 )); then
  echo "ERROR: CONCURRENCIES must contain at least one value." >&2
  exit 2
fi

for concurrency in "${concurrencies[@]}"; do
  if [[ ! "$concurrency" =~ ^[0-9]+$ ]] || (( concurrency == 0 )); then
    echo "ERROR: invalid concurrency '$concurrency'." >&2
    exit 2
  fi

  echo "====================================================================="
  echo "Running benchmark for concurrency: $concurrency"
  echo "====================================================================="

  result_filename="${RESULT_PREFIX}_c${concurrency}.json"
  "$VENV_DIR/bin/vllm" bench serve \
    --backend openai \
    --host "$HOST" \
    --port "$PORT" \
    --endpoint /v1/completions \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0 \
    --num-prompts 128 \
    --request-rate inf \
    --max-concurrency "$concurrency" \
    --ignore-eos \
    --temperature 0 \
    --seed 42 \
    --percentile-metrics ttft,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --result-dir "$RESULT_DIR" \
    --result-filename "$result_filename" \
    --label "${RESULT_PREFIX}_c${concurrency}" \
    "$@"
done

"$VENV_DIR/bin/python" - \
  "$RESULT_DIR" \
  "$INPUT_LEN" \
  "$OUTPUT_LEN" \
  "$SERVED_MODEL_NAME" \
  "$BENCHMARK_CONFIG" \
  "$RESULT_PREFIX" <<'PY'
import glob
import json
import os
import sys

result_dir = sys.argv[1]
input_length = int(sys.argv[2])
output_length = int(sys.argv[3])
model = sys.argv[4]
benchmark_config = sys.argv[5]
result_prefix = sys.argv[6]
records = []
for path in sorted(glob.glob(os.path.join(result_dir, f"{result_prefix}_c*.json"))):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    records.append({
        "file": os.path.basename(path),
        "concurrency": int(data["max_concurrency"]),
        "completed": int(data["completed"]),
        "failed": int(data["failed"]),
        "request_throughput": float(data["request_throughput"]),
        "total_token_throughput": float(data["total_token_throughput"]),
        "mean_ttft_ms": float(data["mean_ttft_ms"]),
        "p99_ttft_ms": float(data["p99_ttft_ms"]),
    })

if not records:
    raise SystemExit("No benchmark result JSON files were produced.")

records.sort(key=lambda item: item["concurrency"])
best = max(records, key=lambda item: item["total_token_throughput"])
summary = {
    "benchmark": {
        "input_length": input_length,
        "output_length": output_length,
        "model": model,
        "benchmark_config": benchmark_config,
    },
    "best": best,
    "results": records,
}
summary_path = os.path.join(result_dir, "summary.json")
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")

print(
    "Highest total token throughput: "
    f"{best['total_token_throughput']:.2f} tok/s "
    f"(concurrency={best['concurrency']})"
)
failed = sum(item["failed"] for item in records)
if failed:
    raise SystemExit(f"Benchmark completed with {failed} failed requests.")
PY

report_args=(
  --project-root "$PROJECT_ROOT"
  --run-dir "$RUN_DIR"
  --summary "$RESULT_DIR/summary.json"
  --benchmark-config "$BENCHMARK_CONFIG"
  --input-length "$INPUT_LEN"
  --output-length "$OUTPUT_LEN"
  --model "$SERVED_MODEL_NAME"
)
decode_summary="$RUN_DIR/results/dp8_decode_c256/summary.json"
if [[ "$BENCHMARK_CONFIG" == "dp8" && -f "$decode_summary" ]]; then
  report_args+=(--decode-summary "$decode_summary")
fi

"$VENV_DIR/bin/python" "$SCRIPT_DIR/update_report.py" \
  "${report_args[@]}"

if (( PUBLISH_REPORTS )); then
  "$SCRIPT_DIR/publish_report.sh"
else
  echo "Skipping Git report publication because PUBLISH_REPORTS=0."
fi
