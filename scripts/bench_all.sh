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

RUN_DIR="${1:-}"
if (( $# > 0 )); then
  shift
fi
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$PROJECT_ROOT/runs/manual-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)
RESULT_DIR="$RUN_DIR/results"
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

  result_filename="vllm_dp8_tp1_len${INPUT_LEN}_c${concurrency}.json"
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
    --num-prompts "$concurrency" \
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
    --label "vllm_dp8_tp1_len${INPUT_LEN}_c${concurrency}" \
    "$@"
done

"$VENV_DIR/bin/python" - "$RESULT_DIR" <<'PY'
import glob
import json
import os
import sys

result_dir = sys.argv[1]
records = []
for path in sorted(glob.glob(os.path.join(result_dir, "vllm_dp8_tp1_len*_c*.json"))):
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
summary = {"best": best, "results": records}
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
