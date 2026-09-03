#!/usr/bin/env bash

set -Eeuo pipefail

export TPU_SKIP_MDS_QUERY="true"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
STATE_DIR="${STATE_DIR:-$PROJECT_ROOT/.state}"
VLLM_SERVICE_LAUNCH="${VLLM_SERVICE_LAUNCH:-$PROJECT_ROOT/vendor/vllm-service-launch/bin/vllm-service-launch}"
VLLM_SERVICE_STATE_ROOT="${VLLM_SERVICE_STATE_ROOT:-$PROJECT_ROOT/.state/vllm-service-launch}"
VLLM_SERVICE_TARGET_ROOT="${VLLM_SERVICE_TARGET_ROOT:-/run/vllm-metrics-targets/targets}"
export VLLM_SERVICE_LAUNCH VLLM_SERVICE_STATE_ROOT VLLM_SERVICE_TARGET_ROOT
VLLM_SERVICE_LAYOUT_ARGS=(
  --state-root "$VLLM_SERVICE_STATE_ROOT"
  --target-root "$VLLM_SERVICE_TARGET_ROOT"
)
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TORCHTPU_DIR="${TORCHTPU_DIR:-$PROJECT_ROOT/third_party/torchtpu-vllm}"
REPOSITORY_MODEL_DIR="$PROJECT_ROOT/models/Qwen3.5-397B-A17B-FP8"
SHARED_MODEL_DIR="$PROJECT_ROOT/../models/Qwen3.5-397B-A17B-FP8"
if [[ -z "${MODEL_DIR:-}" ]]; then
  if [[ -f "$REPOSITORY_MODEL_DIR/model.safetensors.index.json" ]]; then
    MODEL_DIR=$REPOSITORY_MODEL_DIR
  elif [[ -f "$SHARED_MODEL_DIR/model.safetensors.index.json" ]]; then
    MODEL_DIR=$SHARED_MODEL_DIR
  else
    MODEL_DIR=$REPOSITORY_MODEL_DIR
  fi
fi
PORT="${PORT:-18100}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-3600}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-120}"
KEEP_SERVER_RUNNING="${KEEP_SERVER_RUNNING:-0}"
PUBLISH_REPORTS="${PUBLISH_REPORTS:-1}"
MACHINE_IP="${MACHINE_IP:-}"
DP_DECODE_SERVICE_ID=tpu-daily-dp4-tp2-decode-c256
DP_PREFILL_SERVICE_ID=tpu-daily-dp8-prefill
PCP_PREFILL_SERVICE_ID=tpu-daily-pcp8-prefill
PREPARE_ONLY=0
TEST_ONLY=0
BENCHMARK_SELECTION=all
PREFILL_MODE=all
PREFILL_MODE_SPECIFIED=0
PREFILL_WORKLOAD=all
PREFILL_WORKLOAD_SPECIFIED=0
TORCHTPU_COMMIT=
TORCHTPU_COMMIT_SPECIFIED=0
RUN_DP_DECODE=0
RUN_DP_PREFILL=0
RUN_PCP_PREFILL=0
RUN_PREFILL_THROUGHPUT=0
RUN_PREFILL_TTFT=0
RUN_SYNTHETIC_PREFILL=0
RUN_DP_SPEED_BENCH_PREFILL=0
RUN_PCP_SPEED_BENCH_PREFILL=0
BENCHMARK_CONFIGS_JSON=
DP_DECODE_STATUS=not-run
DP_PREFILL_STATUS=not-run
PCP_PREFILL_STATUS=not-run
DP_PREFILL_TTFT_STATUS=not-run
PCP_PREFILL_TTFT_STATUS=not-run
DP_SPEED_BENCH_STATUS=not-run
PCP_SPEED_BENCH_STATUS=not-run
PREFILL_TTFT_LAST_STATUS=
SPEED_BENCH_LAST_STATUS=

mkdir -p "$STATE_DIR" "$PROJECT_ROOT/runs"
if [[ "${DAILY_BENCHMARK_LOCKED:-0}" != 1 ]]; then
  set +e
  DAILY_BENCHMARK_LOCKED=1 flock \
    --exclusive \
    --nonblock \
    --close \
    --conflict-exit-code 75 \
    "$STATE_DIR/daily_benchmark.lock" \
    "$SCRIPT_DIR/daily_benchmark.sh" "$@"
  status=$?
  set -e
  if (( status == 75 )); then
    echo "ERROR: another daily benchmark is already running." >&2
  fi
  exit "$status"
fi

usage() {
  cat <<'EOF'
Usage: scripts/daily_benchmark.sh [--prepare-only] [--test-only]
                                  [--keep-server-running]
                                  [--commit COMMIT]
                                  [--prefill-mode MODE]
                                  [--prefill-workload WORKLOAD]
                                  [--only BENCHMARK]

  --prepare-only         Prepare source/environment without touching a server.
  --test-only            Replay fixed fixtures into an isolated report preview;
                         do not update the environment, start a server, send
                         benchmark requests, publish, or modify durable reports.
  --keep-server-running  Keep a successfully benchmarked server alive.
  --commit COMMIT        Test this exact vllm-torchtpu Git commit instead of the
                         latest origin/main commit. --torchtpu-commit is an alias.
  --only BENCHMARK       Run only one benchmark group. BENCHMARK must be one of:
                         dp-decode, dp-prefill, or pcp-prefill.
  --prefill-mode MODE    Select prefill measurements: all, throughput, or ttft.
                         Defaults to all and applies to every selected prefill
                         group. It is invalid with --only dp-decode.
  --prefill-workload WORKLOAD
                         Select prefill request sets: all, synthetic, or
                         speed-bench. The default all runs the fixed 8K
                         synthetic workload plus the semantic mixed-length
                         SPEED-Bench workload on both DP8 and PCP8.

The default full workflow stops an existing vLLM service on PORT, updates
to the requested vllm-torchtpu commit (or latest main by default), installs its
compatible torch_tpu version with pip, updates
.venv, then runs real-weight C256 DP4/TP2 decode, DP8 prefill, and PCP8 prefill
services. Benchmark groups are independent: a failed group is recorded before
the runner advances. Reports are generated and published after all selected
groups finish. DP8/PCP8 prefill selections run both the throughput and 8K–252K
single-request TTFT sweeps by default. Use --prefill-mode to run only one of
them and --prefill-workload to choose fixed or semantic requests. Omit --only
to run all three benchmark groups.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --prepare-only)
      PREPARE_ONLY=1
      ;;
    --test-only)
      TEST_ONLY=1
      ;;
    --keep-server-running)
      KEEP_SERVER_RUNNING=1
      ;;
    --commit|--torchtpu-commit)
      if (( $# < 2 )); then
        echo "ERROR: $1 requires a Git commit." >&2
        usage >&2
        exit 2
      fi
      TORCHTPU_COMMIT=$2
      TORCHTPU_COMMIT_SPECIFIED=1
      shift
      ;;
    --commit=*|--torchtpu-commit=*)
      TORCHTPU_COMMIT=${1#*=}
      TORCHTPU_COMMIT_SPECIFIED=1
      ;;
    --only)
      if (( $# < 2 )); then
        echo "ERROR: --only requires a benchmark name." >&2
        usage >&2
        exit 2
      fi
      BENCHMARK_SELECTION=$2
      shift
      ;;
    --only=*)
      BENCHMARK_SELECTION=${1#*=}
      ;;
    --prefill-mode)
      if (( $# < 2 )); then
        echo "ERROR: --prefill-mode requires a mode." >&2
        usage >&2
        exit 2
      fi
      PREFILL_MODE=$2
      PREFILL_MODE_SPECIFIED=1
      shift
      ;;
    --prefill-mode=*)
      PREFILL_MODE=${1#*=}
      PREFILL_MODE_SPECIFIED=1
      ;;
    --prefill-workload)
      if (( $# < 2 )); then
        echo "ERROR: --prefill-workload requires a workload." >&2
        usage >&2
        exit 2
      fi
      PREFILL_WORKLOAD=$2
      PREFILL_WORKLOAD_SPECIFIED=1
      shift
      ;;
    --prefill-workload=*)
      PREFILL_WORKLOAD=${1#*=}
      PREFILL_WORKLOAD_SPECIFIED=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument '$1'." >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

case "$BENCHMARK_SELECTION" in
  all)
    RUN_DP_DECODE=1
    RUN_DP_PREFILL=1
    RUN_PCP_PREFILL=1
    BENCHMARK_CONFIGS_JSON='["dp4_tp2_decode_c256", "dp8", "pcp8"]'
    ;;
  dp-decode)
    RUN_DP_DECODE=1
    BENCHMARK_CONFIGS_JSON='["dp4_tp2_decode_c256"]'
    ;;
  dp-prefill)
    RUN_DP_PREFILL=1
    BENCHMARK_CONFIGS_JSON='["dp8"]'
    ;;
  pcp-prefill)
    RUN_PCP_PREFILL=1
    BENCHMARK_CONFIGS_JSON='["pcp8"]'
    ;;
  *)
    echo "ERROR: invalid --only benchmark '$BENCHMARK_SELECTION'." >&2
    echo "Expected dp-decode, dp-prefill, or pcp-prefill." >&2
    exit 2
    ;;
esac
RUN_PREFILL=$(( RUN_DP_PREFILL || RUN_PCP_PREFILL ))

case "$PREFILL_MODE" in
  all)
    RUN_PREFILL_THROUGHPUT=$RUN_PREFILL
    RUN_PREFILL_TTFT=$RUN_PREFILL
    ;;
  throughput)
    RUN_PREFILL_THROUGHPUT=$RUN_PREFILL
    ;;
  ttft)
    RUN_PREFILL_TTFT=$RUN_PREFILL
    ;;
  *)
    echo "ERROR: invalid --prefill-mode '$PREFILL_MODE'." >&2
    echo "Expected all, throughput, or ttft." >&2
    exit 2
    ;;
esac
if (( PREFILL_MODE_SPECIFIED && ! RUN_PREFILL )); then
  echo "ERROR: --prefill-mode requires a selected DP/PCP prefill benchmark." >&2
  exit 2
fi

case "$PREFILL_WORKLOAD" in
  all)
    RUN_SYNTHETIC_PREFILL=$RUN_PREFILL
    RUN_DP_SPEED_BENCH_PREFILL=$RUN_DP_PREFILL
    RUN_PCP_SPEED_BENCH_PREFILL=$RUN_PCP_PREFILL
    ;;
  synthetic)
    RUN_SYNTHETIC_PREFILL=$RUN_PREFILL
    ;;
  speed-bench)
    RUN_DP_SPEED_BENCH_PREFILL=$RUN_DP_PREFILL
    RUN_PCP_SPEED_BENCH_PREFILL=$RUN_PCP_PREFILL
    ;;
  *)
    echo "ERROR: invalid --prefill-workload '$PREFILL_WORKLOAD'." >&2
    echo "Expected all, synthetic, or speed-bench." >&2
    exit 2
    ;;
esac
if (( PREFILL_WORKLOAD_SPECIFIED && ! RUN_PREFILL )); then
  echo "ERROR: --prefill-workload requires a selected DP/PCP prefill benchmark." >&2
  exit 2
fi

for value_name in PORT SERVER_READY_TIMEOUT SERVER_STOP_TIMEOUT; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value == 0 )); then
    echo "ERROR: $value_name must be a positive integer, got '$value'." >&2
    exit 2
  fi
done
if [[ "$KEEP_SERVER_RUNNING" != 0 && "$KEEP_SERVER_RUNNING" != 1 ]]; then
  echo "ERROR: KEEP_SERVER_RUNNING must be 0 or 1." >&2
  exit 2
fi
if [[ "$PUBLISH_REPORTS" != 0 && "$PUBLISH_REPORTS" != 1 ]]; then
  echo "ERROR: PUBLISH_REPORTS must be 0 or 1." >&2
  exit 2
fi
if (( TORCHTPU_COMMIT_SPECIFIED )) &&
    [[ ! "$TORCHTPU_COMMIT" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
  echo "ERROR: --commit must be a 7- to 40-character hexadecimal Git commit ID." >&2
  exit 2
fi
if (( TEST_ONLY && PREPARE_ONLY )); then
  echo "ERROR: --test-only and --prepare-only cannot be used together." >&2
  exit 2
fi
if (( TEST_ONLY && TORCHTPU_COMMIT_SPECIFIED )); then
  echo "ERROR: --commit cannot be used with --test-only because test-only does not update source." >&2
  exit 2
fi
if (( TEST_ONLY && RUN_DP_DECODE && ! RUN_DP_PREFILL && ! RUN_PCP_PREFILL )); then
  echo "ERROR: --test-only currently covers DP/PCP prefill benchmarks only." >&2
  exit 2
fi

detect_machine_ip() {
  local candidate

  if command -v ip >/dev/null 2>&1; then
    candidate=$(
      ip -4 route get 1.1.1.1 2>/dev/null |
        awk '
          {
            for (field = 1; field <= NF; field++) {
              if ($field == "src") {
                print $(field + 1)
                exit
              }
            }
          }
        '
    )
    if [[ -n "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  fi

  if command -v hostname >/dev/null 2>&1; then
    candidate=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [[ -n "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  fi

  return 1
}

if [[ -z "$MACHINE_IP" ]]; then
  MACHINE_IP=$(detect_machine_ip) || {
    echo "ERROR: could not determine the machine IP address." >&2
    echo "Set MACHINE_IP explicitly before running the benchmark." >&2
    exit 1
  }
fi
python3.12 - "$MACHINE_IP" <<'PY'
import ipaddress
import sys

try:
    ipaddress.ip_address(sys.argv[1])
except ValueError as error:
    raise SystemExit(f"ERROR: MACHINE_IP is not a valid IP address: {error}")
PY

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
benchmark_started_at="$(
  printf '%s-%s-%sT%s:%s:%s+00:00' \
    "${timestamp:0:4}" "${timestamp:4:2}" "${timestamp:6:2}" \
    "${timestamp:9:2}" "${timestamp:11:2}" "${timestamp:13:2}"
)"
if (( TEST_ONLY )); then
  TEST_PREVIEW_ROOT="$STATE_DIR/test-only-preview/$timestamp"
  TEST_PROJECT_ROOT="$TEST_PREVIEW_ROOT/project"
  RUN_DIR="$TEST_PREVIEW_ROOT/runs/$timestamp"
else
  RUN_DIR="$PROJECT_ROOT/runs/$timestamp"
fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/job.log") 2>&1

echo "Daily TPU benchmark started at $benchmark_started_at"
echo "Project root: $PROJECT_ROOT"
echo "Run directory: $RUN_DIR"
echo "Machine IP: $MACHINE_IP"
echo "Benchmark selection: $BENCHMARK_SELECTION"
echo "Prefill mode: $PREFILL_MODE"
echo "Prefill workload: $PREFILL_WORKLOAD"
echo "vLLM service launcher: $VLLM_SERVICE_LAUNCH"
echo "vLLM service state: $VLLM_SERVICE_STATE_ROOT"
echo "vLLM metrics targets: $VLLM_SERVICE_TARGET_ROOT"
if (( TORCHTPU_COMMIT_SPECIFIED )); then
  echo "Requested vllm-torchtpu commit: $TORCHTPU_COMMIT"
else
  echo "Requested vllm-torchtpu source: latest origin/main"
fi

if (( TEST_ONLY )); then
  echo "TEST_ONLY: building an isolated fixture-backed report preview."
  TEST_ONLY_PYTHON="${TEST_ONLY_PYTHON:-python3.12}"
  if ! command -v "$TEST_ONLY_PYTHON" >/dev/null 2>&1; then
    echo "ERROR: TEST_ONLY Python is missing: $TEST_ONLY_PYTHON" >&2
    exit 1
  fi
  mkdir -p "$TEST_PROJECT_ROOT"
  cp "$PROJECT_ROOT/README.md" "$TEST_PROJECT_ROOT/README.md"
  cp -a "$PROJECT_ROOT/reports" "$TEST_PROJECT_ROOT/reports"
  mkdir -p "$TEST_PROJECT_ROOT/.state"

  cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "started_at": "$benchmark_started_at",
  "machine_ip": "$MACHINE_IP",
  "benchmark_selection": "$BENCHMARK_SELECTION",
  "prefill_mode": "$PREFILL_MODE",
  "prefill_workload": "$PREFILL_WORKLOAD",
  "torchtpu_vllm_revision": "test-only-fixture",
  "torch_tpu_version": "test-only-fixture",
  "benchmark_configs": $BENCHMARK_CONFIGS_JSON,
  "test_only": true,
  "port": $PORT
}
EOF

  if (( RUN_DP_PREFILL )); then
    "$SCRIPT_DIR/start_dp_server.sh" --test-only
    if (( RUN_SYNTHETIC_PREFILL )); then
      dp_test_prefill_status=not-run
      dp_test_ttft_status=not-run
      dp_test_report_args=()
      if (( RUN_PREFILL_THROUGHPUT )); then
        BENCHMARK_CONFIG=dp8 TEST_ONLY=1 UPDATE_REPORTS=0 PUBLISH_REPORTS=0 \
          "$SCRIPT_DIR/bench_all.sh" "$RUN_DIR"
        dp_test_prefill_status=success
        dp_test_report_args+=(
          --summary "$RUN_DIR/results/dp8/summary.json"
        )
      fi
      if (( RUN_PREFILL_TTFT )); then
        BENCHMARK_CONFIG=dp8 TEST_ONLY=1 \
          "$SCRIPT_DIR/bench_prefill_ttft.sh" "$RUN_DIR"
        dp_test_ttft_status=$(
          "$TEST_ONLY_PYTHON" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
            "$RUN_DIR/results/dp8/single_request_ttft/summary.json"
        )
        dp_test_report_args+=(
          --ttft-summary "$RUN_DIR/results/dp8/single_request_ttft/summary.json"
        )
      fi
      "$TEST_ONLY_PYTHON" "$SCRIPT_DIR/update_report.py" \
        --project-root "$TEST_PROJECT_ROOT" \
        --run-dir "$RUN_DIR" \
        --benchmark-config dp8 \
        --status "$dp_test_prefill_status" \
        --decode-status not-run \
        --ttft-status "$dp_test_ttft_status" \
        --input-length 8192 \
        --output-length 1 \
        --model Qwen3.5-397B-A17B-FP8 \
        "${dp_test_report_args[@]}"
    fi
    if (( RUN_DP_SPEED_BENCH_PREFILL )); then
      BENCHMARK_CONFIG=dp8 SPEED_BENCH_MODE="$PREFILL_MODE" TEST_ONLY=1 \
        "$SCRIPT_DIR/bench_speed_bench_mix.sh" "$RUN_DIR"
      "$TEST_ONLY_PYTHON" "$SCRIPT_DIR/update_speed_bench_report.py" \
        --project-root "$TEST_PROJECT_ROOT" \
        --run-dir "$RUN_DIR" \
        --summary "$RUN_DIR/results/dp8/speed_bench_mix/summary.json" \
        --model Qwen3.5-397B-A17B-FP8
    fi
  fi
  if (( RUN_PCP_PREFILL )); then
    "$SCRIPT_DIR/start_pcp_server.sh" --test-only
    if (( RUN_SYNTHETIC_PREFILL )); then
      pcp_test_prefill_status=not-run
      pcp_test_ttft_status=not-run
      pcp_test_report_args=()
      if (( RUN_PREFILL_THROUGHPUT )); then
        BENCHMARK_CONFIG=pcp8 TEST_ONLY=1 UPDATE_REPORTS=0 PUBLISH_REPORTS=0 \
          "$SCRIPT_DIR/bench_all.sh" "$RUN_DIR"
        pcp_test_prefill_status=success
        pcp_test_report_args+=(
          --summary "$RUN_DIR/results/pcp8/summary.json"
        )
      fi
      if (( RUN_PREFILL_TTFT )); then
        BENCHMARK_CONFIG=pcp8 TEST_ONLY=1 \
          "$SCRIPT_DIR/bench_prefill_ttft.sh" "$RUN_DIR"
        pcp_test_ttft_status=$(
          "$TEST_ONLY_PYTHON" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
            "$RUN_DIR/results/pcp8/single_request_ttft/summary.json"
        )
        pcp_test_report_args+=(
          --ttft-summary "$RUN_DIR/results/pcp8/single_request_ttft/summary.json"
        )
      fi
      "$TEST_ONLY_PYTHON" "$SCRIPT_DIR/update_report.py" \
        --project-root "$TEST_PROJECT_ROOT" \
        --run-dir "$RUN_DIR" \
        --benchmark-config pcp8 \
        --status "$pcp_test_prefill_status" \
        --decode-status not-run \
        --ttft-status "$pcp_test_ttft_status" \
        --input-length 8192 \
        --output-length 1 \
        --model Qwen3.5-397B-A17B-FP8 \
        "${pcp_test_report_args[@]}"
    fi
    if (( RUN_PCP_SPEED_BENCH_PREFILL )); then
      BENCHMARK_CONFIG=pcp8 SPEED_BENCH_MODE="$PREFILL_MODE" TEST_ONLY=1 \
        "$SCRIPT_DIR/bench_speed_bench_mix.sh" "$RUN_DIR"
      "$TEST_ONLY_PYTHON" "$SCRIPT_DIR/update_speed_bench_report.py" \
        --project-root "$TEST_PROJECT_ROOT" \
        --run-dir "$RUN_DIR" \
        --summary "$RUN_DIR/results/pcp8/speed_bench_mix/summary.json" \
        --model Qwen3.5-397B-A17B-FP8
    fi
  fi
  echo "TEST_ONLY preview README: $TEST_PROJECT_ROOT/README.md"
  echo "TEST_ONLY preview reports: $TEST_PROJECT_ROOT/reports"
  exit 0
fi

if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "ERROR: model weight index is missing: $MODEL_DIR/model.safetensors.index.json" >&2
  echo "Set MODEL_DIR to the complete Qwen3.5-397B-A17B-FP8 checkpoint." >&2
  exit 1
fi

if (( ! PREPARE_ONLY )); then
  if [[ ! -x "$VLLM_SERVICE_LAUNCH" ]]; then
    echo "ERROR: vllm-service-launch is not executable: $VLLM_SERVICE_LAUNCH" >&2
    exit 1
  fi
  for service_id in \
      "$DP_DECODE_SERVICE_ID" \
      "$DP_PREFILL_SERVICE_ID" \
      "$PCP_PREFILL_SERVICE_ID"; do
    if "$VLLM_SERVICE_LAUNCH" status \
        "${VLLM_SERVICE_LAYOUT_ARGS[@]}" \
        --service-id "$service_id" >/dev/null 2>&1; then
      echo "Stopping existing launcher service $service_id..."
      "$VLLM_SERVICE_LAUNCH" stop \
        "${VLLM_SERVICE_LAYOUT_ARGS[@]}" \
        --service-id "$service_id"
    fi
  done
fi

environment_update_args=()
if [[ -n "$TORCHTPU_COMMIT" ]]; then
  environment_update_args+=(--commit "$TORCHTPU_COMMIT")
fi
"$SCRIPT_DIR/update_environment.sh" "${environment_update_args[@]}"

source_revision=$(git -C "$TORCHTPU_DIR" rev-parse HEAD)
torch_tpu_version=$(
  "$VENV_DIR/bin/python" -c \
    'from importlib.metadata import version; print(version("torch-tpu"))'
)
model_revision=""
if (( RUN_PREFILL )); then
  model_revision=$(python3.12 -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["revision"])' \
    "$MODEL_DIR/SOURCE.json")
fi
cp "$STATE_DIR/environment.freeze.txt" "$RUN_DIR/environment.freeze.txt"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VENV_DIR/bin/python" - "$MODEL_DIR" <<'PY'
import sys
from transformers import AutoConfig, AutoTokenizer

model_dir = sys.argv[1]
config = AutoConfig.from_pretrained(
    model_dir, local_files_only=True, trust_remote_code=False
)
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, local_files_only=True, trust_remote_code=False
)
print(
    "Offline model metadata OK: "
    f"config={type(config).__name__}, tokenizer={type(tokenizer).__name__}"
)
PY

cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "started_at": "$benchmark_started_at",
  "machine_ip": "$MACHINE_IP",
  "benchmark_selection": "$BENCHMARK_SELECTION",
  "prefill_mode": "$PREFILL_MODE",
  "prefill_workload": "$PREFILL_WORKLOAD",
  "torchtpu_vllm_revision": "$source_revision",
  "torch_tpu_version": "$torch_tpu_version",
  "torch_tpu_install_source": "pip",
  "model_directory": "$MODEL_DIR",
  "model_revision": "$model_revision",
  "model_load_format": "auto",
  "decode_model_directory": "$MODEL_DIR",
  "decode_model_load_format": "auto",
  "decode_parallelism": "DP4/TP2/EP8",
  "decode_workload": "C256/P65536/D1024",
  "benchmark_configs": $BENCHMARK_CONFIGS_JSON,
  "port": $PORT
}
EOF

if (( PREPARE_ONLY )); then
  echo "Preparation completed; TPU server was not started."
  exit 0
fi

SERVER_CONFIG=""
SERVER_SERVICE_ID=""
RUN_SUCCEEDED=0
REPORT_GENERATED=0
BENCHMARK_FAILURES=0

server_port_is_bindable() {
  "$VENV_DIR/bin/python" - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    try:
        probe.bind(("0.0.0.0", port))
    except OSError:
        raise SystemExit(1)
PY
}

stop_server() {
  local target_config
  local target_service_id
  local waited

  if [[ -z "$SERVER_SERVICE_ID" ]]; then
    return
  fi
  if (( KEEP_SERVER_RUNNING && RUN_SUCCEEDED )); then
    echo "Keeping $SERVER_CONFIG launcher service $SERVER_SERVICE_ID running."
    return
  fi

  target_config=$SERVER_CONFIG
  target_service_id=$SERVER_SERVICE_ID
  echo "Stopping $target_config launcher service $target_service_id..."
  if "$VLLM_SERVICE_LAUNCH" status \
      "${VLLM_SERVICE_LAYOUT_ARGS[@]}" \
      --service-id "$target_service_id" >/dev/null 2>&1; then
    "$VLLM_SERVICE_LAUNCH" stop \
      "${VLLM_SERVICE_LAYOUT_ARGS[@]}" \
      --service-id "$target_service_id"
  fi
  for (( waited = 0; waited < SERVER_STOP_TIMEOUT; waited += 1 )); do
    if server_port_is_bindable; then
      break
    fi
    if (( waited > 0 && waited % 10 == 0 )); then
      echo "Waiting for $target_config server to release port $PORT..."
    fi
    sleep 1
  done
  if ! server_port_is_bindable; then
    echo "ERROR: $target_config server did not release port $PORT within ${SERVER_STOP_TIMEOUT}s." >&2
    return 1
  fi
  SERVER_SERVICE_ID=""
  SERVER_CONFIG=""
  echo "$target_config server stopped."
}

start_server() {
  local benchmark_config=$1
  local server_script=$2
  local server_model_dir=$3
  local service_id=$4
  local status_path="$RUN_DIR/${benchmark_config}_service_status.json"
  local ready=0

  if [[ -n "$SERVER_SERVICE_ID" ]]; then
    echo "ERROR: cannot start $benchmark_config while $SERVER_CONFIG is running." >&2
    return 1
  fi

  echo "Starting $benchmark_config inference server..."
  SERVER_CONFIG=$benchmark_config
  SERVER_SERVICE_ID=$service_id
  if ! env \
    PORT="$PORT" \
    VENV_DIR="$VENV_DIR" \
    TORCHTPU_DIR="$TORCHTPU_DIR" \
    MODEL_DIR="$server_model_dir" \
    "$server_script"; then
    echo "ERROR: $benchmark_config launcher submission failed." >&2
    return 1
  fi

  for (( waited = 0; waited < SERVER_READY_TIMEOUT; waited += 2 )); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
      ready=1
      break
    fi
    if ! "$VLLM_SERVICE_LAUNCH" status \
        "${VLLM_SERVICE_LAYOUT_ARGS[@]}" \
        --service-id "$service_id" \
        --json > "$status_path"; then
      echo "ERROR: $benchmark_config server exited during startup." >&2
      return 1
    fi
    if (( waited > 0 && waited % 60 == 0 )); then
      echo "Waiting for $benchmark_config server... ${waited}s elapsed"
    fi
    sleep 2
  done

  if (( ! ready )); then
    echo "ERROR: $benchmark_config server did not become healthy within ${SERVER_READY_TIMEOUT}s." >&2
    return 1
  fi
  "$VLLM_SERVICE_LAUNCH" status \
    "${VLLM_SERVICE_LAYOUT_ARGS[@]}" \
    --service-id "$service_id" \
    --json > "$status_path"
  echo "$benchmark_config server is healthy on port $PORT."
}

run_decode_smoke() {
  local result_dir=$1
  local smoke_dir="$result_dir/smoke"

  echo "Running real-weight DP4/TP2 C8 decode smoke..."
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/bench_decode_sliding_window.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --model Qwen3.5-397B-A17B-FP8 \
    --output-dir "$smoke_dir" \
    --concurrency 8 \
    --data-parallel-size 4 \
    --tensor-parallel-size 2 \
    --prefill-tokens 65536 \
    --decode-tokens 32 \
    --tokenizer-dir "$MODEL_DIR" \
    --rounds 1 \
    --window-seconds 0.1 \
    --step-seconds 0.05 \
    --cache-salt-prefix "tpu-daily-${timestamp}-smoke" \
    2>&1 | tee "$RUN_DIR/dp4_tp2_decode_c256_smoke.log"
}

run_decode_round() {
  local result_dir=$1
  local run_index=$2
  local run_dir="$result_dir/run_${run_index}"

  echo "Running real-weight DP4/TP2 C256 decode round $run_index/1..."
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/bench_decode_sliding_window.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --model Qwen3.5-397B-A17B-FP8 \
    --output-dir "$run_dir" \
    --concurrency 256 \
    --data-parallel-size 4 \
    --tensor-parallel-size 2 \
    --prefill-tokens 65536 \
    --decode-tokens 1024 \
    --tokenizer-dir "$MODEL_DIR" \
    --rounds 1 \
    --window-seconds 1 \
    --step-seconds 0.1 \
    --cache-salt-prefix "tpu-daily-${timestamp}-run${run_index}" \
    2>&1 | tee "$RUN_DIR/dp4_tp2_decode_c256_run_${run_index}.log"
}

run_decode_benchmark() {
  local result_dir="$RUN_DIR/results/dp4_tp2_decode_c256"

  mkdir -p "$result_dir"
  if ! run_decode_smoke "$result_dir"; then
    return 1
  fi
  if ! run_decode_round "$result_dir" 1; then
    return 1
  fi
  if ! "$VENV_DIR/bin/python" "$SCRIPT_DIR/aggregate_decode_runs.py" \
      --result-root "$result_dir" \
      --runs 1; then
    return 1
  fi

  if ! curl -fsS --max-time 5 \
      "http://127.0.0.1:$PORT/health" >/dev/null; then
    return 1
  fi
  echo "DP4/TP2 C256 decode benchmark completed successfully."
}

run_prefill_benchmark() {
  local benchmark_config=$1

  echo "Running $benchmark_config prefill benchmark..."
  if ! BENCHMARK_CONFIG="$benchmark_config" \
      PUBLISH_REPORTS=0 \
      UPDATE_REPORTS=0 \
      "$SCRIPT_DIR/bench_all.sh" "$RUN_DIR" \
      2>&1 | tee "$RUN_DIR/${benchmark_config}_prefill_benchmark.log"; then
    return 1
  fi

  if ! curl -fsS --max-time 5 \
      "http://127.0.0.1:$PORT/health" >/dev/null; then
    return 1
  fi
  echo "$benchmark_config prefill benchmark completed successfully."
}

run_prefill_ttft_benchmark() {
  local benchmark_config=$1
  local summary_path="$RUN_DIR/results/$benchmark_config/single_request_ttft/summary.json"

  echo "Running $benchmark_config single-request prefill TTFT benchmark..."
  if ! BENCHMARK_CONFIG="$benchmark_config" \
      "$SCRIPT_DIR/bench_prefill_ttft.sh" "$RUN_DIR" \
      2>&1 | tee "$RUN_DIR/${benchmark_config}_prefill_ttft_benchmark.log"; then
    return 1
  fi

  PREFILL_TTFT_LAST_STATUS=$(
    "$VENV_DIR/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "$summary_path"
  )
  case "$PREFILL_TTFT_LAST_STATUS" in
    success|partial|failed) ;;
    *)
      echo "ERROR: invalid TTFT summary status '$PREFILL_TTFT_LAST_STATUS'." >&2
      return 1
      ;;
  esac
  if ! curl -fsS --max-time 5 \
      "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "WARNING: $benchmark_config server exited after TTFT results were saved." >&2
  fi
  echo "$benchmark_config single-request TTFT benchmark status: $PREFILL_TTFT_LAST_STATUS."
}

run_speed_bench_benchmark() {
  local benchmark_config=$1
  local config_label
  local summary_path="$RUN_DIR/results/$benchmark_config/speed_bench_mix/summary.json"
  local command_status=0

  case "$benchmark_config" in
    dp8) config_label=DP8 ;;
    pcp8) config_label=PCP8 ;;
    *)
      echo "ERROR: unsupported SPEED-Bench config '$benchmark_config'." >&2
      SPEED_BENCH_LAST_STATUS=failed
      return 1
      ;;
  esac

  SPEED_BENCH_LAST_STATUS=failed
  echo "Running $config_label semantic mixed-length SPEED-Bench workload..."
  if ! BENCHMARK_CONFIG="$benchmark_config" \
      SPEED_BENCH_MODE="$PREFILL_MODE" \
      "$SCRIPT_DIR/bench_speed_bench_mix.sh" "$RUN_DIR" \
      2>&1 | tee "$RUN_DIR/${benchmark_config}_speed_bench_mix.log"; then
    command_status=1
  fi
  if [[ ! -f "$summary_path" ]]; then
    echo "ERROR: $config_label SPEED-Bench mixed-workload summary is missing." >&2
    return 1
  fi
  SPEED_BENCH_LAST_STATUS=$(
    "$VENV_DIR/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "$summary_path"
  )
  case "$SPEED_BENCH_LAST_STATUS" in
    success|partial|failed) ;;
    *)
      echo "ERROR: invalid SPEED-Bench status '$SPEED_BENCH_LAST_STATUS'." >&2
      SPEED_BENCH_LAST_STATUS=failed
      return 1
      ;;
  esac
  if ! curl -fsS --max-time 5 \
      "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "WARNING: $config_label server is unavailable after SPEED-Bench." >&2
  fi
  echo "$config_label SPEED-Bench mixed-workload status: $SPEED_BENCH_LAST_STATUS."
  if (( command_status )) || [[ "$SPEED_BENCH_LAST_STATUS" != success ]]; then
    return 1
  fi
}

on_signal() {
  echo "Received termination signal."
  exit 130
}

record_dp_report() {
  local -a report_args=(
    --project-root "$PROJECT_ROOT"
    --run-dir "$RUN_DIR"
    --benchmark-config dp8
    --status "$DP_PREFILL_STATUS"
    --decode-status "$DP_DECODE_STATUS"
    --decode-parallelism DP4/TP2/EP8
    --input-length 8192
    --output-length 1
    --model Qwen3.5-397B-A17B-FP8
  )

  if [[ "$DP_PREFILL_STATUS" == success ]]; then
    report_args+=(--summary "$RUN_DIR/results/dp8/summary.json")
  fi
  if [[ "$DP_DECODE_STATUS" == success ]]; then
    report_args+=(
      --decode-summary "$RUN_DIR/results/dp4_tp2_decode_c256/aggregate.json"
    )
  fi
  report_args+=(--ttft-status "$DP_PREFILL_TTFT_STATUS")
  if [[ -f "$RUN_DIR/results/dp8/single_request_ttft/summary.json" ]]; then
    report_args+=(
      --ttft-summary "$RUN_DIR/results/dp8/single_request_ttft/summary.json"
    )
  fi

  "$VENV_DIR/bin/python" "$SCRIPT_DIR/update_report.py" \
    "${report_args[@]}"
  REPORT_GENERATED=1
}

record_pcp_report() {
  local -a report_args=(
    --project-root "$PROJECT_ROOT"
    --run-dir "$RUN_DIR"
    --benchmark-config pcp8
    --status "$PCP_PREFILL_STATUS"
    --decode-status not-run
    --input-length 8192
    --output-length 1
    --model Qwen3.5-397B-A17B-FP8
  )

  if [[ "$PCP_PREFILL_STATUS" == success ]]; then
    report_args+=(--summary "$RUN_DIR/results/pcp8/summary.json")
  fi
  report_args+=(--ttft-status "$PCP_PREFILL_TTFT_STATUS")
  if [[ -f "$RUN_DIR/results/pcp8/single_request_ttft/summary.json" ]]; then
    report_args+=(
      --ttft-summary "$RUN_DIR/results/pcp8/single_request_ttft/summary.json"
    )
  fi

  "$VENV_DIR/bin/python" "$SCRIPT_DIR/update_report.py" \
    "${report_args[@]}"
  REPORT_GENERATED=1
}

record_speed_bench_report() {
  local benchmark_config=$1
  local summary_path="$RUN_DIR/results/$benchmark_config/speed_bench_mix/summary.json"

  if [[ ! -f "$summary_path" ]]; then
    echo "ERROR: cannot record $benchmark_config SPEED-Bench report without $summary_path." >&2
    return 1
  fi
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/update_speed_bench_report.py" \
    --project-root "$PROJECT_ROOT" \
    --run-dir "$RUN_DIR" \
    --summary "$summary_path" \
    --model Qwen3.5-397B-A17B-FP8
  REPORT_GENERATED=1
}

trap stop_server EXIT
trap on_signal INT TERM

if (( RUN_DP_DECODE )); then
  if start_server \
      dp4_tp2_decode_c256 \
      "$SCRIPT_DIR/start_dp_decode_server.sh" \
      "$MODEL_DIR" \
      "$DP_DECODE_SERVICE_ID"; then
    if run_decode_benchmark; then
      DP_DECODE_STATUS=success
    else
      DP_DECODE_STATUS=failed
      echo "ERROR: DP4/TP2 C256 decode benchmark failed; recording -1 tok/s." >&2
    fi
  else
    DP_DECODE_STATUS=failed
    echo "ERROR: DP4/TP2 C256 decode server failed; recording -1 tok/s." >&2
  fi
  if [[ "$DP_DECODE_STATUS" == failed ]] ||
      (( RUN_DP_PREFILL || RUN_PCP_PREFILL )); then
    stop_server
  fi
fi

if (( RUN_DP_PREFILL )); then
  if start_server \
      dp8 \
      "$SCRIPT_DIR/start_dp_server.sh" \
      "$MODEL_DIR" \
      "$DP_PREFILL_SERVICE_ID"; then
    if (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_THROUGHPUT )); then
      if run_prefill_benchmark dp8; then
        DP_PREFILL_STATUS=success
      else
        DP_PREFILL_STATUS=failed
        echo "ERROR: DP8 prefill throughput benchmark failed; recording -1 tok/s." >&2
      fi
    fi
    if (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_TTFT )); then
      if run_prefill_ttft_benchmark dp8; then
        DP_PREFILL_TTFT_STATUS=$PREFILL_TTFT_LAST_STATUS
      else
        DP_PREFILL_TTFT_STATUS=failed
        echo "ERROR: DP8 single-request TTFT benchmark failed." >&2
      fi
    fi
    if (( RUN_DP_SPEED_BENCH_PREFILL )); then
      if ! run_speed_bench_benchmark dp8; then
        echo "ERROR: DP8 SPEED-Bench mixed workload failed." >&2
      fi
      DP_SPEED_BENCH_STATUS=$SPEED_BENCH_LAST_STATUS
    fi
  else
    if (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_THROUGHPUT )); then
      DP_PREFILL_STATUS=failed
      echo "ERROR: DP8 prefill server failed; recording -1 tok/s." >&2
    fi
    if (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_TTFT )); then
      DP_PREFILL_TTFT_STATUS=failed
      echo "ERROR: DP8 TTFT server failed." >&2
    fi
    if (( RUN_DP_SPEED_BENCH_PREFILL )); then
      if ! run_speed_bench_benchmark dp8; then
        echo "ERROR: DP8 SPEED-Bench server failed." >&2
      fi
      DP_SPEED_BENCH_STATUS=$SPEED_BENCH_LAST_STATUS
    fi
  fi
  if { (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_THROUGHPUT )) &&
       [[ "$DP_PREFILL_STATUS" == failed ]]; } ||
      { (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_TTFT )) &&
        [[ "$DP_PREFILL_TTFT_STATUS" != success ]]; } ||
      { (( RUN_DP_SPEED_BENCH_PREFILL )) &&
        [[ "$DP_SPEED_BENCH_STATUS" != success ]]; } ||
      (( RUN_PCP_PREFILL )); then
    stop_server
  fi
fi

if (( RUN_PCP_PREFILL )); then
  if start_server \
      pcp8 \
      "$SCRIPT_DIR/start_pcp_server.sh" \
      "$MODEL_DIR" \
      "$PCP_PREFILL_SERVICE_ID"; then
    if (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_THROUGHPUT )); then
      if run_prefill_benchmark pcp8; then
        PCP_PREFILL_STATUS=success
      else
        PCP_PREFILL_STATUS=failed
        echo "ERROR: PCP8 prefill throughput benchmark failed; recording -1 tok/s." >&2
      fi
    fi
    if (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_TTFT )); then
      if run_prefill_ttft_benchmark pcp8; then
        PCP_PREFILL_TTFT_STATUS=$PREFILL_TTFT_LAST_STATUS
      else
        PCP_PREFILL_TTFT_STATUS=failed
        echo "ERROR: PCP8 single-request TTFT benchmark failed." >&2
      fi
    fi
    if (( RUN_PCP_SPEED_BENCH_PREFILL )); then
      if ! run_speed_bench_benchmark pcp8; then
        echo "ERROR: PCP8 SPEED-Bench mixed workload failed." >&2
      fi
      PCP_SPEED_BENCH_STATUS=$SPEED_BENCH_LAST_STATUS
    fi
  else
    if (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_THROUGHPUT )); then
      PCP_PREFILL_STATUS=failed
      echo "ERROR: PCP8 prefill server failed; recording -1 tok/s." >&2
    fi
    if (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_TTFT )); then
      PCP_PREFILL_TTFT_STATUS=failed
      echo "ERROR: PCP8 TTFT server failed." >&2
    fi
    if (( RUN_PCP_SPEED_BENCH_PREFILL )); then
      if ! run_speed_bench_benchmark pcp8; then
        echo "ERROR: PCP8 SPEED-Bench server failed." >&2
      fi
      PCP_SPEED_BENCH_STATUS=$SPEED_BENCH_LAST_STATUS
    fi
  fi
  if { (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_THROUGHPUT )) &&
       [[ "$PCP_PREFILL_STATUS" == failed ]]; } ||
      { (( RUN_SYNTHETIC_PREFILL && RUN_PREFILL_TTFT )) &&
        [[ "$PCP_PREFILL_TTFT_STATUS" != success ]]; } ||
      { (( RUN_PCP_SPEED_BENCH_PREFILL )) &&
        [[ "$PCP_SPEED_BENCH_STATUS" != success ]]; }; then
    stop_server
  fi
fi

if (( RUN_DP_DECODE || (RUN_DP_PREFILL && RUN_SYNTHETIC_PREFILL) )); then
  record_dp_report
fi
if (( RUN_PCP_PREFILL && RUN_SYNTHETIC_PREFILL )); then
  record_pcp_report
fi
if (( RUN_DP_SPEED_BENCH_PREFILL )); then
  if ! record_speed_bench_report dp8; then
    DP_SPEED_BENCH_STATUS=failed
    echo "ERROR: failed to record DP8 SPEED-Bench report." >&2
  fi
fi
if (( RUN_PCP_SPEED_BENCH_PREFILL )); then
  if ! record_speed_bench_report pcp8; then
    PCP_SPEED_BENCH_STATUS=failed
    echo "ERROR: failed to record PCP8 SPEED-Bench report." >&2
  fi
fi

if (( PUBLISH_REPORTS && REPORT_GENERATED )); then
  "$SCRIPT_DIR/publish_report.sh"
elif (( PUBLISH_REPORTS )); then
  echo "Skipping Git report publication because no report was generated."
else
  echo "Skipping Git report publication because PUBLISH_REPORTS=0."
fi

if (( RUN_DP_DECODE )) && [[ "$DP_DECODE_STATUS" == failed ]]; then
  (( BENCHMARK_FAILURES += 1 ))
fi
if (( RUN_DP_PREFILL && RUN_SYNTHETIC_PREFILL && RUN_PREFILL_THROUGHPUT )) &&
    [[ "$DP_PREFILL_STATUS" == failed ]]; then
  (( BENCHMARK_FAILURES += 1 ))
fi
if (( RUN_DP_PREFILL && RUN_SYNTHETIC_PREFILL && RUN_PREFILL_TTFT )) &&
    [[ "$DP_PREFILL_TTFT_STATUS" != success ]] &&
    [[ "$DP_PREFILL_TTFT_STATUS" != not-run ]]; then
  (( BENCHMARK_FAILURES += 1 ))
fi
if (( RUN_PCP_PREFILL && RUN_SYNTHETIC_PREFILL && RUN_PREFILL_THROUGHPUT )) &&
    [[ "$PCP_PREFILL_STATUS" == failed ]]; then
  (( BENCHMARK_FAILURES += 1 ))
fi
if (( RUN_PCP_PREFILL && RUN_SYNTHETIC_PREFILL && RUN_PREFILL_TTFT )) &&
    [[ "$PCP_PREFILL_TTFT_STATUS" != success ]] &&
    [[ "$PCP_PREFILL_TTFT_STATUS" != not-run ]]; then
  (( BENCHMARK_FAILURES += 1 ))
fi
if (( RUN_DP_SPEED_BENCH_PREFILL )) &&
    [[ "$DP_SPEED_BENCH_STATUS" != success ]]; then
  (( BENCHMARK_FAILURES += 1 ))
fi
if (( RUN_PCP_SPEED_BENCH_PREFILL )) &&
    [[ "$PCP_SPEED_BENCH_STATUS" != success ]]; then
  (( BENCHMARK_FAILURES += 1 ))
fi

if (( BENCHMARK_FAILURES )); then
  echo "Daily TPU benchmark completed with $BENCHMARK_FAILURES failed group(s)"
  echo "at $(date -u --iso-8601=seconds); reports were still generated."
  exit 1
fi

RUN_SUCCEEDED=1
echo "Daily TPU benchmark completed successfully at $(date -u --iso-8601=seconds)"
