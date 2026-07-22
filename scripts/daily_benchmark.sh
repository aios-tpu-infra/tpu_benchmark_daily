#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
STATE_DIR="${STATE_DIR:-$PROJECT_ROOT/.state}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TORCHTPU_DIR="${TORCHTPU_DIR:-$PROJECT_ROOT/third_party/torchtpu-vllm}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-397B-A17B-FP8}"
PORT="${PORT:-18100}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-3600}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-60}"
KEEP_SERVER_RUNNING="${KEEP_SERVER_RUNNING:-0}"
MACHINE_IP="${MACHINE_IP:-}"
PREPARE_ONLY=0

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
Usage: scripts/daily_benchmark.sh [--prepare-only] [--keep-server-running]

  --prepare-only         Prepare source/environment without touching a server.
  --keep-server-running  Keep a successfully benchmarked server alive.

The default full workflow stops an existing vLLM service on PORT, updates
vllm-torchtpu/main, installs its compatible torch_tpu version with pip, updates
.venv, starts the dummy-weight server, waits for /health, runs all benchmarks,
saves results, and stops it.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --prepare-only)
      PREPARE_ONLY=1
      ;;
    --keep-server-running)
      KEEP_SERVER_RUNNING=1
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

list_port_listener_pids() {
  if command -v lsof >/dev/null 2>&1; then
    { lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true; } |
      sort -nu
    return
  fi

  if command -v ss >/dev/null 2>&1; then
    { ss -H -ltnp "sport = :$PORT" 2>/dev/null || true; } |
      grep -oE 'pid=[0-9]+' |
      cut -d= -f2 |
      sort -nu || true
    return
  fi

  if command -v fuser >/dev/null 2>&1; then
    { fuser -n tcp "$PORT" 2>/dev/null || true; } |
      tr ' ' '\n' |
      awk '/^[0-9]+$/' |
      sort -nu
    return
  fi

  echo "ERROR: lsof, ss, or fuser is required to inspect port $PORT." >&2
  return 1
}

is_vllm_server_process() {
  local pid=$1
  local command_line

  command_line=$(tr '\0' ' ' 2>/dev/null < "/proc/$pid/cmdline" || true)
  [[ "$command_line" == *"vllm.entrypoints.openai.api_server"* ]] ||
    [[ "$command_line" == *"vllm.entrypoints.cli.main"*" serve "* ]] ||
    [[ "$command_line" == *"vllm serve "* ]] ||
    [[ "$command_line" == *"VLLM::APIServer"* ]]
}

list_process_tree_pids() {
  local parent=$1
  local child

  printf '%s\n' "$parent"
  while IFS= read -r child; do
    [[ -n "$child" ]] || continue
    list_process_tree_pids "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
}

process_group_has_live_members() {
  local target_pgid=$1

  ps -e -o pgid=,stat= | awk -v target="$target_pgid" '
    $1 == target && $2 !~ /^Z/ { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

process_is_live() {
  local pid=$1
  local state

  state=$(ps -o stat= -p "$pid" 2>/dev/null || true)
  [[ -n "$state" && "$state" != Z* ]]
}

stop_existing_server() {
  local current_pgid
  local pid
  local pgid
  local waited
  local tree_pid
  local targets_running
  local -a listener_pids=()
  local -a remaining=()
  local -A target_groups=()
  local -A target_pids=()

  mapfile -t listener_pids < <(list_port_listener_pids)
  if (( ${#listener_pids[@]} == 0 )); then
    if command -v ss >/dev/null 2>&1 &&
        ss -H -ltn "sport = :$PORT" 2>/dev/null | grep -q .; then
      echo "ERROR: port $PORT is occupied, but its listener PID is not visible." >&2
      return 1
    fi
    echo "No existing service is listening on port $PORT."
    return
  fi

  # Validate every listener before sending any signal. A non-vLLM listener is
  # treated as a port conflict instead of being killed.
  for pid in "${listener_pids[@]}"; do
    if [[ ! -e "/proc/$pid" ]]; then
      continue
    fi
    if [[ ! -r "/proc/$pid/cmdline" ]]; then
      echo "ERROR: cannot inspect the process listening on port $PORT (PID $pid)." >&2
      return 1
    fi
    if ! is_vllm_server_process "$pid"; then
      echo "ERROR: port $PORT is owned by a non-vLLM process; refusing to stop it." >&2
      ps -ww -o pid,ppid,pgid,args -p "$pid" >&2 || true
      return 1
    fi
  done

  current_pgid=$(ps -o pgid= -p $$ | tr -d '[:space:]')
  for pid in "${listener_pids[@]}"; do
    [[ -r "/proc/$pid/stat" ]] || continue
    pgid=$(ps -o pgid= -p "$pid" | tr -d '[:space:]')
    if [[ ! "$pgid" =~ ^[0-9]+$ ]] || (( pgid <= 1 )); then
      echo "ERROR: could not determine a safe process group for PID $pid." >&2
      return 1
    fi

    if [[ "$pgid" == "$current_pgid" ]]; then
      # This can happen when a server was backgrounded from the same shell.
      # Signal its process tree so the benchmark runner does not kill itself.
      while IFS= read -r tree_pid; do
        [[ -n "$tree_pid" ]] && target_pids[$tree_pid]=1
      done < <(list_process_tree_pids "$pid")
    else
      target_groups[$pgid]=1
    fi
    echo "Found existing vLLM service: PID $pid, process group $pgid, port $PORT."
  done

  echo "Stopping the existing vLLM service..."
  for pgid in "${!target_groups[@]}"; do
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done
  for pid in "${!target_pids[@]}"; do
    kill -TERM -- "$pid" 2>/dev/null || true
  done

  for (( waited = 0; waited < SERVER_STOP_TIMEOUT; waited++ )); do
    mapfile -t remaining < <(list_port_listener_pids)
    targets_running=0
    for pgid in "${!target_groups[@]}"; do
      if process_group_has_live_members "$pgid"; then
        targets_running=1
        break
      fi
    done
    if (( ! targets_running )); then
      for pid in "${!target_pids[@]}"; do
        if process_is_live "$pid"; then
          targets_running=1
          break
        fi
      done
    fi
    if (( ${#remaining[@]} == 0 && ! targets_running )); then
      echo "Existing vLLM service stopped; port $PORT is free."
      return
    fi
    sleep 1
  done

  echo "Existing service did not stop within ${SERVER_STOP_TIMEOUT}s; sending SIGKILL."
  for pgid in "${!target_groups[@]}"; do
    kill -KILL -- "-$pgid" 2>/dev/null || true
  done
  for pid in "${!target_pids[@]}"; do
    kill -KILL -- "$pid" 2>/dev/null || true
  done

  for (( waited = 0; waited < 10; waited++ )); do
    mapfile -t remaining < <(list_port_listener_pids)
    targets_running=0
    for pgid in "${!target_groups[@]}"; do
      if process_group_has_live_members "$pgid"; then
        targets_running=1
        break
      fi
    done
    if (( ! targets_running )); then
      for pid in "${!target_pids[@]}"; do
        if process_is_live "$pid"; then
          targets_running=1
          break
        fi
      done
    fi
    if (( ${#remaining[@]} == 0 && ! targets_running )); then
      echo "Existing vLLM service was force-stopped; port $PORT is free."
      return
    fi
    sleep 1
  done

  echo "ERROR: port $PORT is still occupied after stopping the existing service." >&2
  return 1
}

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$PROJECT_ROOT/runs/$timestamp"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/job.log") 2>&1

echo "Daily TPU benchmark started at $(date -u --iso-8601=seconds)"
echo "Project root: $PROJECT_ROOT"
echo "Run directory: $RUN_DIR"
echo "Machine IP: $MACHINE_IP"

if (( ! PREPARE_ONLY )); then
  stop_existing_server
fi

"$SCRIPT_DIR/update_environment.sh"

source_revision=$(git -C "$TORCHTPU_DIR" rev-parse HEAD)
torch_tpu_version=$(
  "$VENV_DIR/bin/python" -c \
    'from importlib.metadata import version; print(version("torch-tpu"))'
)
model_revision=$(python3.12 -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["revision"])' \
  "$MODEL_DIR/SOURCE.json")
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
  "started_at": "$(date -u --iso-8601=seconds)",
  "machine_ip": "$MACHINE_IP",
  "torchtpu_vllm_revision": "$source_revision",
  "torch_tpu_version": "$torch_tpu_version",
  "torch_tpu_install_source": "pip",
  "model_directory": "$MODEL_DIR",
  "model_revision": "$model_revision",
  "model_load_format": "dummy",
  "port": $PORT
}
EOF

if (( PREPARE_ONLY )); then
  echo "Preparation completed; TPU server was not started."
  exit 0
fi

if ! command -v setsid >/dev/null 2>&1; then
  echo "ERROR: setsid is required for reliable server cleanup." >&2
  exit 1
fi

SERVER_PID=""
RUN_SUCCEEDED=0

stop_server() {
  if [[ -z "$SERVER_PID" ]]; then
    return
  fi
  if (( KEEP_SERVER_RUNNING && RUN_SUCCEEDED )); then
    echo "Keeping server process group $SERVER_PID running."
    return
  fi

  echo "Stopping server process group $SERVER_PID..."
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  for _ in {1..30}; do
    if ! kill -0 -- "-$SERVER_PID" 2>/dev/null; then
      wait "$SERVER_PID" 2>/dev/null || true
      echo "Server stopped."
      return
    fi
    sleep 1
  done

  echo "Server did not stop after 30 seconds; sending SIGKILL."
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}

on_signal() {
  echo "Received termination signal."
  exit 130
}

trap stop_server EXIT
trap on_signal INT TERM

echo "Starting dummy-weight inference server..."
setsid "$SCRIPT_DIR/run.sh" > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$RUN_DIR/server.pid"

ready=0
for (( waited = 0; waited < SERVER_READY_TIMEOUT; waited += 2 )); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server exited during startup." >&2
    tail -n 200 "$RUN_DIR/server.log" >&2
    exit 1
  fi
  if (( waited > 0 && waited % 60 == 0 )); then
    echo "Waiting for server... ${waited}s elapsed"
  fi
  sleep 2
done

if (( ! ready )); then
  echo "ERROR: server did not become healthy within ${SERVER_READY_TIMEOUT}s." >&2
  tail -n 200 "$RUN_DIR/server.log" >&2
  exit 1
fi
echo "Server is healthy on port $PORT."

"$VENV_DIR/bin/python" "$SCRIPT_DIR/bench_decode_sliding_window.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --model Qwen3.5-397B-A17B-FP8 \
  --output-dir "$RUN_DIR/results/decode_sliding_window" \
  --concurrency 16 \
  --prefill-tokens 65536 \
  --decode-tokens 1024 \
  --tokenizer-dir "$MODEL_DIR" \
  --rounds 3 \
  --window-seconds 10 \
  --step-seconds 1 \
  2>&1 | tee "$RUN_DIR/decode_benchmark.log"

"$SCRIPT_DIR/bench_all.sh" "$RUN_DIR" 2>&1 | tee "$RUN_DIR/benchmark.log"

curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null
RUN_SUCCEEDED=1
echo "Daily TPU benchmark completed successfully at $(date -u --iso-8601=seconds)"
