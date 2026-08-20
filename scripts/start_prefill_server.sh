#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/launcher_env.sh"

BENCHMARK_CONFIG=
TEST_ONLY="${TEST_ONLY:-0}"

server_args=()
while (( $# > 0 )); do
  case "$1" in
    --config)
      if (( $# < 2 )); then
        echo "ERROR: --config requires dp8 or pcp8." >&2
        exit 2
      fi
      if [[ -n "$BENCHMARK_CONFIG" && "$BENCHMARK_CONFIG" != "$2" ]]; then
        echo "ERROR: conflicting benchmark configs were specified." >&2
        exit 2
      fi
      BENCHMARK_CONFIG=$2
      shift
      ;;
    --config=*)
      requested_config=${1#*=}
      if [[ -n "$BENCHMARK_CONFIG" && "$BENCHMARK_CONFIG" != "$requested_config" ]]; then
        echo "ERROR: conflicting benchmark configs were specified." >&2
        exit 2
      fi
      BENCHMARK_CONFIG=$requested_config
      ;;
    --test-only)
      TEST_ONLY=1
      ;;
    *)
      server_args+=("$1")
      ;;
  esac
  shift
done

case "$BENCHMARK_CONFIG" in
  dp8)
    CONFIG_LABEL=DP8
    SERVICE_ID=tpu-daily-dp8-prefill
    DATA_PARALLEL_SIZE=8
    PREFILL_CONTEXT_PARALLEL_SIZE=1
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    COMPILE_SIZES="${COMPILE_SIZES:-512,1024,2048,4096}"
    LONG_PREFILL_TOKEN_THRESHOLD=
    TPU_MOE_SKIP_PADDED_TOKENS="${TPU_MOE_SKIP_PADDED_TOKENS:-1}"
    ;;
  pcp8)
    CONFIG_LABEL=PCP8
    SERVICE_ID=tpu-daily-pcp8-prefill
    DATA_PARALLEL_SIZE=1
    PREFILL_CONTEXT_PARALLEL_SIZE=8
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    COMPILE_SIZES="${COMPILE_SIZES:-512,1024,2048,4096}"
    LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-32768}"
    TPU_MOE_SKIP_PADDED_TOKENS="${TPU_MOE_SKIP_PADDED_TOKENS:-0}"
    ;;
  *)
    echo "ERROR: --config must be dp8 or pcp8, got '$BENCHMARK_CONFIG'." >&2
    exit 2
    ;;
esac

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TORCHTPU_DIR="${TORCHTPU_DIR:-$PROJECT_ROOT/third_party/torchtpu-vllm}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-397B-A17B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18100}"
ROLE=prefill
LAUNCH_ENV_FILE="${LAUNCH_ENV_FILE:-$PROJECT_ROOT/.state/launcher/$SERVICE_ID.env}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
RESET_COMPILE_CACHE="${RESET_COMPILE_CACHE:-1}"
TPU_PARALLEL_PRECOMPILE="${TPU_PARALLEL_PRECOMPILE:-1}"
TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-17179869184}"

require_uint() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value == 0 )); then
    echo "ERROR: $name must be a positive integer, got '$value'." >&2
    exit 2
  fi
}

require_uint PORT "$PORT"
require_uint MAX_MODEL_LEN "$MAX_MODEL_LEN"
require_uint MAX_NUM_BATCHED_TOKENS "$MAX_NUM_BATCHED_TOKENS"
require_uint MAX_NUM_SEQS "$MAX_NUM_SEQS"
require_uint TPU_PREMAPPED_BUFFER_SIZE "$TPU_PREMAPPED_BUFFER_SIZE"
if [[ -n "$LONG_PREFILL_TOKEN_THRESHOLD" ]]; then
  require_uint LONG_PREFILL_TOKEN_THRESHOLD "$LONG_PREFILL_TOKEN_THRESHOLD"
fi
require_uint VLLM_ENGINE_READY_TIMEOUT_S "$VLLM_ENGINE_READY_TIMEOUT_S"
if [[ "$TEST_ONLY" != 0 && "$TEST_ONLY" != 1 ]]; then
  echo "ERROR: TEST_ONLY must be 0 or 1." >&2
  exit 2
fi
if [[ "$RESET_COMPILE_CACHE" != 0 && "$RESET_COMPILE_CACHE" != 1 ]]; then
  echo "ERROR: RESET_COMPILE_CACHE must be 0 or 1." >&2
  exit 2
fi
case "${TPU_MOE_SKIP_PADDED_TOKENS,,}" in
  1|true) TPU_MOE_SKIP_PADDED_TOKENS=1 ;;
  0|false) TPU_MOE_SKIP_PADDED_TOKENS=0 ;;
  *)
    echo "ERROR: TPU_MOE_SKIP_PADDED_TOKENS must be a boolean, got '$TPU_MOE_SKIP_PADDED_TOKENS'." >&2
    exit 2
    ;;
esac
case "${TPU_PARALLEL_PRECOMPILE,,}" in
  1|true) TPU_PARALLEL_PRECOMPILE=1 ;;
  0|false) TPU_PARALLEL_PRECOMPILE=0 ;;
  *)
    echo "ERROR: TPU_PARALLEL_PRECOMPILE must be a boolean, got '$TPU_PARALLEL_PRECOMPILE'." >&2
    exit 2
    ;;
esac

if (( TEST_ONLY )); then
  echo "TEST_ONLY: $CONFIG_LABEL server startup skipped."
  echo "parallelism:             DP=$DATA_PARALLEL_SIZE, PCP=$PREFILL_CONTEXT_PARALLEL_SIZE, TP=1"
  echo "max model length:        $MAX_MODEL_LEN"
  echo "max batched tokens:      $MAX_NUM_BATCHED_TOKENS"
  echo "max sequences:           $MAX_NUM_SEQS"
  echo "compile sizes:           $COMPILE_SIZES"
  echo "parallel precompile:     $TPU_PARALLEL_PRECOMPILE"
  echo "premapped buffer size:   $TPU_PREMAPPED_BUFFER_SIZE"
  echo "skip padded MoE tokens:  $TPU_MOE_SKIP_PADDED_TOKENS"
  if [[ -n "$LONG_PREFILL_TOKEN_THRESHOLD" ]]; then
    echo "long prefill threshold:  $LONG_PREFILL_TOKEN_THRESHOLD"
  fi
  exit 0
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: project environment is missing: $VENV_DIR" >&2
  echo "Run scripts/update_environment.sh first." >&2
  exit 1
fi
if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
  echo "ERROR: vLLM is not installed in the project environment: $VENV_DIR" >&2
  echo "Run scripts/update_environment.sh first." >&2
  exit 1
fi
if [[ ! -d "$TORCHTPU_DIR/src/vllm_torchtpu" ]]; then
  echo "ERROR: vllm-torchtpu submodule is missing: $TORCHTPU_DIR" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/tokenizer.json" ]]; then
  echo "ERROR: local model metadata is incomplete: $MODEL_DIR" >&2
  exit 1
fi

SOURCE_REV=$(git -C "$TORCHTPU_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
TORCH_TPU_VERSION=$(
  "$VENV_DIR/bin/python" -c \
    'from importlib.metadata import version; print(version("torch-tpu"))'
)
COMPILE_SIZES_CACHE_KEY=${COMPILE_SIZES//,/-}
case "$BENCHMARK_CONFIG" in
  dp8)
    CACHE_KEY="${SOURCE_REV}_torch_tpu${TORCH_TPU_VERSION}_dp8_tp1_mml${MAX_MODEL_LEN}_mnbt${MAX_NUM_BATCHED_TOKENS}_mns${MAX_NUM_SEQS}_moeskip${TPU_MOE_SKIP_PADDED_TOKENS}_cs${COMPILE_SIZES_CACHE_KEY}"
    ;;
  pcp8)
    CACHE_KEY="${SOURCE_REV}_torch_tpu${TORCH_TPU_VERSION}_dp1_pcp8_mml${MAX_MODEL_LEN}_mnbt${MAX_NUM_BATCHED_TOKENS}_mns${MAX_NUM_SEQS}_lptt${LONG_PREFILL_TOKEN_THRESHOLD}_moeskip${TPU_MOE_SKIP_PADDED_TOKENS}_cs${COMPILE_SIZES_CACHE_KEY}"
    ;;
esac

export PJRT_DEVICE=TPU
export VLLM_TARGET_DEVICE=tpu
export VLLM_PLUGINS=torchtpu
export PYTHONPATH="$TORCHTPU_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
# TorchTPU's split compiler artifact is not currently serializable. Disable
# both vLLM's compile cache and PyTorch's AOTAutograd cache until upstream can
# persist _SplitCompiledExecutable safely.
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-0}"
export TPU_PARALLEL_PRECOMPILE
export TPU_PREMAPPED_BUFFER_SIZE
export RAY_memory_monitor_refresh_ms=0
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG=1
export RAGGED_GATHER_REDUCE_VERSION=v3
export TPU_MOE_SKIP_PADDED_TOKENS
export VLLM_ENGINE_READY_TIMEOUT_S

# Keep every configurable compilation cache on project storage. By default the
# shared root is cleared before startup to avoid unbounded growth; set
# RESET_COMPILE_CACHE=0 for deliberate reuse while validating a fixed revision.
# Also remove the legacy TorchInductor cache under /tmp, where older launches
# may have left large AOTAutograd artifacts.
LEGACY_TORCHINDUCTOR_CACHE="/tmp/torchinductor_$(id -un)"
case "$LEGACY_TORCHINDUCTOR_CACHE" in
  /tmp/torchinductor_*) ;;
  *)
    echo "ERROR: unsafe legacy TorchInductor cache path: $LEGACY_TORCHINDUCTOR_CACHE" >&2
    exit 1
    ;;
esac
if [[ -L "$LEGACY_TORCHINDUCTOR_CACHE" ]] ||
  [[ -e "$LEGACY_TORCHINDUCTOR_CACHE" && ! -d "$LEGACY_TORCHINDUCTOR_CACHE" ]]; then
  echo "ERROR: legacy TorchInductor cache path must be a real directory: $LEGACY_TORCHINDUCTOR_CACHE" >&2
  exit 1
fi
if [[ -d "$LEGACY_TORCHINDUCTOR_CACHE" ]]; then
  if [[ "$(stat -c '%u' "$LEGACY_TORCHINDUCTOR_CACHE")" != "$(id -u)" ]]; then
    echo "ERROR: legacy TorchInductor cache is not owned by the current user: $LEGACY_TORCHINDUCTOR_CACHE" >&2
    exit 1
  fi
  find "$LEGACY_TORCHINDUCTOR_CACHE" -mindepth 1 -delete
fi

# TorchTPU's Tier-2 root is fixed under /dev/shm and cannot be redirected to an
# absolute path, so disable Tier-2 explicitly to make startup independent of
# /dev/shm.
COMPILE_CACHE_ROOT="$PROJECT_ROOT/cache/compile"
case "$COMPILE_CACHE_ROOT" in
  "$PROJECT_ROOT"/cache/*) ;;
  *)
    echo "ERROR: unsafe compilation cache path: $COMPILE_CACHE_ROOT" >&2
    exit 1
    ;;
esac
if [[ -L "$COMPILE_CACHE_ROOT" ]] ||
  [[ -e "$COMPILE_CACHE_ROOT" && ! -d "$COMPILE_CACHE_ROOT" ]]; then
  echo "ERROR: compilation cache path must be a real directory: $COMPILE_CACHE_ROOT" >&2
  exit 1
fi
mkdir -p "$COMPILE_CACHE_ROOT"
if (( RESET_COMPILE_CACHE )); then
  find "$COMPILE_CACHE_ROOT" -mindepth 1 -delete
  COMPILE_CACHE_ACTION=cleared
else
  COMPILE_CACHE_ACTION=retained
fi

# Keep runtime temporary files on the project's data filesystem as well. This
# path intentionally lives at the mount root so ZMQ's IPC endpoint remains
# below its 107-character Unix socket limit.
DATA_MOUNT="$(findmnt -n -o TARGET --target "$PROJECT_ROOT")"
if [[ -z "$DATA_MOUNT" || "$DATA_MOUNT" == "/" || "$DATA_MOUNT" == "/dev/shm" ]]; then
  echo "ERROR: project must be on a dedicated non-/dev/shm filesystem: $PROJECT_ROOT" >&2
  exit 1
fi
RUNTIME_TMP_ROOT="$DATA_MOUNT/.tbd-$(id -u)"
case "$RUNTIME_TMP_ROOT" in
  "$DATA_MOUNT"/.tbd-*) ;;
  *)
    echo "ERROR: unsafe runtime temporary path: $RUNTIME_TMP_ROOT" >&2
    exit 1
    ;;
esac
if [[ -L "$RUNTIME_TMP_ROOT" ]] ||
  [[ -e "$RUNTIME_TMP_ROOT" && ! -d "$RUNTIME_TMP_ROOT" ]]; then
  echo "ERROR: runtime temporary path must be a real directory: $RUNTIME_TMP_ROOT" >&2
  exit 1
fi
mkdir -p "$RUNTIME_TMP_ROOT"
if [[ "$(stat -c '%u' "$RUNTIME_TMP_ROOT")" != "$(id -u)" ]]; then
  echo "ERROR: runtime temporary path is not owned by the current user: $RUNTIME_TMP_ROOT" >&2
  exit 1
fi
find "$RUNTIME_TMP_ROOT" -mindepth 1 -delete
if (( ${#RUNTIME_TMP_ROOT} > 64 )); then
  echo "ERROR: runtime temporary path is too long for ZMQ IPC: $RUNTIME_TMP_ROOT" >&2
  exit 1
fi

export VLLM_CACHE_ROOT="$COMPILE_CACHE_ROOT/vllm/$CACHE_KEY"
export VLLM_XLA_CACHE_PATH="$COMPILE_CACHE_ROOT/xla/$CACHE_KEY"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=disabled
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="$VLLM_XLA_CACHE_PATH/torch_tpu_tier3"
export TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE_ROOT/torchinductor/$CACHE_KEY"
export XDG_CACHE_HOME="$COMPILE_CACHE_ROOT/xdg/$CACHE_KEY"
export TMPDIR="$RUNTIME_TMP_ROOT"
export TMP="$RUNTIME_TMP_ROOT"
export TEMP="$RUNTIME_TMP_ROOT"
export VLLM_XLA_CHECK_RECOMPILATION=0
export PYTHONUNBUFFERED=1

mkdir -p \
  "$VLLM_CACHE_ROOT" \
  "$VLLM_XLA_CACHE_PATH" \
  "$TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME"

PROFILE_DIR="${PROFILE_DIR:-${VLLM_TORCH_PROFILER_DIR:-}}"
PROFILE_DELAY_ITERATIONS="${PROFILE_DELAY_ITERATIONS:-0}"
PROFILE_MAX_ITERATIONS="${PROFILE_MAX_ITERATIONS:-0}"

profile_args=()
profile_env_keys=()
if [[ -n "$PROFILE_DIR" ]]; then
  mkdir -p "$PROFILE_DIR"
  export VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
  profile_env_keys+=(VLLM_TORCH_PROFILER_DIR)
  profile_args=(
    --profiler-config.profiler torch
    --profiler-config.torch_profiler_dir "$VLLM_TORCH_PROFILER_DIR"
    --profiler-config.ignore_frontend true
    --profiler-config.delay_iterations "$PROFILE_DELAY_ITERATIONS"
    --profiler-config.max_iterations "$PROFILE_MAX_ITERATIONS"
  )
fi

COMPILATION_CONFIG=$(printf \
  '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[%s],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}' \
  "$COMPILE_SIZES")

long_prefill_args=()
if [[ -n "$LONG_PREFILL_TOKEN_THRESHOLD" ]]; then
  long_prefill_args=(
    --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD"
  )
fi

echo "Starting $SERVED_MODEL_NAME from offline metadata at $MODEL_DIR"
echo "vllm-torchtpu revision: $SOURCE_REV"
echo "torch_tpu version:       $TORCH_TPU_VERSION"
echo "benchmark config:        $BENCHMARK_CONFIG"
echo "parallelism:             DP=$DATA_PARALLEL_SIZE, PCP=$PREFILL_CONTEXT_PARALLEL_SIZE, TP=1"
echo "compile sizes: $COMPILE_SIZES"
echo "parallel precompile: $TPU_PARALLEL_PRECOMPILE"
echo "premapped buffer size: $TPU_PREMAPPED_BUFFER_SIZE"
echo "compile cache: $COMPILE_CACHE_ROOT ($COMPILE_CACHE_ACTION before startup)"
echo "legacy TorchInductor cache: $LEGACY_TORCHINDUCTOR_CACHE (cleared before startup)"
echo "runtime temporary path: $RUNTIME_TMP_ROOT (cleared before startup)"
echo "TorchTPU Tier-2 cache: disabled (no /dev/shm dependency)"
echo "unified block pool: enabled (block size auto-derived)"
echo "skip padded MoE tokens: $TPU_MOE_SKIP_PADDED_TOKENS"

write_launcher_env "$LAUNCH_ENV_FILE" \
  PJRT_DEVICE \
  VLLM_TARGET_DEVICE \
  VLLM_PLUGINS \
  PYTHONPATH \
  HF_HUB_OFFLINE \
  TRANSFORMERS_OFFLINE \
  HF_DATASETS_OFFLINE \
  SKIP_JAX_PRECOMPILE \
  VLLM_DISABLE_COMPILE_CACHE \
  TORCHINDUCTOR_AUTOGRAD_CACHE \
  TPU_PARALLEL_PRECOMPILE \
  TPU_PREMAPPED_BUFFER_SIZE \
  RAY_memory_monitor_refresh_ms \
  TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL \
  TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG \
  RAGGED_GATHER_REDUCE_VERSION \
  TPU_MOE_SKIP_PADDED_TOKENS \
  VLLM_ENGINE_READY_TIMEOUT_S \
  VLLM_CACHE_ROOT \
  VLLM_XLA_CACHE_PATH \
  TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE \
  TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT \
  TORCHINDUCTOR_CACHE_DIR \
  XDG_CACHE_HOME \
  TMPDIR \
  TMP \
  TEMP \
  VLLM_XLA_CHECK_RECOMPILATION \
  PYTHONUNBUFFERED \
  "${profile_env_keys[@]}"

ensure_uv_on_path
ensure_vllm_service_launcher

exec vllm-service-launch start \
  --service-id "$SERVICE_ID" \
  --role "$ROLE" \
  --model-alias "$SERVED_MODEL_NAME" \
  --uv-project "$PROJECT_ROOT" \
  --env-file "$LAUNCH_ENV_FILE" \
  --working-directory "$PROJECT_ROOT" \
  --host "$HOST" \
  --port "$PORT" \
  -- serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --generation-config vllm \
  --seed 42 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  "${long_prefill_args[@]}" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size "$DATA_PARALLEL_SIZE" \
  --attention-backend CUSTOM \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --language-model-only \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --no-enable-prefix-caching \
  --prefill-context-parallel-size "$PREFILL_CONTEXT_PARALLEL_SIZE" \
  --cp-kv-cache-interleave-size 256 \
  --no-disable-hybrid-kv-cache-manager \
  --tensor-parallel-size 1 \
  --return-tokens-as-token-ids \
  --compilation-config "$COMPILATION_CONFIG" \
  "${profile_args[@]}" \
  "${server_args[@]}"
