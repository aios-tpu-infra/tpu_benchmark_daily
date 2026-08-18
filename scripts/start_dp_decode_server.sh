#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/launcher_env.sh"

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TORCHTPU_DIR="${TORCHTPU_DIR:-$PROJECT_ROOT/third_party/torchtpu-vllm}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-397B-A17B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18100}"
SERVICE_ID=tpu-daily-dp8-decode-c256
ROLE=decode
LAUNCH_ENV_FILE="${LAUNCH_ENV_FILE:-$PROJECT_ROOT/.state/launcher/$SERVICE_ID.env}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-66560}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.932285943}"
COMPILE_SIZES="${COMPILE_SIZES:-8,16,32,4352,4384}"
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
TPU_PARALLEL_PRECOMPILE="${TPU_PARALLEL_PRECOMPILE:-1}"

require_uint() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value == 0 )); then
    echo "ERROR: $name must be a positive integer, got '$value'." >&2
    exit 2
  fi
}

for value_name in \
    PORT \
    MAX_MODEL_LEN \
    MAX_NUM_BATCHED_TOKENS \
    MAX_NUM_SEQS \
    VLLM_ENGINE_READY_TIMEOUT_S; do
  require_uint "$value_name" "${!value_name}"
done
if [[ ! "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]]; then
  echo "ERROR: GPU_MEMORY_UTILIZATION must be between 0 and 1." >&2
  exit 2
fi
case "${TPU_PARALLEL_PRECOMPILE,,}" in
  1|true) TPU_PARALLEL_PRECOMPILE=1 ;;
  0|false) TPU_PARALLEL_PRECOMPILE=0 ;;
  *)
    echo "ERROR: TPU_PARALLEL_PRECOMPILE must be a boolean, got '$TPU_PARALLEL_PRECOMPILE'." >&2
    exit 2
    ;;
esac

if [[ ! -x "$VENV_DIR/bin/python" || ! -x "$VENV_DIR/bin/vllm" ]]; then
  echo "ERROR: project environment is incomplete: $VENV_DIR" >&2
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
CACHE_KEY="${SOURCE_REV}_torch_tpu${TORCH_TPU_VERSION}_c256_dp8_tp1"
CACHE_KEY+="_mml${MAX_MODEL_LEN}_mnbt${MAX_NUM_BATCHED_TOKENS}"
CACHE_KEY+="_mns${MAX_NUM_SEQS}_gmu${GPU_MEMORY_UTILIZATION}"
CACHE_KEY+="_cs${COMPILE_SIZES_CACHE_KEY}"

export PYTHONPATH="$TORCHTPU_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export JAX_PLATFORMS=tpu,cpu
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
unset TPU_MULTIHOST_BACKEND
export VLLM_TARGET_DEVICE=tpu
export VLLM_PLUGINS=torchtpu
export MODEL_IMPL_TYPE=vllm
export NEW_MODEL_DESIGN=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# The daily environment still requires these caches to be disabled because
# TorchTPU split compiler artifacts are not serializable.
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-0}"
export TPU_PARALLEL_PRECOMPILE
export RAY_memory_monitor_refresh_ms=0

export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
unset TPU_VLLM_KV_CACHE_ALIAS_FALLBACK
export TPU_KV_CACHE_HEADROOM_MIB=6144
export USE_BATCHED_RPA_KERNEL=1
export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd

export USE_MOE_SPARSE_CORE=1
export RAGGED_GATHER_VERSION=v2
export RAGGED_GATHER_REDUCE_VERSION=v2
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
unset TPU_RAGGED_GATHER_REDUCE_IMPL
unset TPU_RAGGED_GATHER_IMPL

export DP_SCHED_BATCH_PREFILL_MAX_ADMIT_PER_FLUSH=0
export VLLM_ENGINE_READY_TIMEOUT_S
export TORCH_TPU_DP_MASTER_ADDR="${TORCH_TPU_DP_MASTER_ADDR:-127.0.0.1}"
export TORCH_TPU_DP_MASTER_PORT="${TORCH_TPU_DP_MASTER_PORT:-29645}"
DEFAULT_LIBTPU_INIT_ARGS=" --xla_tpu_use_dynamic_smem_negotiation=true"
DEFAULT_LIBTPU_INIT_ARGS+=" --xla_tpu_scoped_vmem_limit_kib=65536"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-$DEFAULT_LIBTPU_INIT_ARGS}"
unset TPU_XPROF_DEVICE_COUNTERS
unset VLLM_TORCH_PROFILER_DIR

# Keep every configurable compilation cache on project storage and start each
# server with an empty cache to avoid unbounded growth. Also remove the legacy
# TorchInductor cache under /tmp, where older launches may have left large
# AOTAutograd artifacts.
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
find "$COMPILE_CACHE_ROOT" -mindepth 1 -delete

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
export PYTHONUNBUFFERED=1

mkdir -p \
  "$VLLM_CACHE_ROOT" \
  "$VLLM_XLA_CACHE_PATH" \
  "$TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME"

COMPILATION_CONFIG=$(printf \
  '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[%s],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}' \
  "$COMPILE_SIZES")

echo "Starting real-weight $SERVED_MODEL_NAME from $MODEL_DIR"
echo "vllm-torchtpu revision: $SOURCE_REV"
echo "torch_tpu version:       $TORCH_TPU_VERSION"
echo "benchmark config:        dp8_decode_c256"
echo "parallelism:             TP=1, DP=8, EP=8"
echo "compile sizes:           $COMPILE_SIZES"
echo "parallel precompile:     $TPU_PARALLEL_PRECOMPILE"
echo "compile cache:           $COMPILE_CACHE_ROOT (cleared before startup)"
echo "legacy TorchInductor cache: $LEGACY_TORCHINDUCTOR_CACHE (cleared before startup)"
echo "runtime temporary path:  $RUNTIME_TMP_ROOT (cleared before startup)"
echo "TorchTPU Tier-2 cache:   disabled (no /dev/shm dependency)"
echo "unified block pool:      enabled (block size auto-derived)"

write_launcher_env "$LAUNCH_ENV_FILE" \
  PYTHONPATH \
  HF_HUB_OFFLINE \
  TRANSFORMERS_OFFLINE \
  HF_DATASETS_OFFLINE \
  JAX_PLATFORMS \
  PJRT_DEVICE \
  TPU_BACKEND_TYPE \
  VLLM_TARGET_DEVICE \
  VLLM_PLUGINS \
  MODEL_IMPL_TYPE \
  NEW_MODEL_DESIGN \
  SKIP_JAX_PRECOMPILE \
  VLLM_XLA_CHECK_RECOMPILATION \
  XLA_PYTHON_CLIENT_PREALLOCATE \
  VLLM_ALLOW_LONG_MAX_MODEL_LEN \
  VLLM_DISABLE_COMPILE_CACHE \
  TORCHINDUCTOR_AUTOGRAD_CACHE \
  TPU_PARALLEL_PRECOMPILE \
  RAY_memory_monitor_refresh_ms \
  TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL \
  TPU_KV_CACHE_HEADROOM_MIB \
  USE_BATCHED_RPA_KERNEL \
  RAGGED_GATED_DELTA_RULE_IMPL \
  USE_MOE_SPARSE_CORE \
  RAGGED_GATHER_VERSION \
  RAGGED_GATHER_REDUCE_VERSION \
  ONEHOT_MOE_PERMUTE_THRESHOLD \
  DP_SCHED_BATCH_PREFILL_MAX_ADMIT_PER_FLUSH \
  VLLM_ENGINE_READY_TIMEOUT_S \
  TORCH_TPU_DP_MASTER_ADDR \
  TORCH_TPU_DP_MASTER_PORT \
  LIBTPU_INIT_ARGS \
  VLLM_CACHE_ROOT \
  VLLM_XLA_CACHE_PATH \
  TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE \
  TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT \
  TORCHINDUCTOR_CACHE_DIR \
  XDG_CACHE_HOME \
  TMPDIR \
  TMP \
  TEMP \
  PYTHONUNBUFFERED

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
  --trust-remote-code \
  --seed 42 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --data-parallel-size-local 8 \
  --enable-expert-parallel \
  --language-model-only \
  --mamba-cache-mode align \
  --no-disable-hybrid-kv-cache-manager \
  --kv-cache-dtype fp8 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --async-scheduling \
  --enable-prompt-tokens-details \
  --disable-log-stats \
  --no-enable-log-requests \
  --no-enable-prefix-caching \
  --attention-backend CUSTOM \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config "$COMPILATION_CONFIG" \
  "$@"
