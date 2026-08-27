#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/launcher_env.sh"

TEST_ONLY="${TEST_ONLY:-0}"
extra_vllm_args=()
while (( $# > 0 )); do
  case "$1" in
    --test-only)
      TEST_ONLY=1
      ;;
    --)
      shift
      extra_vllm_args=("$@")
      break
      ;;
    *)
      echo "ERROR: unknown script argument '$1'; put vLLM arguments after --." >&2
      exit 2
      ;;
  esac
  shift
done

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
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18100}"
SERVICE_ID=tpu-daily-dp4-tp2-decode-c256
ROLE=decode
VLLM_SERVICE_LAUNCH="${VLLM_SERVICE_LAUNCH:-$PROJECT_ROOT/vendor/vllm-service-launch/bin/vllm-service-launch}"
VLLM_SERVICE_STATE_ROOT="${VLLM_SERVICE_STATE_ROOT:-$PROJECT_ROOT/.state/vllm-service-launch}"
VLLM_SERVICE_TARGET_ROOT="${VLLM_SERVICE_TARGET_ROOT:-/run/vllm-metrics-targets/targets}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-66560}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
COMPILE_SIZES="${COMPILE_SIZES:-8,16,32,64,72,4096}"
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-bfloat16}"
TPU_GDN_BF16_STATE_IO_MODE="${TPU_GDN_BF16_STATE_IO_MODE:-u32_128_fused_global_halves_initialized_decode}"
RAGGED_GATHER_REDUCE_VERSION="${RAGGED_GATHER_REDUCE_VERSION:-v2}"
TPU_MOE_OWNER_OUTPUT_MODE="${TPU_MOE_OWNER_OUTPUT_MODE:-on}"
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

for value_name in \
    PORT \
    MAX_MODEL_LEN \
    MAX_NUM_BATCHED_TOKENS \
    MAX_NUM_SEQS \
    BLOCK_SIZE \
    TPU_PREMAPPED_BUFFER_SIZE \
    VLLM_ENGINE_READY_TIMEOUT_S; do
  require_uint "$value_name" "${!value_name}"
done
if [[ ! "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]]; then
  echo "ERROR: GPU_MEMORY_UTILIZATION must be between 0 and 1." >&2
  exit 2
fi
if [[ "$RESET_COMPILE_CACHE" != 0 && "$RESET_COMPILE_CACHE" != 1 ]]; then
  echo "ERROR: RESET_COMPILE_CACHE must be 0 or 1." >&2
  exit 2
fi
case "$RAGGED_GATHER_REDUCE_VERSION" in
  v1|v2|v3) ;;
  *)
    echo "ERROR: RAGGED_GATHER_REDUCE_VERSION must be v1, v2, or v3, got '$RAGGED_GATHER_REDUCE_VERSION'." >&2
    exit 2
    ;;
esac
case "$TPU_MOE_OWNER_OUTPUT_MODE" in
  off|on) ;;
  *)
    echo "ERROR: TPU_MOE_OWNER_OUTPUT_MODE must be off or on, got '$TPU_MOE_OWNER_OUTPUT_MODE'." >&2
    exit 2
    ;;
esac
case "$MAMBA_SSM_CACHE_DTYPE" in
  bfloat16|float32) ;;
  *)
    echo "ERROR: MAMBA_SSM_CACHE_DTYPE must be bfloat16 or float32, got '$MAMBA_SSM_CACHE_DTYPE'." >&2
    exit 2
    ;;
esac
case "$TPU_GDN_BF16_STATE_IO_MODE" in
  legacy|u32_128_fused_global_halves_initialized_decode) ;;
  *)
    echo "ERROR: TPU_GDN_BF16_STATE_IO_MODE has unsupported value '$TPU_GDN_BF16_STATE_IO_MODE'." >&2
    exit 2
    ;;
esac
if [[ "$TPU_GDN_BF16_STATE_IO_MODE" != legacy && "$MAMBA_SSM_CACHE_DTYPE" != bfloat16 ]]; then
  echo "ERROR: optimized GDN state I/O requires MAMBA_SSM_CACHE_DTYPE=bfloat16." >&2
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

if [[ "$TEST_ONLY" != 0 && "$TEST_ONLY" != 1 ]]; then
  echo "ERROR: TEST_ONLY must be 0 or 1." >&2
  exit 2
fi
if (( TEST_ONLY )); then
  echo "TEST_ONLY: DP4/TP2 decode server startup skipped."
  printf 'extra vLLM args:'
  if (( ${#extra_vllm_args[@]} > 0 )); then
    printf ' %q' "${extra_vllm_args[@]}"
  fi
  printf '\n'
  exit 0
fi

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
if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "ERROR: local model weights are incomplete: $MODEL_DIR" >&2
  exit 1
fi

SOURCE_REV=$(git -C "$TORCHTPU_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
TORCH_TPU_VERSION=$(
  "$VENV_DIR/bin/python" -c \
    'from importlib.metadata import version; print(version("torch-tpu"))'
)
COMPILE_SIZES_CACHE_KEY=${COMPILE_SIZES//,/-}
CACHE_KEY="${SOURCE_REV}_torch_tpu${TORCH_TPU_VERSION}_c256_dp4_tp2"
CACHE_KEY+="_mml${MAX_MODEL_LEN}_mnbt${MAX_NUM_BATCHED_TOKENS}"
CACHE_KEY+="_mns${MAX_NUM_SEQS}_bs${BLOCK_SIZE}_gmu${GPU_MEMORY_UTILIZATION}"
CACHE_KEY+="_ssm${MAMBA_SSM_CACHE_DTYPE}"
CACHE_KEY+="_gdnio${TPU_GDN_BF16_STATE_IO_MODE}"
CACHE_KEY+="_rpalongctx_seq_lane_owner${TPU_MOE_OWNER_OUTPUT_MODE}_noprefix"
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
export TPU_PREMAPPED_BUFFER_SIZE
export RAY_memory_monitor_refresh_ms=0

export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
unset TPU_VLLM_KV_CACHE_ALIAS_FALLBACK
export TPU_KV_CACHE_HEADROOM_MIB=6144
unset USE_BATCHED_RPA_KERNEL
export USE_BATCHED_RPA_LONGCTX=1
export USE_BATCHED_RPA_SEQ_ON_LANE=1
export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd
export TPU_GDN_BF16_STATE_IO_MODE

export USE_MOE_SPARSE_CORE=1
export RAGGED_GATHER_VERSION=v2
export RAGGED_GATHER_REDUCE_VERSION
export TPU_MOE_OWNER_OUTPUT_MODE
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
echo "benchmark config:        dp4_tp2_decode_c256"
echo "parallelism:             TP=2, DP=4, EP=8"
echo "KV block size:           $BLOCK_SIZE (explicit)"
echo "GDN SSM cache dtype:     $MAMBA_SSM_CACHE_DTYPE"
echo "GDN BF16 state I/O:      $TPU_GDN_BF16_STATE_IO_MODE"
echo "prefix caching:          disabled"
echo "batched RPA:             longctx + seq_on_lane"
echo "ragged gather-reduce:    $RAGGED_GATHER_REDUCE_VERSION"
echo "MoE owner-output:        $TPU_MOE_OWNER_OUTPUT_MODE"
echo "compile sizes:           $COMPILE_SIZES"
echo "parallel precompile:     $TPU_PARALLEL_PRECOMPILE"
echo "premapped buffer size:   $TPU_PREMAPPED_BUFFER_SIZE"
echo "compile cache:           $COMPILE_CACHE_ROOT ($COMPILE_CACHE_ACTION before startup)"
echo "legacy TorchInductor cache: $LEGACY_TORCHINDUCTOR_CACHE (cleared before startup)"
echo "runtime temporary path:  $RUNTIME_TMP_ROOT (cleared before startup)"
echo "TorchTPU Tier-2 cache:   disabled (no /dev/shm dependency)"
echo "unified block pool:      enabled (block size floor $BLOCK_SIZE)"

ensure_uv_on_path
ensure_vllm_service_launcher "$VLLM_SERVICE_LAUNCH"

exec "$VLLM_SERVICE_LAUNCH" start \
  --state-root "$VLLM_SERVICE_STATE_ROOT" \
  --target-root "$VLLM_SERVICE_TARGET_ROOT" \
  --service-id "$SERVICE_ID" \
  --role "$ROLE" \
  --model-alias "$SERVED_MODEL_NAME" \
  --uv-project "$PROJECT_ROOT" \
  --working-directory "$PROJECT_ROOT" \
  --host "$HOST" \
  --port "$PORT" \
  -- serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --seed 42 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --data-parallel-size-local 4 \
  --enable-expert-parallel \
  --language-model-only \
  --mamba-cache-mode none \
  --mamba-ssm-cache-dtype "$MAMBA_SSM_CACHE_DTYPE" \
  --no-disable-hybrid-kv-cache-manager \
  --kv-cache-dtype fp8 \
  --block-size "$BLOCK_SIZE" \
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
  "${extra_vllm_args[@]}"
