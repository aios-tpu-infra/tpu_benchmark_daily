#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TORCHTPU_DIR="${TORCHTPU_DIR:-$PROJECT_ROOT/third_party/torchtpu-vllm}"
MODEL_DIR="${MODEL_DIR:-/mnt/data/models/Qwen3.5-397B-A17B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-66560}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-4352}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
COMPILE_SIZES="${COMPILE_SIZES:-8,16,32,4352,4384}"

require_uint() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value == 0 )); then
    echo "ERROR: $name must be a positive integer, got '$value'." >&2
    exit 2
  fi
}

for value_name in \
  PORT MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS BLOCK_SIZE; do
  require_uint "$value_name" "${!value_name}"
done
if [[ ! "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]]; then
  echo "ERROR: GPU_MEMORY_UTILIZATION must be between 0 and 1." >&2
  exit 2
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
if [[ ! -f "$MODEL_DIR/config.json" ||
      ! -f "$MODEL_DIR/tokenizer.json" ||
      -z "$(find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' -print -quit)" ]]; then
  echo "ERROR: real model weights or metadata are incomplete: $MODEL_DIR" >&2
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
CACHE_KEY+="_mns${MAX_NUM_SEQS}_bs${BLOCK_SIZE}_gmu${GPU_MEMORY_UTILIZATION}"
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
export RAY_memory_monitor_refresh_ms=0

export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
unset TPU_VLLM_KV_CACHE_ALIAS_FALLBACK
export TPU_KV_CACHE_HEADROOM_MIB=6144
export USE_BATCHED_RPA_KERNEL=1
export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd

export USE_MOE_SPARSE_CORE=1
export RAGGED_GATHER_VERSION=v2
export RAGGED_GATHER_REDUCE_VERSION=v2
export ONEHOT_MOE_PERMUTE_THRESHOLD=0
unset TPU_RAGGED_GATHER_REDUCE_IMPL
unset TPU_RAGGED_GATHER_IMPL
unset VLLM_MOE_ROUTING_SIMULATION_STRATEGY

export DP_SCHED_BATCH_PREFILL_MAX_ADMIT_PER_FLUSH=0
export TORCH_TPU_DP_MASTER_ADDR="${TORCH_TPU_DP_MASTER_ADDR:-127.0.0.1}"
export TORCH_TPU_DP_MASTER_PORT="${TORCH_TPU_DP_MASTER_PORT:-29645}"
DEFAULT_LIBTPU_INIT_ARGS=" --xla_tpu_use_dynamic_smem_negotiation=true"
DEFAULT_LIBTPU_INIT_ARGS+=" --xla_tpu_scoped_vmem_limit_kib=65536"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-$DEFAULT_LIBTPU_INIT_ARGS}"
unset TPU_XPROF_DEVICE_COUNTERS
unset VLLM_TORCH_PROFILER_DIR

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PROJECT_ROOT/cache/vllm/$CACHE_KEY}"
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-$PROJECT_ROOT/cache/xla/$CACHE_KEY}"
DEFAULT_TORCHINDUCTOR_CACHE="$PROJECT_ROOT/cache/torchinductor/$CACHE_KEY"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$DEFAULT_TORCHINDUCTOR_CACHE}"
export TORCHINDUCTOR_CACHE_DIR
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/cache/xdg/$CACHE_KEY}"
export TMPDIR="${TMPDIR:-/tmp/tpu-benchmark-daily-c256}"
export PYTHONUNBUFFERED=1

mkdir -p \
  "$VLLM_CACHE_ROOT" \
  "$VLLM_XLA_CACHE_PATH" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$TMPDIR"

COMPILATION_CONFIG=$(printf \
  '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[%s],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}' \
  "$COMPILE_SIZES")

echo "Starting real-weight $SERVED_MODEL_NAME from $MODEL_DIR"
echo "vllm-torchtpu revision: $SOURCE_REV"
echo "torch_tpu version:       $TORCH_TPU_VERSION"
echo "benchmark config:        dp8_decode_c256"
echo "parallelism:             TP=1, DP=8, EP=8"
echo "compile sizes:           $COMPILE_SIZES"

exec "$VENV_DIR/bin/python" \
  -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --generation-config vllm \
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
  --block-size "$BLOCK_SIZE" \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --return-tokens-as-token-ids \
  --compilation-config "$COMPILATION_CONFIG" \
  "$@"
