#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TORCHTPU_DIR="${TORCHTPU_DIR:-$PROJECT_ROOT/third_party/torchtpu-vllm}"
TORCH_TPU_DIR="${TORCH_TPU_DIR:-$PROJECT_ROOT/third_party/torch_tpu}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-397B-A17B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-66560}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
COMPILE_SIZES="${COMPILE_SIZES:-2,4,8,16,$MAX_NUM_BATCHED_TOKENS}"

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
if [[ ! -f "$TORCH_TPU_DIR/pyproject.toml" ]]; then
  echo "ERROR: torch_tpu submodule is missing: $TORCH_TPU_DIR" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/tokenizer.json" ]]; then
  echo "ERROR: local model metadata is incomplete: $MODEL_DIR" >&2
  exit 1
fi

SOURCE_REV=$(git -C "$TORCHTPU_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
TORCH_TPU_REV=$(git -C "$TORCH_TPU_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
COMPILE_SIZES_CACHE_KEY=${COMPILE_SIZES//,/-}
CACHE_KEY="${SOURCE_REV}_${TORCH_TPU_REV}_dp8_tp1_mml${MAX_MODEL_LEN}_mnbt${MAX_NUM_BATCHED_TOKENS}_mns${MAX_NUM_SEQS}_cs${COMPILE_SIZES_CACHE_KEY}"

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
export RAY_memory_monitor_refresh_ms=0
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0
export TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PROJECT_ROOT/cache/vllm/$CACHE_KEY}"
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-$PROJECT_ROOT/cache/xla/$CACHE_KEY}"
export VLLM_XLA_CHECK_RECOMPILATION=0
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY="${VLLM_MOE_ROUTING_SIMULATION_STRATEGY:-uniform_random}"
export PYTHONUNBUFFERED=1

mkdir -p "$VLLM_CACHE_ROOT" "$VLLM_XLA_CACHE_PATH"

PROFILE_DIR="${PROFILE_DIR:-${VLLM_TORCH_PROFILER_DIR:-}}"
PROFILE_DELAY_ITERATIONS="${PROFILE_DELAY_ITERATIONS:-0}"
PROFILE_MAX_ITERATIONS="${PROFILE_MAX_ITERATIONS:-0}"

profile_args=()
if [[ -n "$PROFILE_DIR" ]]; then
  mkdir -p "$PROFILE_DIR"
  export VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
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

echo "Starting $SERVED_MODEL_NAME from offline metadata at $MODEL_DIR"
echo "vllm-torchtpu revision: $SOURCE_REV"
echo "torch_tpu revision:      $TORCH_TPU_REV"
echo "load format: dummy"
echo "compile sizes: $COMPILE_SIZES"

exec "$VENV_DIR/bin/python" \
  -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --load-format dummy \
  --generation-config vllm \
  --seed 42 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size 8 \
  --attention-backend CUSTOM \
  --block-size 256 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --language-model-only \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --no-enable-prefix-caching \
  --prefill-context-parallel-size 1 \
  --cp-kv-cache-interleave-size 256 \
  --no-disable-hybrid-kv-cache-manager \
  --tensor-parallel-size 1 \
  --return-tokens-as-token-ids \
  --compilation-config "$COMPILATION_CONFIG" \
  "${profile_args[@]}" \
  "$@"
