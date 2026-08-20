#!/usr/bin/env bash

ensure_uv_on_path() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    PATH="$HOME/.local/bin:$PATH"
    export PATH
    return
  fi
  echo "ERROR: uv is required by vllm-service-launch." >&2
  return 1
}

ensure_vllm_service_launcher() {
  local launcher=$1
  if [[ -x "$launcher" ]]; then
    return
  fi
  echo "ERROR: vllm-service-launch is not executable: $launcher" >&2
  return 1
}
