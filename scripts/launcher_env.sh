#!/usr/bin/env bash

write_launcher_env() {
  local destination=$1
  shift

  local name
  local temporary
  local value

  mkdir -p "$(dirname -- "$destination")"
  temporary=$(mktemp "${destination}.tmp.XXXXXX")
  chmod 0600 "$temporary"
  for name in "$@"; do
    if [[ ! -v "$name" ]]; then
      echo "ERROR: launcher environment variable is unset: $name" >&2
      rm -f -- "$temporary"
      return 1
    fi
    value=${!name}
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
      echo "ERROR: launcher environment variable contains a newline: $name" >&2
      rm -f -- "$temporary"
      return 1
    fi
    printf '%s=%s\n' "$name" "$value" >> "$temporary"
  done
  mv -f -- "$temporary" "$destination"
}

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
