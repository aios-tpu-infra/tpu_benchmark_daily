#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
SOURCE_DIR="$PROJECT_ROOT/vendor/vllm-service-launch"
INSTALL_ROOT=/

usage() {
  cat <<'EOF'
Usage: scripts/install_vllm_service_launcher.sh [--root ABSOLUTE_PATH]

Installs the repository-owned vllm-service-launch CLI and Python library.

  --root ABSOLUTE_PATH  Stage files below an alternate root without invoking
                        privileged system changes. Intended for packaging and
                        tests.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --root)
      if (( $# < 2 )); then
        echo "ERROR: --root requires an absolute path." >&2
        usage >&2
        exit 2
      fi
      INSTALL_ROOT=$2
      shift
      ;;
    --root=*)
      INSTALL_ROOT=${1#*=}
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

if [[ "$INSTALL_ROOT" != /* ]]; then
  echo "ERROR: --root must be an absolute path." >&2
  exit 2
fi
if [[ "$INSTALL_ROOT" == / && ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "ERROR: system installation must run as root; use sudo." >&2
  exit 1
fi
if ! command -v python3.12 >/dev/null 2>&1; then
  echo "ERROR: python3.12 is required by vllm-service-launch." >&2
  exit 1
fi

destination() {
  local path=$1
  if [[ "$INSTALL_ROOT" == / ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "${INSTALL_ROOT%/}$path"
  fi
}

legacy_paths=(
  /etc/sudoers.d/vllm-service-launch
  /etc/systemd/system/vllm@.service
  /usr/lib/tmpfiles.d/vllm-metrics-targets.conf
)
for legacy_path in "${legacy_paths[@]}"; do
  if [[ -e "$(destination "$legacy_path")" ]]; then
    echo "ERROR: legacy launcher asset exists: $legacy_path" >&2
    echo "Remove the legacy systemd installation before retrying; see vendor/vllm-service-launch/README.md." >&2
    exit 1
  fi
done

install -D -m 0755 \
  "$SOURCE_DIR/bin/vllm-service-launch" \
  "$(destination /usr/local/bin/vllm-service-launch)"

library_destination=$(destination \
  /usr/local/lib/vllm-service-launch/vllm_service_launch)
install -d -m 0755 "$library_destination"
rm -f -- "$library_destination/identity.py"
for module in "$SOURCE_DIR"/lib/vllm_service_launch/*.py; do
  install -m 0644 "$module" "$library_destination/$(basename -- "$module")"
done

echo "Installed repository-owned vllm-service-launch under $INSTALL_ROOT"
