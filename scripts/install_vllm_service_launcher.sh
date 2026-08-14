#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
SOURCE_DIR="$PROJECT_ROOT/vendor/vllm-service-launch"
INSTALL_ROOT=/

usage() {
  cat <<'EOF'
Usage: scripts/install_vllm_service_launcher.sh [--root ABSOLUTE_PATH]

Installs the repository-owned vllm-service-launch CLI, Python library, sudoers
policy, systemd template, and runtime-directory tmpfiles configuration.

  --root ABSOLUTE_PATH  Stage files below an alternate root without invoking
                        systemd. Intended for packaging and tests.
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

VISUDO_BIN=$(command -v visudo || true)
if [[ -z "$VISUDO_BIN" ]]; then
  echo "ERROR: visudo is required to validate the launcher sudoers policy." >&2
  exit 1
fi

SYSTEMD_TMPFILES_BIN=
SYSTEMCTL_BIN=
if [[ "$INSTALL_ROOT" == / ]]; then
  SYSTEMD_TMPFILES_BIN=$(command -v systemd-tmpfiles || true)
  SYSTEMCTL_BIN=$(command -v systemctl || true)
  if [[ -z "$SYSTEMD_TMPFILES_BIN" || -z "$SYSTEMCTL_BIN" ]]; then
    echo "ERROR: systemd-tmpfiles and systemctl are required." >&2
    exit 1
  fi
fi

destination() {
  local path=$1
  if [[ "$INSTALL_ROOT" == / ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "${INSTALL_ROOT%/}$path"
  fi
}

install -D -m 0755 \
  "$SOURCE_DIR/bin/vllm-service-launch" \
  "$(destination /usr/local/bin/vllm-service-launch)"

library_destination=$(destination \
  /usr/local/lib/vllm-service-launch/vllm_service_launch)
install -d -m 0755 "$library_destination"
for module in "$SOURCE_DIR"/lib/vllm_service_launch/*.py; do
  install -m 0644 "$module" "$library_destination/$(basename -- "$module")"
done

sudoers_source="$SOURCE_DIR/sudoers/vllm-service-launch"
sudoers_destination=$(destination /etc/sudoers.d/vllm-service-launch)
"$VISUDO_BIN" -cf "$sudoers_source" >/dev/null
install -D -m 0440 "$sudoers_source" "$sudoers_destination"
"$VISUDO_BIN" -cf "$sudoers_destination" >/dev/null

install -D -m 0644 \
  "$SOURCE_DIR/systemd/vllm@.service" \
  "$(destination /etc/systemd/system/vllm@.service)"
install -D -m 0644 \
  "$SOURCE_DIR/systemd/vllm-metrics-targets.conf" \
  "$(destination /usr/lib/tmpfiles.d/vllm-metrics-targets.conf)"

install -d -m 0755 \
  "$(destination /run/vllm-services)" \
  "$(destination /run/vllm-metrics-targets)" \
  "$(destination /run/vllm-metrics-targets/targets)"

if [[ "$INSTALL_ROOT" == / ]]; then
  "$SYSTEMD_TMPFILES_BIN" --create vllm-metrics-targets.conf
  "$SYSTEMCTL_BIN" daemon-reload
fi

echo "Installed repository-owned vllm-service-launch under $INSTALL_ROOT"
