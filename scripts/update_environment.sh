#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
TORCHTPU_DIR="${TORCHTPU_DIR:-$PROJECT_ROOT/third_party/torchtpu-vllm}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
STATE_DIR="${STATE_DIR:-$PROJECT_ROOT/.state}"
TORCH_TPU_INDEX_URL="${TORCH_TPU_INDEX_URL:-https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/}"
UPDATE_SOURCE=1

usage() {
  cat <<'EOF'
Usage: scripts/update_environment.sh [--no-source-update]

Updates vllm-torchtpu to origin/main, reads its exact compatible torch and
torch-tpu pins, installs both from Google Artifact Registry with pip, and
synchronizes the project-local .venv.

  --no-source-update  Install using the vllm-torchtpu revision already checked out.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --no-source-update)
      UPDATE_SOURCE=0
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

if [[ -n "${UV_BIN:-}" ]]; then
  UV=$UV_BIN
elif command -v uv >/dev/null 2>&1; then
  UV=$(command -v uv)
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  UV="$HOME/.local/bin/uv"
else
  echo "ERROR: uv is required but was not found." >&2
  exit 1
fi

ensure_clean() {
  local name=$1
  local path=$2
  local changes

  changes=$(git -C "$path" status --porcelain --untracked-files=normal)
  if [[ -n "$changes" ]]; then
    echo "ERROR: $name has local changes; refusing to overwrite them: $path" >&2
    git -C "$path" status --short >&2
    exit 1
  fi
}

update_main() {
  local name=$1
  local path=$2
  local target_revision

  ensure_clean "$name" "$path"
  echo "Fetching latest $name main..."
  git -C "$path" fetch --prune --depth 1 origin main
  target_revision=$(git -C "$path" rev-parse FETCH_HEAD)
  git -C "$path" checkout --detach "$target_revision"
}

if (( UPDATE_SOURCE )); then
  # Check an initialized repository before submodule update so local work is
  # not hidden by a checkout attempt.
  if git -C "$TORCHTPU_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    ensure_clean "vllm-torchtpu" "$TORCHTPU_DIR"
  fi

  git -C "$PROJECT_ROOT" submodule update --init --depth 1 -- \
    third_party/torchtpu-vllm
  update_main "vllm-torchtpu" "$TORCHTPU_DIR"
fi

if ! git -C "$TORCHTPU_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: required submodule is not initialized: $TORCHTPU_DIR" >&2
  exit 1
fi
if [[ ! -f "$TORCHTPU_DIR/pyproject.toml" ]]; then
  echo "ERROR: vllm-torchtpu metadata is missing: $TORCHTPU_DIR/pyproject.toml" >&2
  exit 1
fi

torchtpu_vllm_revision=$(git -C "$TORCHTPU_DIR" rev-parse HEAD)
echo "Using vllm-torchtpu revision: $torchtpu_vllm_revision"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating Python 3.12 environment: $VENV_DIR"
  "$UV" venv --python 3.12 "$VENV_DIR"
fi

python_version=$(
  "$VENV_DIR/bin/python" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)
if [[ "$python_version" != 3.12 ]]; then
  echo "ERROR: $VENV_DIR uses Python $python_version; Python 3.12 is required." >&2
  exit 1
fi

# uv-created environments are intentionally unseeded. Add pip to the existing
# environment because torch-tpu itself must now be installed and updated by
# pip, rather than supplied as a locally built wheel override.
if ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "Bootstrapping pip in $VENV_DIR..."
  "$VENV_DIR/bin/python" -m ensurepip --upgrade
fi

mkdir -p "$STATE_DIR"

mapfile -t compatible_versions < <(
  "$VENV_DIR/bin/python" - "$TORCHTPU_DIR/pyproject.toml" <<'PY'
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as file:
    dependencies = tomllib.load(file)["project"]["dependencies"]


def canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


pins: dict[str, list[str]] = {"torch": [], "torch-tpu": []}
for dependency in dependencies:
    match = re.fullmatch(
        r"\s*([A-Za-z0-9_.-]+)\s*==\s*([^;,\s]+)\s*(?:;.*)?",
        dependency,
    )
    if match:
        name = canonicalize(match.group(1))
        if name in pins:
            pins[name].append(match.group(2))

for name in ("torch", "torch-tpu"):
    if len(pins[name]) != 1:
        raise SystemExit(
            f"expected one exact {name} dependency in vllm-torchtpu "
            f"pyproject.toml, found {pins[name]}"
        )
    print(pins[name][0])
PY
)
if (( ${#compatible_versions[@]} != 2 )); then
  echo "ERROR: could not read compatible torch and torch-tpu versions." >&2
  exit 1
fi
compatible_torch_version=${compatible_versions[0]}
compatible_torch_tpu_version=${compatible_versions[1]}

if [[ "$TORCH_TPU_INDEX_URL" != https://* ]]; then
  echo "ERROR: TORCH_TPU_INDEX_URL must use HTTPS." >&2
  exit 2
fi

if [[ -n "${TORCH_TPU_ACCESS_TOKEN:-}" ]]; then
  artifact_registry_token=$TORCH_TPU_ACCESS_TOKEN
  credential_source="TORCH_TPU_ACCESS_TOKEN"
else
  if [[ -n "${GCLOUD_BIN:-}" ]]; then
    GCLOUD=$GCLOUD_BIN
  elif command -v gcloud >/dev/null 2>&1; then
    GCLOUD=$(command -v gcloud)
  else
    echo "ERROR: gcloud is required to authenticate to the torch-tpu registry." >&2
    exit 1
  fi
  artifact_registry_token=$("$GCLOUD" auth print-access-token)
  credential_source=$(
    "$GCLOUD" auth list --filter=status:ACTIVE --format='value(account)' |
      head -n 1
  )
fi
if [[ -z "$artifact_registry_token" ]]; then
  echo "ERROR: no Google Artifact Registry access token was returned." >&2
  exit 1
fi

authenticated_index_url="${TORCH_TPU_INDEX_URL/https:\/\//https:\/\/oauth2accesstoken:${artifact_registry_token}@}"
echo "Installing torch==$compatible_torch_version and torch-tpu==$compatible_torch_tpu_version with pip..."
echo "torch-tpu index:      $TORCH_TPU_INDEX_URL"
echo "credential source:    ${credential_source:-active gcloud account}"

# Install the exact torch pin first. torch-tpu's own metadata currently uses a
# broad torch>= constraint, so resolving both implicitly could select a newer,
# ABI-incompatible torch build. Force-reinstall torch-tpu to replace any old
# source-built wheel whose PEP 440 local suffix still satisfies the public pin.
PIP_INDEX_URL="$authenticated_index_url" PIP_NO_INPUT=1 \
  "$VENV_DIR/bin/python" -m pip install \
    --disable-pip-version-check \
    --upgrade \
    --pre \
    "torch==$compatible_torch_version"
PIP_INDEX_URL="$authenticated_index_url" PIP_NO_INPUT=1 \
  "$VENV_DIR/bin/python" -m pip install \
    --disable-pip-version-check \
    --upgrade \
    --pre \
    --force-reinstall \
    --no-deps \
    "torch-tpu==$compatible_torch_tpu_version"
authenticated_index_url=""

export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo "Synchronizing vllm-torchtpu and remaining dependencies..."
(
  export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken
  export UV_INDEX_TORCH_TPU_REGISTRY_PASSWORD="$artifact_registry_token"
  cd -- "$TORCHTPU_DIR"
  "$UV" pip install \
    --python "$VENV_DIR/bin/python" \
    --upgrade \
    --pre \
    --torch-backend "$UV_TORCH_BACKEND" \
    --keyring-provider disabled \
    --editable .
)
artifact_registry_token=""

"$UV" pip check --python "$VENV_DIR/bin/python"

"$VENV_DIR/bin/python" - \
  "$compatible_torch_version" \
  "$compatible_torch_tpu_version" <<'PY'
import sys
from importlib.metadata import version
from packaging.version import Version

expected_torch = Version(sys.argv[1])
expected_torch_tpu = Version(sys.argv[2])
installed_torch = Version(version("torch"))
installed_torch_tpu = Version(version("torch-tpu"))

if installed_torch.base_version != expected_torch.base_version:
    raise SystemExit(
        f"installed torch version is {installed_torch}; expected {expected_torch}"
    )
if installed_torch_tpu != expected_torch_tpu:
    raise SystemExit(
        "installed torch-tpu version is "
        f"{installed_torch_tpu}; expected registry version {expected_torch_tpu}"
    )
PY

printf '%s\n' "$torchtpu_vllm_revision" > \
  "$STATE_DIR/last_torchtpu_vllm_revision"
printf '%s\n' "$compatible_torch_tpu_version" > \
  "$STATE_DIR/last_torch_tpu_version"
printf '%s\n' "$TORCH_TPU_INDEX_URL" > \
  "$STATE_DIR/last_torch_tpu_index_url"
printf '%s\n' "$torchtpu_vllm_revision" > "$STATE_DIR/last_source_revision"
"$UV" pip freeze --python "$VENV_DIR/bin/python" \
  > "$STATE_DIR/environment.freeze.txt"

"$VENV_DIR/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for distribution in ("vllm_torchtpu", "vllm", "torch", "torch-tpu", "libtpu"):
    try:
        value = version(distribution)
    except PackageNotFoundError:
        value = "MISSING"
    print(f"{distribution}=={value}")

import torch
import torch_tpu

print(f"torch_tpu import OK; PrivateUse1 backend={torch._C._get_privateuse1_backend_name()}")
PY

echo "torch_tpu install source: pip ($TORCH_TPU_INDEX_URL)"
