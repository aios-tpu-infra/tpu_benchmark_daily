#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
TORCHTPU_DIR="${TORCHTPU_DIR:-$PROJECT_ROOT/third_party/torchtpu-vllm}"
TORCH_TPU_DIR="${TORCH_TPU_DIR:-$PROJECT_ROOT/third_party/torch_tpu}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
STATE_DIR="${STATE_DIR:-$PROJECT_ROOT/.state}"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.runtime}"
BAZEL_OUTPUT_USER_ROOT="${BAZEL_OUTPUT_USER_ROOT:-$RUNTIME_DIR/bazel}"
UPDATE_SOURCE=1

usage() {
  cat <<'EOF'
Usage: scripts/update_environment.sh [--no-source-update]

Updates vllm-torchtpu and torch_tpu to origin/main, builds a Python 3.12
torch_tpu wheel locally with Bazel, and synchronizes the project-local .venv.

  --no-source-update  Build and install the revisions already checked out.
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

if [[ -n "${BAZEL_BIN:-}" ]]; then
  BAZEL=$BAZEL_BIN
elif command -v bazelisk >/dev/null 2>&1; then
  BAZEL=$(command -v bazelisk)
elif [[ -x "$HOME/.local/bin/bazelisk" ]]; then
  BAZEL="$HOME/.local/bin/bazelisk"
elif command -v bazel >/dev/null 2>&1; then
  BAZEL=$(command -v bazel)
else
  echo "ERROR: Bazel or Bazelisk is required but was not found." >&2
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
  # Check initialized repositories before submodule update so local work is not
  # hidden by a checkout attempt.
  if git -C "$TORCHTPU_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    ensure_clean "vllm-torchtpu" "$TORCHTPU_DIR"
  fi
  if git -C "$TORCH_TPU_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    ensure_clean "torch_tpu" "$TORCH_TPU_DIR"
  fi

  git -C "$PROJECT_ROOT" submodule update --init --depth 1 -- \
    third_party/torchtpu-vllm \
    third_party/torch_tpu

  update_main "vllm-torchtpu" "$TORCHTPU_DIR"
  update_main "torch_tpu" "$TORCH_TPU_DIR"
fi

for submodule in "$TORCHTPU_DIR" "$TORCH_TPU_DIR"; do
  if ! git -C "$submodule" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: required submodule is not initialized: $submodule" >&2
    exit 1
  fi
done

torchtpu_vllm_revision=$(git -C "$TORCHTPU_DIR" rev-parse HEAD)
torch_tpu_revision=$(git -C "$TORCH_TPU_DIR" rev-parse HEAD)
torch_tpu_short_revision=${torch_tpu_revision:0:12}
echo "Using vllm-torchtpu revision: $torchtpu_vllm_revision"
echo "Using torch_tpu revision:      $torch_tpu_revision"

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

mkdir -p "$STATE_DIR" "$RUNTIME_DIR/wheels" "$BAZEL_OUTPUT_USER_ROOT"

# vllm-torchtpu declares an exact compatible nightly version. Keep that public
# version so standard dependency checks pass, and add the actual source SHA as
# a PEP 440 local version. An exact specifier without a local part accepts it.
compatible_torch_tpu_version=$(
  "$VENV_DIR/bin/python" - "$TORCHTPU_DIR/pyproject.toml" <<'PY'
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as file:
    dependencies = tomllib.load(file)["project"]["dependencies"]

matches = [
    match.group(1)
    for dependency in dependencies
    if (
        match := re.fullmatch(
            r"torch[-_]tpu\s*==\s*([^;,\s]+)", dependency, re.IGNORECASE
        )
    )
]
if len(matches) != 1:
    raise SystemExit(
        "expected one exact torch-tpu dependency in vllm-torchtpu pyproject.toml"
    )
print(matches[0])
PY
)
wheel_base_version=$(
  "$VENV_DIR/bin/python" - "$TORCH_TPU_DIR/pyproject.toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as file:
    print(tomllib.load(file)["project"]["version"])
PY
)
if [[ "$compatible_torch_tpu_version" != "$wheel_base_version"* ]] || \
    [[ "$compatible_torch_tpu_version" == *+* ]]; then
  echo "ERROR: unsupported vllm-torchtpu torch-tpu pin:" >&2
  echo "  $compatible_torch_tpu_version (source base is $wheel_base_version)" >&2
  exit 1
fi
wheel_version="${compatible_torch_tpu_version}+g${torch_tpu_short_revision}"
wheel_suffix=${wheel_version#"$wheel_base_version"}
bazel_version=$(<"$TORCH_TPU_DIR/.bazelversion")

echo "Building torch_tpu==$wheel_version locally with Bazel $bazel_version..."
(
  cd -- "$TORCH_TPU_DIR"
  "$BAZEL" \
    --output_user_root="$BAZEL_OUTPUT_USER_ROOT" \
    build \
    -c opt \
    --config=no_rbe \
    --repo_env="WHEEL_VERSION_EXTRAS=$wheel_suffix" \
    --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
    --define PYTHON_VERSION=3.12 \
    //ci/wheel:torch_tpu_wheel
)

bazel_wheel_dir="$TORCH_TPU_DIR/bazel-bin/ci/wheel"
mapfile -t built_wheels < <(
  find -L "$bazel_wheel_dir" -maxdepth 1 -type f \
    -name "torch_tpu-${wheel_version}-cp312-cp312-*.whl" \
    -print | sort
)
if (( ${#built_wheels[@]} != 1 )); then
  echo "ERROR: expected one Python 3.12 wheel for torch_tpu==$wheel_version," >&2
  echo "but found ${#built_wheels[@]} beneath $bazel_wheel_dir." >&2
  find -L "$bazel_wheel_dir" -maxdepth 1 -type f -name '*.whl' -print >&2 || true
  exit 1
fi

wheel_cache_dir="$RUNTIME_DIR/wheels/$torch_tpu_revision"
mkdir -p "$wheel_cache_dir"
wheel_path="$wheel_cache_dir/$(basename -- "${built_wheels[0]}")"
wheel_tmp=$(mktemp "$wheel_cache_dir/.torch_tpu-wheel.XXXXXX")
trap 'rm -f -- "$wheel_tmp"' EXIT
cp -- "${built_wheels[0]}" "$wheel_tmp"
chmod 0644 "$wheel_tmp"
mv -f -- "$wheel_tmp" "$wheel_path"
trap - EXIT
wheel_uri=$(
  "$VENV_DIR/bin/python" - "$wheel_path" <<'PY'
import pathlib
import sys

print(pathlib.Path(sys.argv[1]).resolve().as_uri())
PY
)

override_file="$STATE_DIR/environment.overrides.txt"
printf 'torch-tpu @ %s\n' "$wheel_uri" > "$override_file"

export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo "Installing vllm-torchtpu with the locally built torch_tpu wheel..."
(
  cd -- "$TORCHTPU_DIR"
  "$UV" pip install \
    --python "$VENV_DIR/bin/python" \
    --upgrade \
    --pre \
    --torch-backend "$UV_TORCH_BACKEND" \
    --keyring-provider disabled \
    --no-sources-package torch-tpu \
    --overrides "$override_file" \
    --reinstall-package torch-tpu \
    --editable .
)

"$UV" pip check --python "$VENV_DIR/bin/python"

installed_torch_tpu_version=$(
  "$VENV_DIR/bin/python" -c \
    'from importlib.metadata import version; print(version("torch-tpu"))'
)
if [[ "$installed_torch_tpu_version" != "$wheel_version" ]]; then
  echo "ERROR: installed torch-tpu version is $installed_torch_tpu_version;" >&2
  echo "expected locally built version $wheel_version." >&2
  exit 1
fi

printf '%s\n' "$torchtpu_vllm_revision" > \
  "$STATE_DIR/last_torchtpu_vllm_revision"
printf '%s\n' "$torch_tpu_revision" > "$STATE_DIR/last_torch_tpu_revision"
printf '%s\n' "$wheel_path" > "$STATE_DIR/last_torch_tpu_wheel"
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
PY

echo "Local torch_tpu wheel: $wheel_path"
