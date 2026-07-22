#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPORT_DIR="${REPORT_DIR:-$PROJECT_ROOT/reports}"
STATE_DIR="${STATE_DIR:-$PROJECT_ROOT/.state}"
REPORT_REMOTE="${REPORT_REMOTE:-origin}"
REPORT_BRANCH="${REPORT_BRANCH:-main}"

required_reports=(
  index.html
  latest.json
  throughput.svg
  throughput_history.csv
  throughput_history.json
)
for filename in "${required_reports[@]}"; do
  if [[ ! -f "$REPORT_DIR/$filename" ]]; then
    echo "ERROR: generated report is missing: $REPORT_DIR/$filename" >&2
    exit 1
  fi
done

current_branch=$(git -C "$PROJECT_ROOT" symbolic-ref --quiet --short HEAD || true)
if [[ "$current_branch" != "$REPORT_BRANCH" ]]; then
  echo "ERROR: report publication requires branch '$REPORT_BRANCH'; current branch is '${current_branch:-detached HEAD}'." >&2
  exit 1
fi
if ! git -C "$PROJECT_ROOT" remote get-url "$REPORT_REMOTE" >/dev/null 2>&1; then
  echo "ERROR: Git remote '$REPORT_REMOTE' is not configured." >&2
  exit 1
fi
if [[ -z "$(git -C "$PROJECT_ROOT" config user.name || true)" ]] ||
    [[ -z "$(git -C "$PROJECT_ROOT" config user.email || true)" ]]; then
  echo "ERROR: Git user.name and user.email are required to publish reports." >&2
  exit 1
fi

mkdir -p "$STATE_DIR"
exec 8> "$STATE_DIR/report_publish.lock"
if ! flock --nonblock 8; then
  echo "ERROR: another report publisher is already running." >&2
  exit 75
fi

if ! git -C "$PROJECT_ROOT" diff --cached --quiet; then
  echo "ERROR: the Git index already contains staged changes; refusing an automatic commit." >&2
  exit 1
fi

# Source updates intentionally leave the vllm-torchtpu submodule gitlink
# modified. It is allowed in the worktree but is never staged by this publisher.
unexpected_changes=$(git -C "$PROJECT_ROOT" status --porcelain=v1 \
  --untracked-files=all -- \
  . \
  ':(exclude)README.md' \
  ':(exclude)reports' \
  ':(exclude)reports/**' \
  ':(exclude)third_party/torchtpu-vllm')
if [[ -n "$unexpected_changes" ]]; then
  echo "ERROR: the worktree contains changes outside README.md/reports and the vllm-torchtpu submodule:" >&2
  printf '%s\n' "$unexpected_changes" >&2
  echo "Refusing to create an unattended commit." >&2
  exit 1
fi

echo "Checking $REPORT_REMOTE/$REPORT_BRANCH before publishing..."
git -C "$PROJECT_ROOT" fetch --no-tags "$REPORT_REMOTE" \
  "refs/heads/$REPORT_BRANCH"
local_tip=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
remote_tip=$(git -C "$PROJECT_ROOT" rev-parse FETCH_HEAD)
if [[ "$local_tip" != "$remote_tip" ]]; then
  echo "ERROR: local $REPORT_BRANCH ($local_tip) is not synchronized with $REPORT_REMOTE/$REPORT_BRANCH ($remote_tip)." >&2
  echo "Resolve the branch difference before the next automatic publication." >&2
  exit 1
fi

mapfile -t report_metadata < <(
  python3 - "$REPORT_DIR/latest.json" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    latest = json.load(file)
run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(latest["run_id"]))
print(run_id)
print(f'{float(latest["total_token_throughput"]):.2f}')
PY
)
if (( ${#report_metadata[@]} != 2 )); then
  echo "ERROR: could not read report metadata from $REPORT_DIR/latest.json." >&2
  exit 1
fi
run_id=${report_metadata[0]}
throughput=${report_metadata[1]}

git -C "$PROJECT_ROOT" add -A -- README.md reports

while IFS= read -r -d '' staged_path; do
  case "$staged_path" in
    README.md|reports/*) ;;
    *)
      echo "ERROR: refusing to commit unexpected staged path: $staged_path" >&2
      git -C "$PROJECT_ROOT" restore --staged -- README.md reports
      exit 1
      ;;
  esac
done < <(git -C "$PROJECT_ROOT" diff --cached --name-only -z)

if git -C "$PROJECT_ROOT" diff --cached --quiet; then
  echo "Benchmark report is already current; no commit is needed."
  exit 0
fi

commit_message="Update benchmark report: $run_id ($throughput tok/s)"
git -C "$PROJECT_ROOT" commit --no-gpg-sign -m "$commit_message"
if ! git -C "$PROJECT_ROOT" push "$REPORT_REMOTE" "HEAD:refs/heads/$REPORT_BRANCH"; then
  echo "ERROR: report commit was created locally but could not be pushed." >&2
  echo "After resolving the remote branch, retry: git push $REPORT_REMOTE $REPORT_BRANCH" >&2
  exit 1
fi

echo "Published benchmark report to $REPORT_REMOTE/$REPORT_BRANCH."
