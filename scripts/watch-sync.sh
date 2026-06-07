#!/bin/zsh
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYNC_SCRIPT="$PROJECT_ROOT/scripts/sync-to-pi.sh"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1}"

exit_sh() {
  echo "Ctrl+C detected. Exiting..."
  exit 0
}
trap exit_sh SIGINT

calc_signature() {
  find "$PROJECT_ROOT" -type f \
    ! -path "*/.git/*" \
    ! -path "*/.venv/*" \
    ! -path "*/__pycache__/*" \
    ! -name ".DS_Store" \
    ! -name "*.pyc" \
    ! -name "*.pyo" \
    ! -name "*.csv" \
    ! -name "team7_output.zip" \
    -print0 \
  | xargs -0 stat -f "%m %z %N" \
  | sort \
  | shasum
}

echo "Initial sync..."
"$SYNC_SCRIPT"
last_signature="$(calc_signature)"

while true; do
  current_signature="$(calc_signature)"

  if [ "$current_signature" != "$last_signature" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] change detected"
    "$SYNC_SCRIPT" && last_signature="$current_signature"
  fi

  sleep "$INTERVAL_SECONDS"
done
