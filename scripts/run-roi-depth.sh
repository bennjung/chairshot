#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

ROI_CSV="${ROI_CSV:-kp_roi_log.csv}"
DEPTH_CSV="${DEPTH_CSV:-roi_depth_log.csv}"
KP_ARGS="${KP_ARGS:-}"
DEPTH_ARGS="${DEPTH_ARGS:-}"

kp_pid=""
depth_pid=""

cleanup() {
  echo "\nStopping ROI/depth collectors..."
  if [ -n "$depth_pid" ] && kill -0 "$depth_pid" 2>/dev/null; then
    kill "$depth_pid" 2>/dev/null
  fi
  if [ -n "$kp_pid" ] && kill -0 "$kp_pid" 2>/dev/null; then
    kill "$kp_pid" 2>/dev/null
  fi
  wait 2>/dev/null
  echo "ROI log: $PROJECT_ROOT/$ROI_CSV"
  echo "Depth log: $PROJECT_ROOT/$DEPTH_CSV"
  exit 0
}
trap cleanup INT TERM

echo "Starting AI Camera ROI collector..."
python3 develops/set_kp_imx.py --log-csv "$ROI_CSV" $KP_ARGS &
kp_pid=$!

sleep 3

echo "Starting D435 ROI depth collector..."
python3 develops/collect_roi_depth.py --roi-csv "$ROI_CSV" --output-csv "$DEPTH_CSV" $DEPTH_ARGS &
depth_pid=$!

echo "Running. Press Ctrl+C to stop both processes."
wait "$kp_pid" "$depth_pid"
cleanup
