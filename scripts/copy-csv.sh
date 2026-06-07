#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${DEST_DIR:-$HOME/pics}"

if [ "$#" -gt 0 ]; then
  CSV_FILES=("$@")
else
  CSV_FILES=("kp_roi_log.csv" "roi_depth_log.csv")
fi

mkdir -p "$DEST_DIR"

copied=0
missing=0

for csv_file in "${CSV_FILES[@]}"; do
  if [ "$csv_file" = /* ]; then
    src="$csv_file"
  else
    src="$PROJECT_ROOT/$csv_file"
  fi

  if [ ! -f "$src" ]; then
    echo "missing: $src"
    missing=$((missing + 1))
    continue
  fi

  cp "$src" "$DEST_DIR/"
  echo "copied: $src -> $DEST_DIR/"
  copied=$((copied + 1))
done

echo "done: copied=$copied missing=$missing dest=$DEST_DIR"

if [ "$copied" -eq 0 ]; then
  exit 1
fi
