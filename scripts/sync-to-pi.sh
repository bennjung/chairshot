#!/bin/zsh
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PI_TARGET="${PI_TARGET:-pi@172.30.1.71:~/chairshot/}"

rsync -az --delete \
  --exclude-from="$PROJECT_ROOT/.rsyncignore" \
  --timeout=5 \
  "$PROJECT_ROOT/" "$PI_TARGET"

if [ $? -eq 0 ]; then
  echo "synced -> $PI_TARGET"
else
  echo "sync failed -> $PI_TARGET" >&2
  exit 1
fi
