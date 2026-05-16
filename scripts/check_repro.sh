#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/4] repository root"
pwd

echo "[2/4] expected folders"
for path in data/raw data/processed outputs third_party; do
  if [ -e "$ROOT_DIR/$path" ]; then
    echo "OK  $path"
  else
    echo "MISS $path"
  fi
done

echo "[3/4] dataset/tool hints"
echo "Put CICIoT2023 files under: $ROOT_DIR/data/raw"
echo "Put GNN4ID repo under:      $ROOT_DIR/third_party/GNN4ID"

echo "[4/4] CLI sanity"
python "$ROOT_DIR/main.py" -h

