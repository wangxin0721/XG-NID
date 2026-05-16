#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT_DIR/data/raw" "$ROOT_DIR/data/processed" "$ROOT_DIR/outputs" "$ROOT_DIR/third_party"

echo "Workspace prepared:"
echo "  $ROOT_DIR/data/raw"
echo "  $ROOT_DIR/data/processed"
echo "  $ROOT_DIR/outputs"
echo "  $ROOT_DIR/third_party"
echo
echo "Next:"
echo "  1) Put CICIoT2023 dataset under data/raw"
echo "  2) Put GNN4ID under third_party/GNN4ID or clone it there"
echo "  3) Export graphs, then run main.py inspect/train/eval"

