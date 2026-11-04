#!/usr/bin/env bash
# Foreground implementation checks and a small raw-graph probe; no full training.
set -euo pipefail
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo="$PWD"
python_bin="${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}"
export CUDA_VISIBLE_DEVICES="${1:-1}"
"$python_bin" -B "$bundle/scripts/test_attention_operator_pair.py"
"$python_bin" -B "$bundle/scripts/probe_attention_operator_pair.py" \
  --repo "$repo" --device cuda:0 \
  --output "$repo/outputs/${XGNID_OPERATOR_PROBE_NAME:-attention_operator_probe_s42_v1}"
