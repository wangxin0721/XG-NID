#!/usr/bin/env bash
# Two fixed150-epoch raw-graph trainings after server CUDA probe verification.
set -euo pipefail
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo="$PWD"
python_bin="${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}"
export CUDA_VISIBLE_DEVICES="${1:-1}"
"$python_bin" -B "$bundle/scripts/test_attention_operator_pair.py"
"$python_bin" -B "$bundle/scripts/test_attention_operator_training.py"
"$python_bin" -B "$bundle/scripts/run_attention_operator_pair.py" --repo "$repo" --metadata-only
"$python_bin" -B -u "$bundle/scripts/run_attention_operator_pair.py" --repo "$repo" \
  --output "$repo/outputs/${XGNID_OPERATOR_OUTPUT_NAME:-attention_operator_pair_s42_v1}"
