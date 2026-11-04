#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
output="$repo/outputs/residual_pair_s42_v1"
audit="$repo/outputs/cross_layer_fold_check_v1.json"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Residual output/log/PID exists; refusing overwrite or duplicate launch'; exit 1
fi
"$python_bin" -B "$bundle/scripts/test_residual_pair.py"
CUDA_VISIBLE_DEVICES='' "$python_bin" -B "$bundle/scripts/test_residual_pair_tensor.py"
if [[ ! -e "$audit" ]]; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${1:-1}" "$python_bin" -B \
    "$bundle/scripts/check_cross_layer_fold.py" --source "$repo/outputs/cross_layer_pair_s42_v1" \
    --output "$audit" --device cuda:0
fi
"$python_bin" -B "$bundle/scripts/run_residual_pair.py" --repo "$repo" \
  --config "$bundle/configs/residual_pair_s42_v1.json" --output "$output" --mode check
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/run_residual_pair.py" --repo "$repo" \
  --config "$bundle/configs/residual_pair_s42_v1.json" --output "$output" --mode run \
  --physical-gpu "${1:-1}" --allow-shared-gpu > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "Res-Base -> Res-Skip submitted on GPU ${1:-1}; both complete150 epochs."
echo "Checkpoint/fold audit passed. Log: $output.log"
