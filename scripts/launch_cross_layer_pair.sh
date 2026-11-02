#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
output="$repo/outputs/cross_layer_pair_s42_v1"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Output/log/PID exists; refusing duplicate launch'; exit 1
fi
"$python_bin" -B "$bundle/scripts/test_cross_layer_pair.py"
CUDA_VISIBLE_DEVICES=''  "$python_bin" -B "$bundle/scripts/test_cross_layer_pair_tensor.py"
"$python_bin" -B "$bundle/scripts/run_cross_layer_pair.py" --repo "$repo" \
  --config "$bundle/configs/cross_layer_pair_s42_v1.json" --output "$output" --mode check
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/run_cross_layer_pair.py" --repo "$repo" \
  --config "$bundle/configs/cross_layer_pair_s42_v1.json" --output "$output" --mode run \
  --physical-gpu "${1:-1}" --allow-shared-gpu > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "CL-Last -> CL-Multi submitted, GPU ${1:-1}; each runs all150 epochs, early stopping disabled."
echo 'Cross-layer diagnostics and all preflights precede training; selected and trained-best checkpoints are separate.'
echo "Log: $output.log"
