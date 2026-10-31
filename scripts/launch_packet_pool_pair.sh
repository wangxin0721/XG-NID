#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
output="$repo/outputs/packet_pool_pair_s42_v1"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Output/log/PID exists; refusing duplicate launch'; exit 1
fi
CUDA_VISIBLE_DEVICES='' "$python_bin" -B "$bundle/scripts/test_packet_pooling_tensor.py"
"$python_bin" -B "$bundle/scripts/run_packet_pool_pair.py" --repo "$repo" \
  --config "$bundle/configs/packet_pool_pair_s42_v1.json" --mode check --output "$output"
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/run_packet_pool_pair.py" --repo "$repo" \
  --config "$bundle/configs/packet_pool_pair_s42_v1.json" --mode run --output "$output" \
  --physical-gpu "${1:-1}" --allow-shared-gpu > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "P0 -> P1 submitted, GPU ${1:-1}; tensor probes and epoch0 replay must pass before training."
echo "Log: $output.log"
