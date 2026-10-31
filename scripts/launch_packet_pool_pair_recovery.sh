#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
source_run="$repo/outputs/packet_pool_pair_s42_v1"
output="$repo/outputs/packet_pool_pair_s42_recovery_v1"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Recovery output/log/PID exists; refusing duplicate launch'; exit 1
fi
CUDA_VISIBLE_DEVICES='' "$python_bin" -B "$bundle/scripts/test_packet_pooling_tensor.py"
"$python_bin" -B "$bundle/scripts/recover_packet_pool_pair.py" --repo "$repo" \
  --config "$bundle/configs/packet_pool_pair_s42_v1.json" --failed-run "$source_run" \
  --output "$output" --mode check
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/recover_packet_pool_pair.py" --repo "$repo" \
  --config "$bundle/configs/packet_pool_pair_s42_v1.json" --failed-run "$source_run" \
  --output "$output" --mode run --physical-gpu "${1:-1}" --allow-shared-gpu > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "P0/P1 recovery submitted, GPU ${1:-1}; original failed run preserved, prepared node caches reused."
echo "Log: $output.log"
