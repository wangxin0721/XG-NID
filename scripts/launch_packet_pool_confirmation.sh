#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
output="$repo/outputs/packet_pool_confirmation_s42_v1"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Confirmation output/log/PID exists; refusing another round or overwrite'; exit 1
fi
"$python_bin" -B "$bundle/scripts/test_packet_pool_confirmation.py"
CUDA_VISIBLE_DEVICES='' "$python_bin" -B "$bundle/scripts/test_packet_pooling_tensor.py"
CUDA_VISIBLE_DEVICES='' "$python_bin" -B "$bundle/scripts/test_packet_pool_confirmation_tensor.py"
"$python_bin" -B "$bundle/scripts/run_packet_pool_confirmation.py" --repo "$repo" \
  --config "$bundle/configs/packet_pool_confirmation_s42_v1.json" --output "$output" --mode check
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/run_packet_pool_confirmation.py" --repo "$repo" \
  --config "$bundle/configs/packet_pool_confirmation_s42_v1.json" --output "$output" --mode run \
  --physical-gpu "${1:-1}" --allow-shared-gpu > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "One P0/P1 budget confirmation submitted, GPU ${1:-1}; both run all150 epochs."
echo "The completed20-epoch screen is preserved. Log: $output.log"
