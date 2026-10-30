#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
output="$repo/outputs/fusion_dimension_pair_s42_v1"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Existing output/log/PID; refusing duplicate launch'; exit 1
fi
"$python_bin" -B "$bundle/scripts/run_fusion_dimension_pair.py" --repo "$repo" \
  --config "$bundle/configs/fusion_dimension_pair_s42_v1.json" --mode check --output "$output"
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/run_fusion_dimension_pair.py" --repo "$repo" \
  --config "$bundle/configs/fusion_dimension_pair_s42_v1.json" --mode run --physical-gpu "${1:-1}" \
  --allow-shared-gpu --output "$output" > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "D8 -> D32 submitted on GPU ${1:-1}; new heads, frozen encoders, max100/patience20, epoch0 excluded."
echo "Log: $output.log"
