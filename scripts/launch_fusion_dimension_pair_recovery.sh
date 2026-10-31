#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
source_output="$repo/outputs/fusion_dimension_pair_s42_v1"
output="$repo/outputs/fusion_dimension_pair_s42_recovery_v1"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Recovery output/log/PID exists; refusing duplicate launch'; exit 1
fi
"$python_bin" -B "$bundle/scripts/recover_fusion_dimension_pair.py" --repo "$repo" \
  --config "$bundle/configs/fusion_dimension_pair_s42_v1.json" --mode check \
  --source "$source_output" --output "$output"
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/recover_fusion_dimension_pair.py" --repo "$repo" \
  --config "$bundle/configs/fusion_dimension_pair_s42_v1.json" --mode run \
  --source "$source_output" --output "$output" --physical-gpu "${1:-1}" \
  --allow-shared-gpu > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "Recovery submitted, GPU ${1:-1}; reuse caches/clean starts/probes, preserve source."
echo "Log: $output.log"
