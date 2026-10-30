#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
output="$repo/outputs/flow_stats_layernorm_joint_s42_v1"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Existing output/log/PID; refusing duplicate launch'; exit 1
fi
"$python_bin" -B "$bundle/scripts/run_flow_stats_layernorm.py" --repo "$repo" \
  --config "$bundle/configs/flow_stats_layernorm_s42_v1.json" --mode check --output "$output"
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/run_flow_stats_layernorm.py" --repo "$repo" \
  --config "$bundle/configs/flow_stats_layernorm_s42_v1.json" --mode run --physical-gpu "${1:-1}" \
  --allow-shared-gpu --output "$output" > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "LayerNorm residual screening submitted, GPU ${1:-1}. Log: $output.log"
echo 'Three probe updates are discarded. Training starts only after all stability/replay checks pass.'
