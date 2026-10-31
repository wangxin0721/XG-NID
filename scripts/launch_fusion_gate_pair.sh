#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
output="$repo/outputs/fusion_gate_pair_s42_v1"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Output/log/PID exists; refusing duplicate launch'; exit 1
fi
"$python_bin" -B "$bundle/scripts/run_fusion_gate_pair.py" --repo "$repo" \
  --config "$bundle/configs/fusion_gate_pair_s42_v1.json" --mode check --output "$output"
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/run_fusion_gate_pair.py" --repo "$repo" \
  --config "$bundle/configs/fusion_gate_pair_s42_v1.json" --mode run --output "$output" \
  --physical-gpu "${1:-1}" --allow-shared-gpu > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "D32-Equal -> D32-Gate submitted, GPU ${1:-1}; probes must pass before training."
echo "Log: $output.log"
