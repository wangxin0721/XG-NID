#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
output="$repo/outputs/joint_fusion32_s42_v1"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Output/log/PID exists; refusing duplicate launch'; exit 1
fi
"$python_bin" -B "$bundle/scripts/run_joint_fusion32.py" --repo "$repo" \
  --config "$bundle/configs/joint_fusion32_s42_v1.json" --output "$output" --mode check
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/run_joint_fusion32.py" --repo "$repo" \
  --config "$bundle/configs/joint_fusion32_s42_v1.json" --output "$output" --mode run \
  --physical-gpu "${1:-1}" --allow-shared-gpu > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "J32-Equal -> J32-Gate submitted, GPU ${1:-1}; raw-graph probes precede training."
echo "Log: $output.log"
