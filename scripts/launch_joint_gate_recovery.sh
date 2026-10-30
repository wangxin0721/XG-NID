#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
output="$repo/outputs/joint_gate_recovery_v2_s42_v1"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Existing output/log/PID; inspect before another launch'; exit 1
fi
"$python_bin" -B "$bundle/scripts/run_joint_gate.py" --repo "$repo" \
 --config "$bundle/configs/joint_gate_recovery_v2_20261030.json" --mode check --output "$output"
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/run_joint_gate.py" --repo "$repo" \
 --config "$bundle/configs/joint_gate_recovery_v2_20261030.json" --mode run --physical-gpu "${1:-1}" \
 --allow-shared-gpu --output "$output" > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "Recovery V2 reused, J0/J2 submitted. Log: $output.log"
