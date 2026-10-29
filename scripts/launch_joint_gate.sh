#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
output="$repo/outputs/joint_gate_s42_v1"
if [[ -e "$output" || -e "$output.log" ]]; then echo 'Existing output/log; refusing duplicate launch'; exit 1; fi
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/run_joint_gate.py" --repo "$repo" \
 --config "$bundle/configs/joint_gate_20261029.json" --physical-gpu "${1:-1}" \
 --allow-shared-gpu --output "$output" > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "Joint J0/J2 submitted. Log: $output.log"
