#!/usr/bin/env bash
set -euo pipefail
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
repo=$(pwd)
gpu=${1:-1}
python_bin=/home/nzxn/miniconda3/envs/wx/bin/python
output="$repo/outputs/fusion_bh_b1024_s42_resume_v1"
log="$output.log"
if [[ -e "$output" || -e "$log" || -e "$output.pid" ]]; then
  echo "Recovery output/log/PID already exists; inspect it before another launch."
  exit 1
fi
"$python_bin" -B -u "$bundle/scripts/resume_fusion_bh_b1024_server.py" --repo "$repo" --mode check
(set -o noclobber; : > "$log")
nohup "$python_bin" -B -u "$bundle/scripts/resume_fusion_bh_b1024_server.py" \
  --repo "$repo" --mode resume --physical-gpu "$gpu" --allow-shared-gpu \
  --output "$output" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$output.pid"
printf 'B-H1024 recovery submitted, PID=%s\nLog: %s\n' "$pid" "$log"
