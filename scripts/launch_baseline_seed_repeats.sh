#!/usr/bin/env bash
set -euo pipefail
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
repo=$(pwd)
gpu=${1:-1}
python_bin=/home/nzxn/miniconda3/envs/wx/bin/python
output="$repo/outputs/baseline_seed_repeats_b512_s43_s44_v1"
log="$output.log"
if [[ -e "$output" || -e "$log" || -e "$output.pid" ]]; then
  echo "Baseline repeat output/log/PID exists; inspect before another launch."
  exit 1
fi
"$python_bin" -B -u "$bundle/scripts/run_baseline_seed_repeats_server.py" --repo "$repo" --mode check
(set -o noclobber; : > "$log")
nohup "$python_bin" -B -u "$bundle/scripts/run_baseline_seed_repeats_server.py" \
  --repo "$repo" --mode queue --physical-gpu "$gpu" --allow-shared-gpu \
  --output "$output" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$output.pid"
printf 'Four baseline repeats submitted, PID=%s\nLog: %s\n' "$pid" "$log"
