#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
source="$repo/outputs/fair_vector_five_runs_v1"
output="$repo/outputs/fair_vector_five_runs_recovery_v1"
if [[ -e "$output" || -e "$output.log" || -e "$output.pid" ]]; then
  echo 'Recovery output/log/PID already exists; inspect before another launch.'
  exit 1
fi
"$python_bin" -B "$bundle/scripts/resume_fair_vector.py" --repo "$repo" \
  --config "$bundle/configs/fair_vector_20261029.json" --source "$source" --output "$output" --mode check
(set -o noclobber; : > "$output.log")
nohup "$python_bin" -B -u "$bundle/scripts/resume_fair_vector.py" --repo "$repo" \
  --config "$bundle/configs/fair_vector_20261029.json" --source "$source" --output "$output" \
  --mode queue --physical-gpu "${1:-0}" --allow-shared-gpu > "$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
echo "Recovery submitted on GPU ${1:-0}; log: $output.log"
