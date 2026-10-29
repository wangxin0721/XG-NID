#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
gpu=${1:-1}
sharing=${2:-}
if [[ -n "$sharing" && "$sharing" != '--allow-shared-gpu' ]]; then exit 1; fi
output="$repo/outputs/fair_vector_five_runs_v1"
log="$output.log"
if [[ -e "$output" || -e "$log" || -e "$output.pid" ]]; then
  echo 'Output/log/PID exists; inspect before another launch.'
  exit 1
fi
"$python_bin" -B "$bundle/scripts/run_fair_vector.py" --repo "$repo" \
  --config "$bundle/configs/fair_vector_20261029.json" --mode check
(set -o noclobber; : > "$log")
extra=()
if [[ -n "$sharing" ]]; then extra+=("$sharing"); fi
nohup "$python_bin" -B -u "$bundle/scripts/run_fair_vector.py" --repo "$repo" \
  --config "$bundle/configs/fair_vector_20261029.json" --mode queue --physical-gpu "$gpu" \
  "${extra[@]}" --output "$output" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$output.pid"
printf 'Five fixed trainings submitted, PID=%s\nLog: %s\n' "$pid" "$log"
