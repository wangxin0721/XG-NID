#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${XGNID_PYTHON:-/home/nzxn/miniconda3/envs/wx/bin/python}
gpu=${1:-1}
sharing=${2:-}
if [[ -n "$sharing" && "$sharing" != '--allow-shared-gpu' ]]; then
  echo 'Second argument, if present, must be --allow-shared-gpu'
  exit 1
fi
output="$repo/outputs/fusion_seed_repeats_s43_s44_v1"
log="$output.log"
if [[ -e "$output" || -e "$log" || -e "$output.pid" ]]; then
  echo 'Output/log/PID exists; inspect before another launch.'
  exit 1
fi
"$python_bin" -B "$bundle/scripts/run_fusion_seed_repeats.py" --repo "$repo" \
  --config "$bundle/configs/fusion_seed_repeats_20261029.json" --mode check
(set -o noclobber; : > "$log")
extra=()
if [[ -n "$sharing" ]]; then extra+=("$sharing"); fi
nohup "$python_bin" -B -u "$bundle/scripts/run_fusion_seed_repeats.py" --repo "$repo" \
  --config "$bundle/configs/fusion_seed_repeats_20261029.json" --mode queue --physical-gpu "$gpu" \
  "${extra[@]}" --output "$output" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$output.pid"
printf 'Six N0/F1/F2 trainings submitted, PID=%s\nLog: %s\n' "$pid" "$log"
