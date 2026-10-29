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
output="$repo/outputs/frozen_fusion_n0_b512_s42_v1"
log="$output.log"
if [[ -e "$output" || -e "$log" || -e "$output.pid" ]]; then
  echo 'Output/log/PID exists; inspect it before another launch.'
  exit 1
fi
"$python_bin" -B "$bundle/scripts/run_frozen_fusion.py" --repo "$repo" \
  --config "$bundle/configs/frozen_fusion_20261029.json" --mode check
(set -o noclobber; : > "$log")
extra=()
if [[ -n "$sharing" ]]; then extra+=("$sharing"); fi
nohup "$python_bin" -B -u "$bundle/scripts/run_frozen_fusion.py" --repo "$repo" \
  --config "$bundle/configs/frozen_fusion_20261029.json" --mode queue --physical-gpu "$gpu" \
  "${extra[@]}" --output "$output" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$output.pid"
printf 'F0/F1/F2 queue submitted, PID=%s\nLog: %s\n' "$pid" "$log"
