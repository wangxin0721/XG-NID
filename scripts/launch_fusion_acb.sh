#!/usr/bin/env bash
# Invoke from the SERVER repository; this bundle never changes its checkout.
set -euo pipefail
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
repo=$(pwd)
gpu=${1:-0}
python_bin=/home/nzxn/miniconda3/envs/wx/bin/python
output="$repo/outputs/fusion_acb_b512_s42_v1"
log="$output.log"
if [[ -e "$output" || -e "$log" || -e "$output.pid" ]]; then
  echo "A/C/B output or log already exists. Inspect it; no duplicate launch."
  exit 1
fi
"$python_bin" -B -u "$bundle/scripts/run_fusion_acb_server.py" --repo "$repo" --mode check
# Atomically claim the log before starting the background job.
(set -o noclobber; : > "$log")
nohup "$python_bin" -B -u "$bundle/scripts/run_fusion_acb_server.py" \
  --repo "$repo" --mode queue --physical-gpu "$gpu" --allow-shared-gpu \
  --output "$output" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$output.pid"
printf 'A/C/B request submitted, PID=%s\nLog: %s\n' "$pid" "$log"
