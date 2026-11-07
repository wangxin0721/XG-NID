#!/usr/bin/env bash
set -euo pipefail

bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo=${XGNID_REPO:-"$PWD"}
python=${XGNID_PYTHON:-python}
gpu=${1:-1}
cd -- "$repo"
mkdir -p outputs
test ! -e outputs/flow_query_joint_packet2_s42_v1

# Exercise the actual resume/Adam path on CPU before submitting the GPU run.
CUDA_VISIBLE_DEVICES="" "$python" -B "$bundle/scripts/test_flow_query_joint_packet2.py"

run_log=$(mktemp "$PWD/outputs/flow_query_joint_packet2_s42_v1.XXXXXX.log")
CUDA_VISIBLE_DEVICES="$gpu" nohup "$python" -B -u \
  "$bundle/scripts/run_flow_query_joint_packet2.py" --repo "$PWD" --train \
  >"$run_log" 2>&1 </dev/null &
printf 'Training PID: %s\nLog: %s\n' "$!" "$run_log"
printf 'Watch: tail -n 50 -F %q\n' "$run_log"
