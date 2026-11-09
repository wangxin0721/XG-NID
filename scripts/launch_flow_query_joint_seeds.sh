#!/usr/bin/env bash
set -euo pipefail

bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo=${XGNID_REPO:-"$PWD"}
python=${XGNID_PYTHON:-python}
gpu=${1:-1}
retry=${2:-}
if [[ $# -gt 2 || ( -n "$retry" && "$retry" != --retry-source-check-failure ) ]]; then
  printf 'Usage: %s [GPU] [--retry-source-check-failure]\n' "$0" >&2
  exit 2
fi
cd -- "$repo"
mkdir -p outputs
retry_args=()
if [[ "$retry" == --retry-source-check-failure ]]; then
  retry_args+=(--retry-source-check-failure)
else
  test ! -e outputs/flow_query_joint_seeds43_44_v1
fi

CUDA_VISIBLE_DEVICES="" "$python" -B "$bundle/scripts/test_flow_query_joint_seeds.py"
CUDA_VISIBLE_DEVICES="" "$python" -B "$bundle/scripts/run_flow_query_joint_seeds.py" --repo "$PWD" --check-source-only
run_log=$(mktemp "$PWD/outputs/flow_query_joint_seeds43_44_v1.XXXXXX.log")
CUDA_VISIBLE_DEVICES="$gpu" nohup "$python" -B -u \
  "$bundle/scripts/run_flow_query_joint_seeds.py" --repo "$PWD" --train "${retry_args[@]}" \
  >"$run_log" 2>&1 </dev/null &
printf 'Training PID: %s\nLog: %s\n' "$!" "$run_log"
printf 'Watch: tail -n 50 -F %q\n' "$run_log"
