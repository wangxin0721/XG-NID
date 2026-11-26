#!/usr/bin/env bash
# Run with bash; code SHA and existing worktree are never silently replaced.
set -euo pipefail
xgnid_repo="${1:?repository path required}"
xgnid_revision="${2:?full code SHA required}"
shift 2
[[ "$xgnid_revision" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full immutable SHA is required' >&2; exit 1; }
xgnid_repo="$(cd "$xgnid_repo" && pwd)"
xgnid_code="$(dirname "$xgnid_repo")/XG-NID-flow-packet-node-modulation-v1"
cd "$xgnid_repo"
git cat-file -e "$xgnid_revision^{commit}"
if [ -e "$xgnid_code" ]; then
  [ -e "$xgnid_code/.git" ] || { echo 'Destination exists without a worktree' >&2; exit 1; }
  [ "$(git -C "$xgnid_code" rev-parse HEAD)" = "$xgnid_revision" ] || { echo 'Existing worktree uses a different SHA' >&2; exit 1; }
  [ -z "$(git -C "$xgnid_code" status --porcelain --untracked-files=no)" ] || { echo 'Existing code has edits' >&2; exit 1; }
else
  git worktree add --detach "$xgnid_code" "$xgnid_revision"
fi
xgnid_output="$xgnid_repo/outputs/flow_packet_node_modulation_v1"
mkdir -p "$xgnid_repo/tmp" "$xgnid_output"
if [ -f "$xgnid_output/launcher.pid" ] && kill -0 "$(cat "$xgnid_output/launcher.pid")" 2>/dev/null; then
  echo 'Recorded launcher is still running; inspect it before restarting' >&2
  exit 1
fi
nohup python -u -B "$xgnid_code/scripts/launch_flow_packet_node_modulation.py" \
  --repo "$xgnid_repo" --resume auto "$@" \
  >> "$xgnid_repo/tmp/flow_packet_node_modulation_v1.launcher.log" 2>&1 < /dev/null &
echo $! > "$xgnid_output/launcher.pid"
echo "Launcher PID: $(cat "$xgnid_output/launcher.pid")"
echo "Launcher log: $xgnid_repo/tmp/flow_packet_node_modulation_v1.launcher.log"
echo "Training log: $xgnid_output/publication/training.log"
