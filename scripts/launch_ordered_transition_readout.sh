#!/bin/sh
# Use the existing server training environment. Source data and previous outputs are read in place.
set -eu
repo=$(CDPATH= cd -- "${1:-.}" && pwd)
action=${2:-42}
run_name=${3:-ordered_transition_readout_200}
code=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
case "$action" in
  42|43|44|test) ;;
  *) printf '%s\n' 'Use 42, 43, 44 or test as the second argument.' >&2; exit 2 ;;
esac
case "$run_name" in
  ''|.|..|*/*) printf '%s\n' 'Use a directory name directly under outputs.' >&2; exit 2 ;;
esac
mkdir -p "$repo/outputs/$run_name"
log="$repo/outputs/$run_name-$action.log"
finish() {
  rc=$?
  trap - 0
  cp "$log" "$repo/outputs/$run_name/launcher-$action.log"
  printf '%s\n' "$rc" > "$repo/outputs/$run_name/exitcode-$action.txt"
  # Publish every generated file for this experiment, including failed-run evidence.
  if python -B "$code/scripts/sync_server_outputs.py" publish --repo "$repo" --only "$run_name"; then
    exit "$rc"
  else
    printf '%s\n' 'Result upload failed. Rerun sync_server_outputs.py publish for this directory.' >&2
    exit 1
  fi
}
exec >> "$log" 2>&1
trap finish 0
export PYTHONPATH="$code/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$action" = test ]; then
  python -u "$code/scripts/run_ordered_transition_readout.py" --repo "$repo" --output "outputs/$run_name" --test
else
  python -u "$code/scripts/run_ordered_transition_readout.py" --repo "$repo" --output "outputs/$run_name" --seed "$action" --resume
fi
