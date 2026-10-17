"""Read-only artifact inventory; never infer process status from old JSON."""
import argparse
import json
from pathlib import Path
import subprocess


def collect(run_dir, pid_file=None, log_file=None):
    root = Path(run_dir)
    result = {"run_dir": str(root.resolve()), "exists": root.is_dir(), "artifacts": {}}
    for relative in ("summary.json", "status.json", "E0/config.json", "E0/status.json", "E0/metrics.json",
                     "E0/epochs.jsonl", "E0/best.pt", "E0/last.pt", "E0/predictions.csv",
                     "E1/config.json", "E1/status.json", "E1/metrics.json", "E1/epochs.jsonl",
                     "E1/best.pt", "E1/last.pt", "E1/predictions.csv"):
        path = root / relative
        entry = {"exists": path.is_file()}
        if path.is_file():
            entry["bytes"] = path.stat().st_size
            if path.suffix == ".json":
                try:
                    entry["content"] = json.loads(path.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    entry["read_error"] = str(exc)
            elif path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as stream:
                    lines = [line for line in stream if line.strip()]
                entry["epochs_recorded"] = len(lines)
                entry["last_epoch"] = lines[-1].strip() if lines else None
        result["artifacts"][relative] = entry
    if pid_file:
        path = Path(pid_file)
        value = path.read_text().strip() if path.is_file() else None
        result["pid_file"] = {"path": str(path), "content": value}
        if value and value.isdecimal():
            # PID reuse is possible: command/elapsed time must match the run.
            result["process_snapshot"] = subprocess.run(["ps", "-p", value, "-o", "pid,ppid,lstart,etime,args"],
                                                        capture_output=True, text=True).stdout
    if log_file:
        path = Path(log_file)
        if path.is_file():
            with path.open(encoding="utf-8", errors="replace") as stream:
                from collections import deque
                result["log_tail"] = list(deque(stream, maxlen=40))
        else:
            result["log_tail"] = None
    result["interpretation"] = "File existence/status alone does not establish liveness or complete scientific validation."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.run_dir, args.pid_file, args.log_file), indent=2, ensure_ascii=False))
