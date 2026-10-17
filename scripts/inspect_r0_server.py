"""Read-only R0 preflight during E1: no torch imports or graph deserialization.

Run from a file or via `git show ...:scripts/inspect_r0_server.py | python -`.
Filename counts describe current naming only, not author sample identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys


def command(argv, cwd=None):
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=15)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def inspect(repo, data):
    report = {"schema": "xgnid.r0.preflight.v1", "time_utc": datetime.now(timezone.utc).isoformat(),
              "repo": str(repo), "inspection_python": sys.executable,
              "e1_user_report": "running; must preserve existing process",
              "graph_tensors_loaded": False, "cuda_context_created": False,
              "training_started": False, "processes": [], "process_read_errors": []}
    for name, args in (("git_head", ["git", "rev-parse", "HEAD"]),
                       ("git_branch", ["git", "branch", "--show-current"]),
                       ("git_status", ["git", "status", "--short"])):
        report[name] = command(args, cwd=repo)
    proc = Path("/proc")
    report["proc_available"] = proc.is_dir()
    if proc.is_dir():
        for directory in proc.iterdir():
            if not directory.name.isdecimal():
                continue
            try:
                args = (directory / "cmdline").read_bytes().decode(errors="replace").strip("\0").split("\0")
                if not any(Path(arg).name == "train_grouped.py" for arg in args):
                    continue
                entry = {"pid": int(directory.name), "argv": args}
                for key, filename in (("python_executable", "exe"), ("cwd", "cwd")):
                    try:
                        entry[key] = str((directory / filename).readlink())
                    except OSError as exc:
                        entry[key] = {"error": str(exc)}
                for option, key in (("--output", "output"), ("--experiments", "experiments_first_arg")):
                    if option in args and args.index(option) + 1 < len(args):
                        entry[key] = args[args.index(option) + 1]
                report["processes"].append(entry)
            except FileNotFoundError:
                # A process can exit while /proc is being read.
                continue
            except OSError as exc:
                report["process_read_errors"].append({"pid": directory.name, "error": str(exc)})
    report["nvidia_smi"] = command(["nvidia-smi"])
    graph_info = {"path": str(data), "exists": data.is_dir(), "filename_counts": {},
                  "examples": {}, "author_sample_identity_verified": False}
    counts = Counter()
    if data.is_dir():
        try:
            for path in data.rglob("*.pt"):
                match = re.fullmatch(r"data_(test_)?(\d+)\.pt", path.name)
                part = ("source_test_name" if match[1] else "source_train_name") if match else "other"
                counts[part] += 1
                graph_info["examples"].setdefault(part, str(path.relative_to(data)))
        except OSError as exc:
            graph_info["scan_error"] = str(exc)
    graph_info["filename_counts"] = dict(counts)
    report["graphs"] = graph_info
    report["csv_candidates"] = []
    source = repo / "data/processed/CICIoT2023_processed"
    if source.is_dir():
        for path in source.rglob("*.csv"):
            report["csv_candidates"].append({"path": str(path), "bytes": path.stat().st_size,
                                              "sha256": "not computed in lightweight preflight"})
    report["next_gate"] = "Review process/runtime and source membership evidence before manifest scan; GPU smoke waits for free GPU."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not repo.is_dir():
        parser.error("Repository directory does not exist")
    data = args.data.resolve() if args.data else repo / "data/processed/CICIoT2023_processed/processed"
    report = inspect(repo, data)
    content = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content + "\n")
    print(content)


if __name__ == "__main__":
    main()
