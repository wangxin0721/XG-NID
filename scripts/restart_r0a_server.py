"""Launch an unchanged, smoke-verified R0-A from epoch 1 if no old checkpoint exists.

Server only, standard library. Keeps interrupted output and graph-rebuild evidence.
Never resumes weights or overwrites old/new runs. --status never launches a job.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


BASE_COMMIT = "6cce237"
BASE_IDENTITY = "082c4dc27605235106a5409aeacb5cd2aa9e9ff5bbcbf59f7099ab609b040080"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(text):
    return hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def paths(repo):
    return (repo / "outputs/r0_aligned_restart_control_v2",
            repo / "outputs/r0_aligned_full_s42_v2",
            repo / "outputs/r0_aligned_smoke_s42_v1/summary.json")


def no_training_processes():
    names = {"train_grouped.py", "run_r0_edge.py", "run_r0_aligned_server.py", "xgnid_r0_aligned.py"}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            parts = (entry / "cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if parts and "python" in Path(parts[0]).name.lower():
            require(not names.intersection(Path(p).name for p in parts[1:] if p.endswith(".py")),
                    "Known training process still exists, PID=" + entry.name)


def readiness(repo):
    require((repo / "outputs/r0_full_launch.json").is_file(), "Wrong repository: original R0 launch is missing")
    old = repo / "outputs/r0_aligned_full_s42_v1"
    checkpoint = old / "last.pt"
    require(not checkpoint.exists(), "Found checkpoint at absolute path; retain it and inspect before restarting: " + str(checkpoint))
    if (old / "summary.json").is_file():
        require(read(old / "summary.json").get("status") != "completed", "Old R0-A already completed; no duplicate run")
    no_training_processes()
    control, output, smoke_path = paths(repo)
    require(not output.exists(), "New R0-A output already exists; retained: " + str(output))
    smoke = read(smoke_path)
    require((smoke["status"], smoke["exit_code"], smoke["mode"], smoke["epochs_completed"])
            == ("completed", 0, "smoke", 2), "R0-A CUDA smoke is incomplete")
    require(smoke["optimizer_step_passed"] and smoke["test"] is None
            and smoke["binding"]["runtime"]["device_type"] == "cuda", "Invalid CUDA smoke")
    require(smoke["binding"]["tool_identity"] == BASE_IDENTITY, "Different R0-A entry passed smoke")
    require(sha(smoke_path.parent / "last.pt") == smoke["last_checkpoint_sha256"], "Smoke checkpoint changed")
    text = subprocess.check_output(["git", "show", BASE_COMMIT + ":scripts/run_r0_aligned_server.py"], cwd=repo).decode("utf-8-sig")
    text = text.replace("\r\n", "\n")
    require(identity(text) == BASE_IDENTITY, "Locked training entry identity differs")
    rebuild = repo / "outputs/r0_graph_rebuild_author_offline_v2"
    report = read(rebuild / "summary.json")
    require((report["status"], report["exit_code"], report["graphs_completed"], report["all_old_inputs_and_labels_equal"])
            == ("completed", 0, 185557, True), "Graph rebuild equivalence not confirmed")
    require((rebuild / "exitcode").read_text().strip() == "0", "Rebuild supervisor did not exit successfully")
    require(sha(rebuild / "graphs.jsonl") == report["graphs_manifest_sha256"], "Graph rebuild ledger changed")
    require(report["old_manifest_id"] == smoke["binding"]["manifest_id"], "Rebuilt graph comparison used different membership")
    return text, {"old_output": str(old), "old_output_exists": old.is_dir(), "old_checkpoint_absolute": str(checkpoint),
                  "old_checkpoint_exists": False, "start_epoch": 1, "target_epoch": 30,
                  "base_commit": BASE_COMMIT, "training_entry_identity": BASE_IDENTITY,
                  "smoke_report": str(smoke_path), "smoke_report_sha256": sha(smoke_path),
                  "rebuild_summary_sha256": sha(rebuild / "summary.json"),
                  "graph_input": "original frozen graph membership; full rebuild confirms identical inputs/labels",
                  "not_a_checkpoint_resume": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(sys.platform.startswith("linux"), "Run on the Linux server only")
    repo = args.repo.resolve()
    control, output, smoke = paths(repo)
    if args.status:
        value = {"control": str(control), "output": str(output)}
        for key, path in (("launch", control / "launch.json"), ("supervisor_status", control / "status.json"),
                          ("training_status", output / "status.json"), ("summary", output / "summary.json")):
            value[key] = read(path) if path.is_file() else None
        for name in ("training.pid", "supervisor.pid", "exitcode"):
            value[name] = (control / name).read_text().strip() if (control / name).is_file() else None
        if value["training.pid"]:
            proc = Path("/proc") / value["training.pid"] / "cmdline"
            try:
                value["current_pid_command"] = proc.read_bytes().decode().replace("\0", " ")
            except OSError:
                value["current_pid_command"] = None
        for key, path, size in (("epochs_tail", output / "epochs.jsonl", 3), ("log_tail", control / "run.log", 10)):
            if path.is_file():
                with path.open(encoding="utf-8", errors="replace") as stream:
                    value[key] = list(deque(stream, maxlen=size))
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    if args.supervise:
        rc = 1
        try:
            text, evidence = readiness(repo)
            require(evidence == read(control / "launch.json")["readiness"], "Readiness evidence changed after submission")
            require(identity((control / "run_r0_aligned_server.py").read_text(encoding="utf-8")) == identity(text), "Staged training source changed")
            command = [sys.executable, "-B", "-u", str(control / "run_r0_aligned_server.py"),
                       "--repo", str(repo), "--mode", "full", "--device", "cuda:0",
                       "--smoke-report", str(smoke), "--output", str(output)]
            child = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL)
            (control / "training.pid").write_text(str(child.pid) + "\n")
            write(control / "status.json", {"status": "process_started", "training_pid": child.pid})
            rc = child.wait()
            if rc == 0:
                final = read(output / "summary.json")
                require((final["status"], final["exit_code"], final["epochs_completed"]) == ("completed", 0, 30), "Training completion inconsistent")
            write(control / "status.json", {"status": "completed" if rc == 0 else "failed", "exit_code": rc})
        except BaseException as exc:
            rc = 1
            traceback.print_exc()
            write(control / "status.json", {"status": "failed", "exit_code": rc, "error": str(exc)})
        finally:
            (control / "exitcode").write_text(str(rc) + "\n")
        return rc
    text, evidence = readiness(repo)
    control.mkdir(exist_ok=False)
    (control / "run_r0_aligned_server.py").write_text(text, encoding="utf-8")
    shutil.copyfile(Path(__file__), control / "supervisor.py")
    write(control / "launch.json", {"readiness": evidence, "python": sys.executable, "output": str(output)})
    write(control / "status.json", {"status": "submitted"})
    with (control / "run.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-B", "-u", str(control / "supervisor.py"), "--repo", str(repo), "--supervise"],
                                 cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), start_new_session=True)
    (control / "supervisor.pid").write_text(str(child.pid) + "\n")
    print(json.dumps({"submission": "recorded; check actual training status", "supervisor_pid": child.pid,
                      "output": str(output), "log": str(control / "run.log"), "readiness": evidence}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
