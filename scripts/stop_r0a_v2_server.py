"""Explicit user-requested stop and deletion of R0-A v2 only (Linux server)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time


TARGET_NAMES = ("r0_aligned_full_s42_v2", "r0_aligned_restart_control_v2", "r0_aligned_train_f1_v2")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def deletion_targets(repo):
    outputs = repo / "outputs"
    require(outputs.resolve() == outputs and outputs.is_dir(), "outputs must be a real directory within this repository")
    targets = [outputs / name for name in TARGET_NAMES]
    for target in targets:
        require(not target.is_symlink() and target.resolve() == target and target.parent == outputs,
                "Refusing linked or unexpected deletion target: " + str(target))
        require(not target.exists() or target.is_dir(), "Expected directory: " + str(target))
    return targets


def process(pid):
    root = Path("/proc") / str(pid)
    try:
        uid = root.stat().st_uid
        if uid != os.getuid():
            return None
        status = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if status[0] == "Z":
            return None
        return {"pid": int(pid), "start": status[19],
                "args": (root / "cmdline").read_bytes().decode().strip("\0").split("\0"),
                "cwd": str((root / "cwd").resolve()), "uid": uid}
    except (FileNotFoundError, ProcessLookupError):
        return None


def option(parts, key):
    if key not in parts:
        return None
    index = parts.index(key) + 1
    return parts[index] if index < len(parts) else None


def role_of(info, repo):
    if not info or info["uid"] != os.getuid():
        return None
    parts = info["args"]
    if not parts or "python" not in Path(parts[0]).name.lower():
        return None
    cwd = Path(info["cwd"])
    scripts = {(cwd / p).resolve() for p in parts[1:] if p.endswith(".py")}
    repo_arg = option(parts, "--repo")
    if repo_arg is None or (cwd / repo_arg).resolve() != repo:
        return None
    control = repo / "outputs/r0_aligned_restart_control_v2"
    output_arg = option(parts, "--output")
    if control / "run_r0_aligned_server.py" in scripts and option(parts, "--mode") == "full" \
            and output_arg and (cwd / output_arg).resolve() == repo / "outputs/r0_aligned_full_s42_v2":
        return "trainer"
    if control / "supervisor.py" in scripts and "--supervise" in parts:
        return "supervisor"
    if any(p.name in {"watch_r0a_train_f1.py", "xgnid_watch_r0a_f1.py"} for p in scripts):
        return "observer"
    return None


def discover(repo):
    found = []
    for item in Path("/proc").iterdir():
        if item.name.isdigit() and int(item.name) != os.getpid():
            info = process(item.name)
            role = role_of(info, repo)
            if role:
                found.append(dict(info, role=role))
    return found


def terminate(info, repo):
    current = process(info["pid"])
    if current is None:
        return
    require(current["start"] == info["start"] and role_of(current, repo) == info["role"], "PID identity changed")
    require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"),
            "Kernel/Python must support pidfd for identity-safe signaling; nothing deleted")
    descriptor = os.pidfd_open(info["pid"])
    try:
        current = process(info["pid"])
        require(current is not None and current["start"] == info["start"]
                and role_of(current, repo) == info["role"], "PID identity changed before signaling")
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process(info["pid"]) is None:
                return
            time.sleep(.25)
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process(info["pid"]) is None:
                return
            time.sleep(.25)
        raise RuntimeError("Process did not exit; nothing deleted: " + str(info["pid"]))
    except ProcessLookupError:
        return
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    require(sys.platform.startswith("linux"), "Run on the Linux server only")
    repo = args.repo.resolve()
    require((repo / "outputs/r0_full_launch.json").is_file(), "Wrong repository: original R0 launch not found")
    targets = deletion_targets(repo)
    require(shutil.rmtree.avoids_symlink_attacks, "Need symlink-safe recursive deletion")
    found = discover(repo)
    # Fail closed when a recorded PID is alive but no longer identifies this run.
    control = repo / "outputs/r0_aligned_restart_control_v2"
    for name, role in (("training.pid", "trainer"), ("supervisor.pid", "supervisor")):
        path = control / name
        if path.is_file():
            value = path.read_text().strip()
            require(value.isdigit(), "Invalid PID file: " + str(path))
            info = process(value)
            require(info is None or role_of(info, repo) == role,
                    "Recorded PID belongs to another process; refusing stop/deletion: " + value)
    print(json.dumps({"matched_processes": found, "delete_targets": list(map(str, targets))}, ensure_ascii=False), flush=True)
    for role in ("observer", "trainer", "supervisor"):
        for info in found:
            if info["role"] == role:
                terminate(info, repo)
    require(not discover(repo), "A matching R0-A process remains; nothing deleted")
    targets = deletion_targets(repo)
    receipt = {"status": "processes_stopped", "stopped_processes": found, "deleted": [],
               "retained": ["original R0", "R0-A smoke", "R0-V", "graph data", "older R0-A v1"]}
    path = repo / "outputs/r0a_v2_cleanup_receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            receipt["deleted"].append(str(target))
            path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt["status"] = "completed"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
