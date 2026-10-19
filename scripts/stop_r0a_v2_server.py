"""Explicit user-requested stop and deletion of R0-A v2 only (Linux server)."""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time


TARGET_NAMES = ("r0_aligned_full_s42_v2", "r0_aligned_restart_control_v2", "r0_aligned_train_f1_v2")
SCRIPT_NAMES = {"run_r0_aligned_server.py", "supervisor.py", "watch_r0a_train_f1.py", "xgnid_watch_r0a_f1.py"}


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


def candidate_args(parts):
    return bool(parts and "python" in Path(parts[0]).name.lower()
                and SCRIPT_NAMES.intersection(Path(p).name for p in parts[1:] if p.endswith(".py")))


def process(pid, *, scanning=False):
    root = Path("/proc") / str(pid)
    parts = None
    try:
        uid = root.stat().st_uid
        if uid != os.getuid():
            require(scanning, "Recorded PID belongs to another user; nothing deleted: " + str(pid))
            return None
        parts = (root / "cmdline").read_bytes().decode().strip("\0").split("\0")
        # Do not inspect cwd for unrelated same-user services (e.g. protected
        # processes). A candidate with unreadable cwd still fails closed.
        if scanning and not candidate_args(parts):
            return None
        status = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if status[0] == "Z":
            return None
        return {"pid": int(pid), "start": status[19],
                "args": parts,
                "cwd": str((root / "cwd").resolve()), "uid": uid}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as exc:
        if not scanning or candidate_args(parts):
            raise RuntimeError("Cannot verify target/candidate PID " + str(pid)
                               + "; no deletion performed: " + str(exc)) from exc
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
            info = process(item.name, scanning=True)
            role = role_of(info, repo)
            if role:
                found.append(dict(info, role=role))
    return found


def terminate(info, repo):
    current = process(info["pid"])
    if current is None:
        return
    require(current["start"] == info["start"] and role_of(current, repo) == info["role"], "PID identity changed")
    descriptor = None
    try:
        if callable(getattr(os, "pidfd_open", None)) and callable(getattr(signal, "pidfd_send_signal", None)):
            try:
                descriptor = os.pidfd_open(info["pid"])
            except OSError as exc:
                if exc.errno not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
                    raise

        def send(number):
            current = process(info["pid"])
            if current is None:
                return
            require(current["start"] == info["start"] and current["args"] == info["args"]
                    and role_of(current, repo) == info["role"], "PID identity changed before signaling")
            if descriptor is not None:
                try:
                    signal.pidfd_send_signal(descriptor, number)
                    return
                except OSError as exc:
                    if exc.errno != errno.ENOSYS:
                        raise
                    current = process(info["pid"])
                    if current is None:
                        return
                    require(current["start"] == info["start"] and current["args"] == info["args"]
                            and role_of(current, repo) == info["role"], "PID identity changed before fallback")
            # Older server Python/kernel: standard kill after checking identity
            # again immediately before EACH signal; never signal a whole group.
            os.kill(info["pid"], number)

        send(signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process(info["pid"]) is None:
                return
            time.sleep(.25)
        send(signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process(info["pid"]) is None:
                return
            time.sleep(.25)
        raise RuntimeError("Process did not exit; nothing deleted: " + str(info["pid"]))
    except ProcessLookupError:
        return
    finally:
        if descriptor is not None:
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
