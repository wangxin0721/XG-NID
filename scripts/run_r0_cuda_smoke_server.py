"""Launch only the server CUDA smoke using existing, verified CPU artifacts.

Uses no shell variables. Locates the preparation by the CPU smoke manifest ID,
including the nonstandard worktree directory names observed in the handoff.
Never launches full training or modifies the original E0--E3 worktree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def preparation(repo, manifest_id):
    # Avoid walking through all graph shards. Both known output layouts are
    # checked explicitly; never choose a different manifest by latest timestamp.
    candidates = set(repo.glob("outputs/r0_prepare_*/source_evidence.json"))
    candidates.update(repo.glob("*/outputs/r0_prepare_*/source_evidence.json"))
    matches = []
    for evidence_path in candidates:
        directory = evidence_path.parent
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            continue
        summary = read(summary_path)
        if (summary.get("status"), summary.get("exit_code"), summary.get("manifest_id")) == ("completed", 0, manifest_id):
            matches.append(directory.resolve())
    if len(matches) != 1:
        raise ValueError(f"Expected one completed preparation matching CPU manifest; found {matches}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    cpu = read(repo / "outputs/r0_cpu_smoke_server_v1/summary.json")
    if (cpu.get("status"), cpu.get("exit_code"), cpu.get("mode"), cpu.get("epochs_completed")) != ("completed", 0, "smoke", 2):
        raise ValueError("The existing server CPU smoke did not complete two epochs")
    if cpu["binding"]["runtime"]["device_type"] != "cpu" or cpu.get("optimizer_step_passed") is not True:
        raise ValueError("Unexpected CPU smoke runtime or optimizer status")
    manifest_id = cpu["binding"]["manifest_id"]
    prep = preparation(repo, manifest_id)
    evidence = read(prep / "source_evidence.json")
    code = Path(evidence["code_root"]).resolve()
    manifest = prep / "manifest"
    if read(manifest / "manifest.json")["manifest_id"] != manifest_id:
        raise ValueError("Preparation manifest differs from the verified CPU run")
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    if actual_commit != evidence["code_git_commit"]:
        raise ValueError("Preparation worktree commit has changed")
    runner = code / "scripts/run_r0_edge.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    active = subprocess.run(["pgrep", "-af", "[p]ython.*(train_grouped.py|run_r0_edge.py)"],
                            capture_output=True, text=True)
    if active.returncode not in (0, 1):
        raise RuntimeError(f"Could not check active training processes: {active.stderr}")
    if active.stdout.strip():
        raise RuntimeError("Training processes are still present; do not overlap GPU jobs:\n" + active.stdout)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = repo / "outputs" / ("r0_cuda_smoke_" + stamp)
    log = output.with_suffix(".log")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    print(f"Code: {code}\nManifest: {manifest}\nOutput: {output}\nLog: {log}", flush=True)
    print("Starting CUDA smoke only; no full training. Loading the frozen manifest may take a moment.", flush=True)
    command = [sys.executable, "-u", str(runner), "--mode", "smoke", "--manifest-dir", str(manifest),
               "--device", "cuda:0", "--workers", "0", "--output", str(output)]
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        print(f"CUDA smoke PID: {process.pid}", flush=True)
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            rc = process.wait()
        except BaseException:
            # This launcher owns only this newly started smoke child.
            process.terminate()
            process.wait()
            raise
    output.with_suffix(".exitcode").write_text(str(rc) + "\n", encoding="utf-8")
    print(f"CUDA smoke exit code: {rc}\nSummary: {output / 'summary.json'}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
