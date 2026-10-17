"""Server-only background launch/status for one fixed R0 formal run.

Finds existing CPU/CUDA smoke artifacts by manifest identity. Preserves the
verified model worktree and writes a durable supervisor PID, training PID,
log, and exit code. Does not resume, retry, or choose by test performance.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def completed_smoke(value, device):
    return ((value.get("status"), value.get("exit_code"), value.get("mode"), value.get("epochs_completed"))
            == ("completed", 0, "smoke", 2) and value.get("optimizer_step_passed") is True
            and value.get("test") is None and value.get("binding", {}).get("runtime", {}).get("device_type") == device)


def active_training():
    result = subprocess.run(["pgrep", "-af", "[p]ython.*(train_grouped.py|run_r0_edge.py)"],
                            capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError("Cannot check current training processes: " + result.stderr)
    if result.stdout.strip():
        raise RuntimeError("Existing training process; no overlapping launch:\n" + result.stdout)


def artifacts(repo):
    e1 = read(repo / "outputs/grouped_full_E1_s42_v1/status.json")
    if (e1.get("status"), e1.get("exit_code")) != ("completed", 0):
        raise ValueError("E1 has not completed successfully")
    cpu = read(repo / "outputs/r0_cpu_smoke_server_v1/summary.json")
    if not completed_smoke(cpu, "cpu"):
        raise ValueError("Missing completed server CPU smoke")
    binding = cpu["binding"]
    candidates = set(repo.glob("outputs/r0_prepare_*/source_evidence.json"))
    candidates.update(repo.glob("*/outputs/r0_prepare_*/source_evidence.json"))
    preparations = []
    for path in candidates:
        summary_path = path.parent / "summary.json"
        if summary_path.is_file():
            value = read(summary_path)
            if (value.get("status"), value.get("exit_code"), value.get("manifest_id")) == ("completed", 0, binding["manifest_id"]):
                preparations.append(path.parent.resolve())
    if len(preparations) != 1:
        raise ValueError(f"Expected one matching preparation, found {preparations}")
    prep = preparations[0]
    evidence = read(prep / "source_evidence.json")
    code = Path(evidence["code_root"]).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    if commit != evidence["code_git_commit"]:
        raise ValueError("Verified model worktree commit has changed")
    for name, expected in binding["code"].items():
        if identity((code / name).read_text(encoding="utf-8-sig")) != expected:
            raise ValueError("Model source changed since CPU smoke: " + name)
    config = read(code / "configs/r0_edge.json")
    if identity(config) != binding["config_id"]:
        raise ValueError("R0 protocol config has changed")
    controls = {"epochs": 30, "batch_size": 64, "test_batch_size": 1, "lr": 0.01,
                "weight_decay": 0.0, "optimizer": "Adam", "selection": "last_epoch", "seed": 42,
                "scheduler": None, "claim": "protocol_reimplementation"}
    if any(config.get(k) != v for k, v in controls.items()):
        raise ValueError("Unexpected fixed R0 training controls")
    manifest = prep / "manifest"
    meta = read(manifest / "manifest.json")
    expected_counts = {"train": {str(i): 20000 for i in range(8)},
                       "test": {str(i): 1090 if i == 1 else 467 if i == 7 else 4000 for i in range(8)}}
    if meta["manifest_id"] != binding["manifest_id"] or meta["partition_label_counts"] != expected_counts:
        raise ValueError("Manifest identity or per-class counts differ")
    reports = []
    for path in repo.glob("outputs/r0_cuda_smoke_*/summary.json"):
        value = read(path)
        if not completed_smoke(value, "cuda"):
            continue
        other = value["binding"]
        if any(other[k] != binding[k] for k in ("manifest_id", "config_id", "code")):
            continue
        if sha(path.parent / "last.pt") != value.get("last_checkpoint_sha256"):
            raise ValueError("CUDA smoke checkpoint differs: " + str(path))
        tolerance = value["intervention_tolerance"]
        if len(value["gat_attribute_shapes"]) != 6 or any(v is None for v in value["gat_attribute_shapes"].values()):
            raise ValueError("CUDA smoke did not observe all GAT attribute inputs")
        if not any(v > tolerance for v in value["isolated_attribute_max_logprob_delta"].values()):
            raise ValueError("No CUDA smoke edge intervention exceeds jitter tolerance")
        if not any(v is not None and v > 0 for v in value["edge_projection_gradient_max"].values()):
            raise ValueError("No nonzero CUDA smoke edge gradient")
        reports.append((path.resolve(), value))
    if len(reports) != 1:
        raise ValueError(f"Expected one matching CUDA smoke; found {len(reports)}. No automatic choice or rerun.")
    smoke, value = reports[0]
    return {"code_root": str(code), "code_commit": commit, "manifest_dir": str(manifest),
            "manifest_id": binding["manifest_id"], "config_id": binding["config_id"],
            "smoke_report": str(smoke), "smoke_report_sha256": sha(smoke),
            "runtime_expected": value["binding"]["runtime"], "controls": controls}


def pointer(repo):
    return repo / "outputs/r0_full_launch.json"


def supervise(repo):
    record = read(pointer(repo))
    output = Path(record["output"])
    rc = 1
    try:
        # Recheck immediately before starting the sole training child.
        verified = artifacts(repo)
        if any(record[k] != verified[k] for k in verified):
            raise ValueError("Readiness evidence changed after launch submission")
        active_training()
        command = [record["python"], "-u", str(Path(record["code_root"]) / "scripts/run_r0_edge.py"),
                   "--mode", "full", "--manifest-dir", record["manifest_dir"],
                   "--smoke-report", record["smoke_report"], "--device", "cuda:0", "--workers", "0",
                   "--output", record["output"]]
        print("Starting one fixed R0 full run: 30 epochs, no automatic retry or resume.", flush=True)
        process = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL)
        output.with_suffix(".pid").write_text(str(process.pid) + "\n", encoding="utf-8")
        print(f"Training PID: {process.pid}", flush=True)
        rc = process.wait()
        print(f"R0 training process exit code: {rc}", flush=True)
    except BaseException:
        traceback.print_exc()
    finally:
        output.with_suffix(".exitcode").write_text(str(rc) + "\n", encoding="utf-8")
    return rc


def status(repo):
    record = read(pointer(repo))
    output = Path(record["output"])
    value = {"output": str(output), "log": str(output.with_suffix(".log")),
             "manifest_id": record["manifest_id"], "controls": record["controls"]}
    for key, path in (("pid", output.with_suffix(".pid")), ("supervisor_pid", output.with_suffix(".supervisor.pid")),
                      ("exit_code", output.with_suffix(".exitcode"))):
        value[key] = path.read_text().strip() if path.is_file() else None
    if value["pid"]:
        value["process_snapshot"] = subprocess.run(["ps", "-p", value["pid"], "-o", "pid,ppid,lstart,etime,args"],
                                                  capture_output=True, text=True).stdout.strip()
    path = output / "status.json"
    value["training_status"] = read(path) if path.is_file() else "not yet reported by training entry"
    for field, path in (("epochs_tail", output / "epochs.jsonl"), ("log_tail", output.with_suffix(".log"))):
        if path.is_file():
            with path.open(encoding="utf-8", errors="replace") as stream:
                value[field] = list(deque(stream, maxlen=3 if field == "epochs_tail" else 12))
    if (output / "summary.json").is_file():
        result = read(output / "summary.json")
        value["summary"] = {k: result.get(k) for k in ("status", "exit_code", "mode", "claim", "epochs_completed", "selection")}
        if result.get("test"):
            value["test"] = {k: result["test"][k] for k in ("samples", "accuracy", "macro_f1", "weighted_f1")}
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.status:
        status(repo)
        return 0
    if args.supervise:
        return supervise(repo)
    if pointer(repo).exists():
        status(repo)
        raise FileExistsError("A formal R0 launch is already recorded; no automatic second run. Use --status.")
    record = artifacts(repo)
    active_training()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = repo / "outputs" / ("r0_full_s42_" + stamp)
    record.update(schema="xgnid.r0.full-launch.v1", python=sys.executable, output=str(output),
                  submitted_utc=datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents two simultaneous launchers from both starting.
    with pointer(repo).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with output.with_suffix(".log").open("x", encoding="utf-8") as stream:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--repo", str(repo), "--supervise"],
                                   cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    output.with_suffix(".supervisor.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
    print(json.dumps({"launch_submission": "recorded; check status for actual training progress",
                      "supervisor_pid": process.pid, "output": str(output), "log": str(output.with_suffix(".log")),
                      "controls": record["controls"], "runtime_expected": record["runtime_expected"]},
                     indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
