"""Pause only the original E3 process and preserve a verified restart snapshot.

Server only. Never reboot, delete data, or signal unrelated jobs. A stopped
process cannot survive reboot; only the saved checkpoint will be resumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("Run on the Linux server only")
    repo = args.repo.resolve()
    run = repo / "outputs/grouped_full_E3_s42_v1"
    if read(run / "status.json").get("status") == "completed":
        print(json.dumps({"status": "already_completed", "action": "No pause needed; retain final results"}))
        return
    model = run / "E3"
    config = read(model / "config.json")
    if config["experiment"] != "E3" or config["mode"] != "full":
        raise RuntimeError("Wrong experiment identity")
    pointer = run.with_suffix(".restart.json")
    if pointer.exists():
        raise FileExistsError(f"Restart snapshot already recorded: {pointer}; inspect it instead of repeating pause")
    pid = int(run.with_suffix(".pid").read_text().strip())
    proc = Path("/proc") / str(pid)
    if proc.stat().st_uid != os.geteuid():
        raise RuntimeError("PID does not belong to the current user")
    argv = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    cwd = (proc / "cwd").resolve()
    if not any(Path(value).name == "train_grouped.py" for value in argv):
        raise RuntimeError("PID is not the frozen training entry; no signal sent")
    if "--output" not in argv or (cwd / argv[argv.index("--output") + 1]).resolve() != run:
        raise RuntimeError("PID output directory differs; no signal sent")
    if "--experiments" not in argv or argv[argv.index("--experiments") + 1] != "E3":
        raise RuntimeError("PID is not an E3-only run; no signal sent")
    birth = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
    paused = False
    try:
        if (proc / "stat").read_text().rsplit(")", 1)[1].split()[19] != birth:
            raise RuntimeError("PID identity changed")
        os.kill(pid, signal.SIGSTOP)
        paused = True
        for _ in range(100):
            state = next(line for line in (proc / "status").read_text().splitlines() if line.startswith("State:"))
            if state.split()[1] in ("T", "t"):
                break
            time.sleep(.05)
        else:
            raise RuntimeError("Could not confirm stopped process")
        snapshot = repo / "outputs" / ("e3_restart_snapshot_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        snapshot.mkdir(exist_ok=False)
        files = {}
        for filename in ("config.json", "last.pt", "best.pt", "epochs.jsonl"):
            source, target = model / filename, snapshot / filename
            if not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError(f"Missing checkpoint artifact: {source}")
            shutil.copy2(source, target)
            if sha(source) != sha(target):
                raise RuntimeError(f"Snapshot copy verification failed: {filename}")
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            files[filename] = sha(target)
        # Trusted local training checkpoints only; load to CPU, never create a CUDA context.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        last = torch.load(snapshot / "last.pt", map_location="cpu", weights_only=False)
        best = torch.load(snapshot / "best.pt", map_location="cpu", weights_only=False)
        required = {"model_state", "optimizer_state", "config", "epoch", "best_epoch", "best_val_macro_f1",
                    "validation", "sampler_rng_state", "torch_rng_state", "cuda_rng_states"}
        if not required.issubset(last) or last["config"] != config or best["config"] != config:
            raise RuntimeError("Checkpoint state/config is incomplete")
        if not (0 < best["epoch"] <= last["epoch"] <= config["epochs"]):
            raise RuntimeError("Invalid checkpoint epoch ordering")
        if best["epoch"] != last["best_epoch"] and last["best_epoch"] != last["epoch"]:
            raise RuntimeError("No checkpoint available for recorded best epoch")
        record = {"schema": "xgnid.e3-restart-snapshot.v1", "status": "snapshot_verified_process_stopped",
                  "created_utc": datetime.now(timezone.utc).isoformat(), "original_run": str(run),
                  "snapshot": str(snapshot), "paused_pid": pid, "pid_start_ticks": birth, "argv": argv,
                  "files_sha256": files, "saved_epoch": last["epoch"], "best_epoch": last["best_epoch"],
                  "stale_epochs": last["epoch"] - last["best_epoch"], "epoch_budget": config["epochs"],
                  "best_checkpoint_from_last": best["epoch"] != last["best_epoch"],
                  "gpu_context_created": False, "reboot_performed": False,
                  "note": "Only the saved completed epoch is recoverable; current partial epoch is discarded. Other jobs were not stopped."}
        for target in (snapshot / "snapshot.json", pointer):
            with target.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
        os.sync()
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        if paused and proc.exists() and (proc / "stat").read_text().rsplit(")", 1)[1].split()[19] == birth:
            os.kill(pid, signal.SIGCONT)
        raise


if __name__ == "__main__":
    main()
