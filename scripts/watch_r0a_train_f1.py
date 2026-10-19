"""Server-only CPU TRAIN F1 observer for the active R0-A v2.

Captures stable last.pt snapshots in a separate folder while evaluating queued
epochs. Never changes the trainer, uses GPU, loads test graphs or selects models.
Epochs overwritten before capture cannot be recovered and are explicitly listed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import zipfile


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def stable_snapshot(path):
    def version(st):
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            data = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat()
    except FileNotFoundError:
        return None
    if (not data or len(data) != before.st_size or not version(before) == version(after) == version(current)
            or not zipfile.is_zipfile(io.BytesIO(data))):
        return None
    return data


def within(root, path):
    result = (root / path).resolve()
    result.relative_to(root)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Run only on the Linux server")
    if args.interval < 10:
        parser.error("Use --interval >= 10 seconds")
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    sys.dont_write_bytecode = True
    os.nice(10)
    repo = args.repo.resolve()
    output = repo / "outputs/r0_aligned_full_s42_v2"
    control = repo / "outputs/r0_aligned_restart_control_v2"
    observer = repo / "outputs/r0_aligned_train_f1_v2"
    observer.mkdir(exist_ok=True)
    import fcntl
    lock = (observer / "observer.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("An R0-A F1 observer is already running")
    config = read(output / "config.json")
    binding = config["binding"]
    launch = read(repo / "outputs/r0_full_launch.json")
    code = within(repo, launch["code_root"])
    manifest = within(repo, launch["manifest_dir"])
    if (config["mode"] != "full" or config["samples"] != 160000 or binding["manifest_id"] != launch["manifest_id"]
            or binding["experiment"]["experiment"] != "R0-A"):
        raise ValueError("Unexpected R0-A run/data binding")
    sys.path.insert(0, str(code / "scripts"))
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader
    import run_r0_edge as base
    if base.ROOT.resolve() != code:
        raise ValueError("Wrong frozen worktree imported")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    def verify_code():
        if base.code_hashes() != binding["base_code"]:
            raise ValueError("Frozen model/data source changed")
        if identity(config["base_config"]) != binding["experiment"]["base_config_id"]:
            raise ValueError("Base protocol changed")

    verify_code()
    training_source = code / "third_party/GNN4ID/Utility/Training.py"
    if identity(training_source.read_text(encoding="utf-8-sig")) != binding["training_source_identity"]:
        raise ValueError("Official training source changed")
    record = {"binding": binding, "observed_output": str(output), "partition": "train", "device": "cpu",
              "note": "Independent full TRAIN evaluation, not test or validation; can lag training"}
    record_path = observer / "observer.json"
    if record_path.exists() and read(record_path) != record:
        raise ValueError("Observer folder belongs to another run")
    if not record_path.exists():
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshots = observer / "checkpoints"
    snapshots.mkdir(exist_ok=True)
    tasks, errors = queue.Queue(), queue.Queue()
    stop, finished = threading.Event(), threading.Event()
    captured = {}
    for path in sorted(snapshots.glob("epoch_*.pt")):
        epoch = int(path.stem.split("_")[1])
        captured[epoch] = hashlib.sha256(path.read_bytes()).hexdigest()
        tasks.put((epoch, path))

    def collect():
        try:
            while not stop.is_set():
                raw = stable_snapshot(output / "last.pt")
                if raw is not None:
                    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
                    epoch = payload["epoch"]
                    if payload.get("binding") != binding or type(epoch) is not int or not 1 <= epoch <= 30:
                        raise ValueError("Checkpoint binding/epoch differs")
                    digest = hashlib.sha256(raw).hexdigest()
                    if epoch in captured:
                        if captured[epoch] != digest:
                            raise ValueError("Already captured checkpoint changed for the same epoch")
                    else:
                        if captured and epoch < max(captured):
                            raise ValueError("Checkpoint epoch moved backwards")
                        path = snapshots / f"epoch_{epoch:02d}.pt"
                        temporary = path.with_suffix(".tmp")
                        temporary.write_bytes(raw)
                        temporary.replace(path)
                        captured[epoch] = digest
                        tasks.put((epoch, path))
                        print(f"Captured TRAIN checkpoint epoch={epoch}; queued for CPU evaluation", flush=True)
                    del payload, raw
                supervisor = read(control / "status.json")
                if supervisor.get("status") in ("completed", "failed"):
                    break
                stop.wait(args.interval)
        except BaseException as exc:
            errors.put(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=collect, name="training-checkpoint-observer", daemon=True)
    thread.start()
    try:
        print("CPU observer: all 160000 TRAIN graphs only; no test access; no optimizer steps.", flush=True)
        train = base.R0Graphs(manifest, "train")
        if len(train) != 160000 or train.meta["manifest_id"] != binding["manifest_id"]:
            raise ValueError("Training manifest differs")
        # Independent evaluation uses fixed numeric order and no shuffled loader.
        train.rows.sort(key=lambda row: int(Path(row["path"]).stem.split("_")[-1]))
        model = base.ReferenceEdge(train[0], use_edge_attr=True).eval()
        with torch.no_grad():
            model(Batch.from_data_list([train[i] for i in range(16)]))
        loader = DataLoader(train, batch_size=64, shuffle=False, num_workers=0)
        results_path = observer / "train_f1.jsonl"
        evaluated = {}
        if results_path.is_file():
            for line in results_path.read_text(encoding="utf-8").splitlines():
                result = json.loads(line)
                if result["partition"] != "train" or result["device"] != "cpu":
                    raise ValueError("Unexpected existing observer results")
                evaluated[result["epoch"]] = result["checkpoint_sha256"]
        while True:
            if not errors.empty():
                raise errors.get()
            try:
                epoch, path = tasks.get(timeout=1)
            except queue.Empty:
                if finished.is_set():
                    break
                continue
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if epoch in evaluated:
                if evaluated[epoch] != digest:
                    raise ValueError("Existing F1 result has different checkpoint SHA")
                continue
            verify_code()
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            if payload["binding"] != binding or payload["epoch"] != epoch:
                raise ValueError("Queued checkpoint differs")
            # Official R0-A stores unwrapped keys. Adapter delegates to the same
            # locked official Edge module; only the state-dictionary prefix differs.
            model.load_state_dict({"reference." + k: v for k, v in payload["model_state"].items()}, strict=True)
            del payload, raw
            model.eval()
            truth, predictions = [], []
            tick = time.monotonic()
            print(f"Evaluating TRAIN epoch={epoch}: 0/160000", flush=True)
            with torch.no_grad():
                for step, batch in enumerate(loader, 1):
                    values = model(batch)
                    if not torch.isfinite(values).all():
                        raise ValueError("Nonfinite CPU evaluation output")
                    truth.extend(batch.y.reshape(-1).tolist())
                    predictions.extend(values.argmax(-1).tolist())
                    if step % 100 == 0:
                        print(f"TRAIN epoch={epoch} evaluated={len(truth)}/160000", flush=True)
            if len(truth) != 160000:
                raise ValueError("Incomplete training-set evaluation")
            scores = base.metrics(truth, predictions)
            result = {"epoch": epoch, "partition": "train", "device": "cpu", "samples": 160000,
                      "train_accuracy": scores["accuracy"], "train_macro_f1": scores["macro_f1"],
                      "train_weighted_f1": scores["weighted_f1"], "checkpoint_sha256": digest,
                      "evaluation_seconds": time.monotonic() - tick}
            with results_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
            evaluated[epoch] = digest
            print(f"TRAIN epoch={epoch} Macro-F1={scores['macro_f1']:.4%} Accuracy={scores['accuracy']:.4%}", flush=True)
        if not errors.empty():
            raise errors.get()
        end = max(captured, default=0)
        missing = sorted(set(range(1, end + 1)) - set(captured))
        print(json.dumps({"observer_finished": True, "captured_epochs": sorted(captured),
                          "evaluated_epochs": sorted(evaluated), "missing_epochs_before_last_capture": missing,
                          "note": "Missing checkpoints cannot be reconstructed; this is not test F1"}), flush=True)
        return 0
    finally:
        stop.set()
        thread.join(timeout=5)
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("CPU F1 observer stopped; separate R0-A training is unaffected.", flush=True)
