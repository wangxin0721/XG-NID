"""Read-only CPU F1 monitor for completed R0 checkpoints, never the test set.

Run on the server. Each result uses ALL source-training graphs and a stable
in-memory checkpoint snapshot. Slow evaluations may skip intermediate epochs.
No training source, checkpoint, launch record, or experiment output is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def version(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def snapshot(path):
    # The original trainer overwrites last.pt directly. Reject concurrent writes;
    # deserialize immutable bytes, never a live file that training may truncate.
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            data = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat()
    except FileNotFoundError:
        return None
    if not data or len(data) != before.st_size or not version(before) == version(after) == version(current):
        return None
    # torch.save uses a ZIP archive; its directory is finalized at save end.
    # A briefly paused writer can have stable stat values but an incomplete ZIP.
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return None
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--interval", type=int, default=60, help="Polling seconds between evaluations (minimum 10)")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--once", action="store_true", help="Evaluate the latest complete checkpoint once")
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("--interval must be at least 10")

    # Set before importing torch. This observer never allocates GPU memory.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    sys.dont_write_bytecode = True
    if hasattr(os, "nice"):
        os.nice(10)

    repo = args.repo.resolve()
    launch = read(repo / "outputs/r0_full_launch.json")
    output = Path(launch["output"])
    code = Path(launch["code_root"]).resolve()
    sys.path.insert(0, str(code / "scripts"))
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader
    import run_r0_edge as runner

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    if runner.ROOT.resolve() != code:
        raise ValueError("Monitor imported the wrong R0 worktree")
    config = read(output / "config.json")
    binding = config["binding"]
    if config.get("mode") != "full" or any(binding[k] != launch[k] for k in ("manifest_id", "config_id")):
        raise ValueError("Formal run and launch identities differ")

    def verify_code():
        if runner.code_hashes() != binding["code"]:
            raise ValueError("R0 source changed since training; refuse mismatched evaluation")

    verify_code()
    print(f"{stamp()} CPU observer starting; full TRAIN set only, no test access.", flush=True)
    train = runner.R0Graphs(Path(launch["manifest_dir"]), "train")
    if train.meta["manifest_id"] != binding["manifest_id"] or len(train) != 160000:
        raise ValueError("Unexpected R0 training manifest/count")
    model = runner.ReferenceEdge(train[0], use_edge_attr=True).to(device)
    model.eval()
    # Materialize lazy reference layers before strict checkpoint loading.
    with torch.no_grad():
        model(Batch.from_data_list([train[i] for i in range(16)]))
    loader = DataLoader(train, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    last_epoch = 0
    while True:
        raw = snapshot(output / "last.pt")
        if raw is not None:
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            epoch = payload["epoch"]
            if payload.get("binding") != binding:
                raise ValueError("Checkpoint binding differs from this formal run")
            if payload.get("config") != runner.configuration(code / "configs/r0_edge.json"):
                raise ValueError("Checkpoint protocol configuration differs")
            if epoch < last_epoch:
                raise ValueError("Checkpoint epoch moved backwards")
            if epoch > last_epoch:
                verify_code()
                checkpoint_sha = hashlib.sha256(raw).hexdigest()
                model.load_state_dict(payload["model_state"], strict=True)
                del payload, raw
                model.eval()
                truth, predictions = [], []
                tick = time.monotonic()
                print(f"{stamp()} Evaluating TRAIN epoch={epoch}: 0/{len(train)} samples; CPU only.", flush=True)
                with torch.no_grad():
                    for step, graph in enumerate(loader, 1):
                        logprob = model(graph)
                        if not torch.isfinite(logprob).all():
                            raise ValueError("Nonfinite monitor output")
                        truth.extend(graph.y.reshape(-1).tolist())
                        predictions.extend(logprob.argmax(-1).tolist())
                        if step % 100 == 0:
                            print(f"{stamp()} TRAIN epoch={epoch} evaluated={len(truth)}/{len(train)}", flush=True)
                if len(truth) != len(train):
                    raise ValueError("Incomplete monitor training evaluation")
                result = runner.metrics(truth, predictions)
                print(json.dumps({"observed_at": stamp(), "checkpoint_epoch": epoch,
                    "partition": "train", "samples": result["samples"], "device": "cpu",
                    "train_accuracy": result["accuracy"], "train_macro_f1": result["macro_f1"],
                    "train_weighted_f1": result["weighted_f1"],
                    "checkpoint_sha256": checkpoint_sha, "evaluation_seconds": time.monotonic() - tick},
                    ensure_ascii=False), flush=True)
                print(f"TRAIN epoch={epoch} Accuracy={result['accuracy']:.2%} "
                      f"Macro-F1={result['macro_f1']:.2%} Weighted-F1={result['weighted_f1']:.2%}", flush=True)
                last_epoch = epoch
                if args.once:
                    return 0
                # Pick up a newer checkpoint immediately if training advanced
                # during this CPU pass, including the final completed epoch.
                continue
        state = read(output / "status.json")
        if state.get("status") in ("completed", "failed"):
            if state["status"] == "failed":
                print(f"{stamp()} Training failed; monitor stops. Check the original R0 log.", flush=True)
                return 1
            summary = read(output / "summary.json")
            if last_epoch == summary.get("epochs_completed"):
                print(f"{stamp()} Final TRAIN checkpoint evaluated; monitor finished.", flush=True)
                return 0
        print(f"{stamp()} Waiting for a new complete checkpoint; last displayed epoch={last_epoch}.", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCPU F1 observer stopped; the separate R0 training process is unaffected.", flush=True)
