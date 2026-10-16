"""Controlled pilot/full training on a frozen grouped-input benchmark.

Pilot never loads test graphs. Full runs require successful smoke and pilot
reports. Each sample is read by explicit path and verified against its hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import torch_geometric
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xgnid.edge_experiments import build_experiment, validate_graph
from build_grouped_split import LABELS, sha
from grouped_graphs import FrozenGraphs
from smoke_grouped_gpu import initialize, seed_all, state_hash, write


class GraphRows(Dataset):
    def __init__(self, frozen, rows):
        self.frozen, self.rows = frozen, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        graph = self.frozen.load(self.rows[index])
        validate_graph(graph)
        return graph


class TrainBatches(Sampler):
    """Shuffle once per epoch; merge a final singleton for Flow BatchNorm."""
    def __init__(self, size, batch_size, seed):
        if size < 2 or batch_size < 2:
            raise ValueError("At least two training samples and batch-size >= 2 are required")
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.last_order_sha256 = None

    def __len__(self):
        return math.ceil(self.size / self.batch_size) - int(self.size % self.batch_size == 1)

    def __iter__(self):
        order = torch.randperm(self.size, generator=self.generator).tolist()
        self.last_order_sha256 = hashlib.sha256(json.dumps(order).encode()).hexdigest()
        offset = 0
        while offset < self.size:
            stop = self.size if self.size - offset == self.batch_size + 1 else offset + self.batch_size
            yield order[offset:stop]
            offset = stop


def select_rows(frozen, part, limit=None):
    counts, result = Counter(), []
    for row in frozen.rows:
        if row["partition"] == part and (limit is None or counts[row["label"]] < limit):
            result.append(row)
            counts[row["label"]] += 1
    if set(counts) != set(range(8)):
        raise ValueError(f"{part} must contain all eight classes")
    return result


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, loader, device, export_rows=False):
    model.eval()
    labels, predictions, exports, gates, events = [], [], [], [], []
    cpu_forward_seconds = 0.
    sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if device.type == "cuda":
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
            tick = time.perf_counter()
            scores = model(batch)
            cpu_forward_seconds += time.perf_counter() - tick
            if device.type == "cuda":
                end_event.record()
                events.append((start_event, end_event))
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite evaluation predictions")
            truth, predicted = batch.y.view(-1).tolist(), scores.argmax(-1).tolist()
            labels.extend(truth)
            predictions.extend(predicted)
            weights = getattr(model, "_last_cache", {}).get("gate_weights")
            weights = weights.detach().cpu().tolist() if weights is not None else None
            if weights is not None:
                gates.extend(weights)
            if export_rows:
                for i, sample_id in enumerate(batch.sample_id):
                    exports.append({"sample_id": sample_id, "label": truth[i], "prediction": predicted[i],
                                    "flow_weight": weights[i][0] if weights else "",
                                    "packet_weight": weights[i][1] if weights else ""})
    sync(device)
    wall = time.perf_counter() - started
    metrics = {"accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(8)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=list(range(8)), average="weighted", zero_division=0)),
        "per_class": classification_report(labels, predictions, labels=list(range(8)), target_names=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(8))).tolist(),
        "samples": len(labels), "wall_seconds_including_graph_io": wall,
        "forward_seconds": sum(a.elapsed_time(b) for a, b in events) / 1000 if events else cpu_forward_seconds,
        "forward_timing_method": "CUDA events around model call" if events else "CPU perf_counter around model call"}
    if gates:
        tensor = torch.tensor(gates)
        metrics["gate_weights"] = {"mean": tensor.mean(0).tolist(), "std": tensor.std(0, unbiased=False).tolist(),
            "min": tensor.min(0).values.tolist(), "max": tensor.max(0).values.tolist(),
            "fraction_flow_above_0_95": float((tensor[:, 0] > .95).float().mean()),
            "fraction_packet_above_0_95": float((tensor[:, 1] > .95).float().mean())}
    return metrics, exports


def checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def verify_training_artifact(report_path, report, name):
    config_path = report_path.parent / name / "config.json"
    if sha(config_path) != report["models"][name]["config_sha256"]:
        raise ValueError(f"{name} training config hash mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for path, digest in config["source_hashes"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Training code changed after {name}: {path}")


def readiness(args, frozen, controls):
    smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    smoke_config = json.loads((args.smoke_report.parent / "config.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "completed" or smoke.get("split_id") != frozen.metadata["split_id"]:
        raise ValueError("Successful smoke on the same split is required")
    if not all(smoke.get("paired_initial_states_identical", {}).get(pair) is True for pair in ("E0/E1", "E2/E3")):
        raise ValueError("Smoke did not verify paired initial states")
    for name in ("E0", "E1", "E2", "E3"):
        if not smoke.get("models", {}).get(name, {}).get("parameter_update_verified"):
            raise ValueError("Smoke did not verify all four optimizers")
    if torch.device(smoke["device"]).type != torch.device(args.device).type:
        raise ValueError("Smoke device family does not match training; no CPU/GPU substitution")
    for path, digest in smoke_config["source_hashes"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Model/data code changed after smoke: {path}; rerun smoke")
    if args.mode == "full":
        if args.pilot_report is None:
            raise ValueError("Full training requires --pilot-report from a completed E0/E1 pilot")
        pilot = json.loads(args.pilot_report.read_text(encoding="utf-8"))
        if pilot.get("status") != "completed" or pilot.get("mode") != "pilot" or pilot.get("split_id") != frozen.metadata["split_id"]:
            raise ValueError("Pilot identity/status mismatch")
        if pilot.get("controls") != controls or not all(name in pilot.get("models", {}) for name in ("E0", "E1")):
            raise ValueError("Pilot must cover E0/E1 with identical training controls")
        for name in ("E0", "E1"):
            verify_training_artifact(args.pilot_report, pilot, name)
        if any(name != "E0" for name in args.experiments) and "E0" not in args.experiments:
            if args.baseline_report is None:
                raise ValueError("Full E1/E2/E3 requires a completed full E0 --baseline-report")
            baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
            if baseline.get("status") != "completed" or baseline.get("mode") != "full" or baseline.get("split_id") != frozen.metadata["split_id"] or "E0" not in baseline.get("models", {}):
                raise ValueError("Full E0 baseline identity/status mismatch")
            if baseline.get("controls") != controls or baseline.get("epochs") != args.epochs:
                raise ValueError("Full E0 baseline training controls/budget differ")
            verify_training_artifact(args.baseline_report, baseline, "E0")
        if "E3" in args.experiments and "E2" not in args.experiments:
            if args.fusion_control_report is None:
                raise ValueError("Full E3 requires completed E2 --fusion-control-report")
            control = json.loads(args.fusion_control_report.read_text(encoding="utf-8"))
            if control.get("status") != "completed" or control.get("mode") != "full" or control.get("split_id") != frozen.metadata["split_id"] or "E2" not in control.get("models", {}):
                raise ValueError("Full E2 fusion control identity/status mismatch")
            if control.get("controls") != controls or control.get("epochs") != args.epochs:
                raise ValueError("Full E2/E3 training controls/budget differ")
            verify_training_artifact(args.fusion_control_report, control, "E2")
    return smoke, smoke_config


def train_one(name, args, frozen, rows, device, controls, smoke, smoke_config):
    output = args.output / name
    output.mkdir()
    write(output / "status.json", {"status": "running"})
    seed_all(args.seed)
    warm = [frozen.load(row) for row in frozen.training_candidates()]
    multi = [i for i, graph in enumerate(warm) if graph["packet"].x.size(0) >= 3]
    if len(multi) < 2:
        raise ValueError("No suitable initialization batch; preserve smoke batch selection")
    order = multi[:2] + [i for i in range(len(warm)) if i not in multi[:2]]
    batch = Batch.from_data_list([warm[i] for i in order[:args.batch_size]]).to(device)
    model = build_experiment(name, warm[0]).to(device)
    initialize(model, name, batch)
    initial = state_hash(model.state_dict().items())
    if args.seed == smoke_config["seed"] and initial != smoke["models"][name]["initial_state_sha256"]:
        raise ValueError("Training initialization differs from verified smoke")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sampler = TrainBatches(len(rows["train"]), args.batch_size, args.seed)
    loader_kwargs = {"num_workers": args.workers, "pin_memory": device.type == "cuda",
                     "persistent_workers": args.workers > 0}
    train_loader = DataLoader(GraphRows(frozen, rows["train"]), batch_sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(GraphRows(frozen, rows["val"]), batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    config = {"experiment": name, "mode": args.mode, "controls": controls, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "epochs": args.epochs, "workers": args.workers, "use_edge_attr": name != "E1",
        "fusion": "equal_logits" if name == "E2" else "adaptive_logits" if name == "E3" else None,
        "split_id": frozen.metadata["split_id"], "sample_manifest_sha256": frozen.metadata["sample_manifest_sha256"],
        "data_root": str(frozen.root), "split_directory": str(frozen.directory),
        "source_manifest_sha256": frozen.metadata["source_manifest_sha256"], "protocol": frozen.metadata["protocol"],
        "initial_state_sha256": initial, "torch": torch.__version__, "pyg": torch_geometric.__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": {p: sha(ROOT / p) for p in ("scripts/train_grouped.py", "scripts/grouped_graphs.py",
            "scripts/smoke_grouped_gpu.py", "src/xgnid/model.py", "src/xgnid/edge_experiments.py", "third_party/GNN4ID/Utility/Model.py")},
        "selected_partition_sizes": {part: len(values) for part, values in rows.items()},
        "selected_label_counts": {part: dict(Counter(row["label"] for row in values)) for part, values in rows.items()},
        "train_sample_ids_sha256": hashlib.sha256(json.dumps([r["sample_id"] for r in rows["train"]]).encode()).hexdigest(),
        "val_sample_ids_sha256": hashlib.sha256(json.dumps([r["sample_id"] for r in rows["val"]]).encode()).hexdigest(),
        "gradient_monitor": "clip_grad_norm_(infinity,error_if_nonfinite=True); no finite gradient clipping",
        "singleton_batch": "merge into preceding batch; no training sample dropped",
        "smoke_report_sha256": sha(args.smoke_report),
        "scope": "diagnostic subset; no test evaluation" if args.mode == "pilot" else "new grouped-input benchmark; flow independence unverified"}
    write(output / "config.json", config)
    best, best_epoch, stale, history = -1., 0, 0, []
    wall_started = time.perf_counter()
    with (output / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            sync(device)
            tick = time.perf_counter()
            loss_sum, samples = 0., 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.nll_loss(model(batch), batch.y.view(-1))
                if not torch.isfinite(loss):
                    raise ValueError(f"{name}: nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * batch.num_graphs
                samples += batch.num_graphs
            sync(device)
            train_seconds = time.perf_counter() - tick
            if samples != len(rows["train"]):
                raise ValueError("Training did not consume exactly its selected samples")
            val, _ = evaluate(model, val_loader, device)
            entry = {"epoch": epoch, "train_loss": loss_sum / samples, "train_samples": samples,
                "train_seconds": train_seconds, "train_order_sha256": sampler.last_order_sha256,
                "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "val_weighted_f1": val["weighted_f1"],
                "val_wall_seconds": val["wall_seconds_including_graph_io"],
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
            history.append(entry)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            improved = val["macro_f1"] > best
            if improved:
                best, best_epoch, stale = val["macro_f1"], epoch, 0
            else:
                stale += 1
            payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "config": config, "epoch": epoch, "best_epoch": best_epoch, "best_val_macro_f1": best,
                "validation": val, "sampler_rng_state": sampler.generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
            checkpoint(output / "last.pt", payload)
            if improved:
                checkpoint(output / "best.pt", payload)
            print(name, json.dumps(entry), flush=True)
            if stale >= args.patience:
                break
    best_checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    metrics = {"mode": args.mode, "best_epoch": best_epoch, "stopped_epoch": history[-1]["epoch"],
        "best_validation": best_checkpoint["validation"], "training_validation_checkpoint_wall_seconds": time.perf_counter() - wall_started,
        "parameter_count": sum(p.numel() for p in model.parameters()), "initial_state_sha256": initial}
    if args.mode == "full":
        test_loader = DataLoader(GraphRows(frozen, rows["test"]), batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        metrics["test"], predictions = evaluate(model, test_loader, device, export_rows=True)
        with (output / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "label", "prediction", "flow_weight", "packet_weight"])
            writer.writeheader()
            writer.writerows(predictions)
    else:
        metrics["test"] = None
    metrics["run_wall_seconds"] = time.perf_counter() - wall_started
    metrics["config_sha256"] = sha(output / "config.json")
    write(output / "metrics.json", metrics)
    write(output / "status.json", {"status": "completed", "exit_code": 0})
    return metrics


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "status.json", {"status": "running"})
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; no CPU fallback")
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        controls = {"seed": args.seed, "batch_size": args.batch_size, "lr": args.lr, "hidden_dim": 64,
            "optimizer": "Adam", "early_stop_patience": args.patience, "early_stop_min_delta": 0.,
            "auxiliary_losses": False, "calibration": False, "class_weighting": False, "hard_negatives": False,
            "loss": "plain NLL", "device_family": device.type}
        frozen = FrozenGraphs(args.split_dir, args.expected_split_id, args.data_root)
        smoke, smoke_config = readiness(args, frozen, controls)
        rows = {part: select_rows(frozen, part, args.pilot_train_per_class if part == "train" else args.pilot_val_per_class)
                for part in ("train", "val")} if args.mode == "pilot" else {
                    part: select_rows(frozen, part) for part in ("train", "val", "test")}
        results = {}
        for name in sorted(set(args.experiments)):
            print(f"Starting {args.mode} {name} on {device}; partitions={ {k:len(v) for k,v in rows.items()} }", flush=True)
            results[name] = train_one(name, args, frozen, rows, device, controls, smoke, smoke_config)
            write(args.output / "summary_partial.json", results)
        for left, right in (("E0", "E1"), ("E2", "E3")):
            if left in results and right in results and results[left]["initial_state_sha256"] != results[right]["initial_state_sha256"]:
                raise ValueError("Paired training initial states differ")
        report = {"status": "completed", "exit_code": 0, "mode": args.mode,
            "split_id": frozen.metadata["split_id"], "controls": controls, "epochs": args.epochs, "models": results}
        write(args.output / "summary.json", report)
        write(args.output / "status.json", {"status": "completed", "exit_code": 0})
        return report
    except BaseException as exc:
        (args.output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        write(args.output / "status.json", {"status": "failed", "exit_code": exit_code})
        for name in args.experiments:
            status = args.output / name / "status.json"
            if status.exists() and json.loads(status.read_text()).get("status") == "running":
                write(status, {"status": "failed", "exit_code": exit_code, "error_log": "../error.log"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--expected-split-id", required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--fusion-control-report", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--experiments", nargs="+", choices=["E0", "E1", "E2", "E3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pilot-train-per-class", type=int, default=32)
    parser.add_argument("--pilot-val-per-class", type=int, default=16)
    args = parser.parse_args()
    args.epochs = args.epochs if args.epochs is not None else (2 if args.mode == "pilot" else 100)
    args.experiments = args.experiments or (["E0", "E1"] if args.mode == "pilot" else ["E0"])
    if min(args.epochs, args.patience, args.pilot_train_per_class, args.pilot_val_per_class) < 1 or args.batch_size < 2 or args.workers < 0 or args.lr <= 0:
        parser.error("Invalid positive training budget/configuration")
    run(args)
