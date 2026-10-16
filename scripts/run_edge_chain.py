"""Prepare a local pilot split, smoke-test, and run controlled E0--E4 trials.

Use the nids Python environment. Existing output directories are never reused.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from xgnid.graph_building import build_graph_from_row, PACKET_COLUMNS
from xgnid.edge_experiments import build_experiment, validate_graph, RELATIONS

LABELS = ["Benign", "WebBased", "Spoofing", "Recon", "Mirai", "Dos", "DDos", "BruteForce"]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def prepare(args):
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    graphs, records, content_seen = [], [], set()
    split = {"train": [], "val": [], "test": []}
    for source in audit["sources"]:
        path = Path(source["path"])
        if sha(path) != source["sha256"]:
            raise ValueError("Source changed after audit")
        is_test = path.stem.endswith("_test")
        counts = Counter()
        source_indices = []
        # This is deliberately a bounded pilot, not a full-data reproduction.
        for frame in pd.read_csv(path, chunksize=256):
            for row_index, row in frame.iterrows():
                label = int(row["Label"])
                if counts[label] >= args.per_class:
                    continue
                values = {k: ast.literal_eval(str(row[k])) for k in PACKET_COLUMNS}
                n = len(values["udps.payload_data"])
                if n == 0 or any(len(v) != n for v in values.values()):
                    raise ValueError(f"Invalid packet field alignment: {path}:{row_index}")
                # Exclude labels from identity to detect contradictory duplicates too.
                fingerprint = hashlib.sha256(json.dumps(
                    {k: str(v) for k, v in row.items() if k != "Label"}, sort_keys=True).encode()).hexdigest()
                if fingerprint in content_seen:
                    continue
                graph = build_graph_from_row(row)
                validate_graph(graph)
                sample_id = f"{source['sha256']}:{row_index}"
                graph.sample_id = sample_id
                graphs.append(graph)
                source_indices.append(len(graphs) - 1)
                records.append({"sample_id": sample_id, "content_hash": fingerprint,
                                "source": path.name, "row_index": int(row_index), "label": label})
                content_seen.add(fingerprint)
                counts[label] += 1
            if len(counts) == 8 and min(counts.values()) >= args.per_class:
                break
        if set(counts) != set(range(8)) or min(counts.values()) < args.per_class:
            raise ValueError(f"Insufficient samples: {counts}")
        if is_test:
            split["test"] = source_indices
        else:
            split["train"], split["val"] = train_test_split(
                source_indices, test_size=0.25, random_state=args.seed,
                stratify=[records[i]["label"] for i in source_indices])
    if any(not v for v in split.values()):
        raise ValueError("Empty split")
    torch.save(graphs, args.output / "graphs.pt")
    write(args.output / "split.json", {"split_indices": split,
          "samples": records, "seed": args.seed, "label_mapping": dict(enumerate(LABELS)),
          "split_label_counts": {k: dict(Counter(records[i]["label"] for i in ids)) for k, ids in split.items()},
          "source_audit": audit, "scope": "bounded pilot; new validation split; not historical reproduction",
          "flow_leakage_status": "unverified: capture/flow identifiers unavailable",
          "edge_fields": {"contain/rev_contain": ["direction", "ip_size", "transport_size", "payload_size"],
                          "link": ["delta_time[1:] (extractor packet.delta_time)"]},
          "normalization": "none (existing preprocessed features)",
          "code_hashes": {p: sha(ROOT / p) for p in ["src/xgnid/graph_building.py", "scripts/run_edge_chain.py"]}})


def smoke(graphs, output):
    # Use actual multi-packet samples with nonidentical packet features.
    candidates = [g for g in graphs if g["packet"].x.size(0) >= 3]
    if len(candidates) < 2:
        raise ValueError("Need two multi-packet real samples for smoke test")
    batch = Batch.from_data_list(candidates[:2])
    results = {}
    for name in ("E0", "E1", "E2", "E3", "E4"):
        seed_all(42)
        model = build_experiment(name, graphs[0])
        # Always initialize lazy GAT edge projections with real attributes, also
        # in No Edge mode; then restore the intended flag before optimization.
        initialize(model, name, batch)
        model.eval()
        from torch_geometric.nn import GATConv
        received, handles = {}, []
        for module_name, module in model.named_modules():
            if isinstance(module, GATConv):
                def capture(module, inputs, kwargs, key=module_name):
                    attr = kwargs.get("edge_attr", inputs[2] if len(inputs) > 2 else None)
                    received[key] = list(attr.shape) if attr is not None else None
                handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
        original = model(batch).detach()
        for handle in handles:
            handle.remove()
        assert len(received) == (6 if name in ("E0", "E1") else 12)
        assert all((value is None) == (name in ("E1", "E4")) for value in received.values())
        sensitivity = {}
        for relation in RELATIONS:
            changed = batch.clone()
            attr = changed[relation].edge_attr
            attr.zero_()
            attr[:, 0] = torch.arange(attr.size(0), dtype=attr.dtype) * 10
            if name in ("E0", "E1"):
                # Isolate one relation at a time at the reference API. This
                # diagnostic intentionally bypasses the normal mirror check.
                attrs = {r: changed[r].edge_attr if name == "E0" else None for r in RELATIONS}
                probe = model.reference(changed.x_dict, changed.edge_index_dict, attrs, changed)
            else:
                probe = model(changed)
            difference = float((probe.detach() - original).abs().max())
            sensitivity[str(relation)] = difference
        if name in ("E1", "E4"):
            assert max(sensitivity.values()) == 0, sensitivity
        else:
            assert max(sensitivity.values()) > 1e-7, sensitivity
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        loss = F.nll_loss(model(batch), batch.y.view(-1))
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().max() > 0 for g in grads)
        if name == "E3":
            assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.gate.parameters())
        if name == "E2":
            assert all(p.grad is None for p in model.gate.parameters())
        optimizer.step()
        assert any(not torch.equal(before[k], p) for k, p in model.named_parameters())
        results[name] = {"loss": float(loss.detach()), "attribute_sensitivity": sensitivity,
                         "actual_GAT_inputs": received,
                         "finite_backward_and_update": True}
    # Equal initial states for both causal comparisons; no lazy-init RNG drift.
    for left, right in (("E0", "E1"), ("E2", "E3")):
        states = []
        for name in (left, right):
            seed_all(42)
            model = build_experiment(name, graphs[0])
            initialize(model, name, batch)
            states.append(model.state_dict())
        assert states[0].keys() == states[1].keys()
        assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    # A single stored graph set supplies all modes; forward must not alter it.
    snapshot = batch.clone()
    for name in ("E0", "E1", "E2", "E3"):
        seed_all(42)
        model = build_experiment(name, graphs[0])
        initialize(model, name, batch)
        model.eval()
        model(batch)
        for node in ("flow", "packet"):
            assert torch.equal(batch[node].x, snapshot[node].x)
        for relation in RELATIONS:
            assert torch.equal(batch[relation].edge_index, snapshot[relation].edge_index)
            assert torch.equal(batch[relation].edge_attr, snapshot[relation].edge_attr)
        assert torch.equal(batch.y, snapshot.y)
    bad = batch.clone()
    bad[RELATIONS[2]].edge_attr = torch.zeros((1, 4))
    try:
        validate_graph(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed relation dimension was accepted")
    results["invalid_relation_dimensions_rejected"] = True
    results["paired_initial_states_identical"] = True
    results["input_graphs_unchanged"] = True
    write(output / "smoke.json", results)


def initialize(model, name, batch):
    from xgnid.model import _PaperStyleHeteroEncoder
    model.eval()
    if name == "E1":
        model.use_edge_attr = True
    encoders = [m for m in model.modules() if isinstance(m, _PaperStyleHeteroEncoder)]
    for encoder in encoders:
        encoder.use_edge_attr = True
    with torch.no_grad():
        model(batch)
    if name == "E1":
        model.use_edge_attr = False
    for encoder in encoders:
        encoder.use_edge_attr = name != "E4"


def evaluate(model, loader):
    model.eval()
    labels, predictions, gates = [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            assert torch.isfinite(out).all()
            labels.extend(batch.y.view(-1).tolist())
            predictions.extend(out.argmax(-1).tolist())
            if hasattr(model, "_last_cache"):
                gates.extend(model._last_cache["gate_weights"].tolist())
    return {"accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, labels=list(range(8)), average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, predictions, labels=list(range(8)), average="weighted", zero_division=0),
            "per_class": classification_report(labels, predictions, labels=list(range(8)), target_names=LABELS,
                                               output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(8))).tolist(),
            "inference_seconds": time.perf_counter() - started,
            "labels": labels, "predictions": predictions, "gate_weights": gates}


def run(args, graphs, split):
    for name in args.experiments:
        output = args.output / name
        output.mkdir()
        seed_all(args.seed)
        model = build_experiment(name, graphs[0])
        initialize(model, name, Batch.from_data_list([graphs[i] for i in split["train"][:2]]))
        loaders = {k: DataLoader([graphs[i] for i in ids], batch_size=16, shuffle=k == "train",
                   generator=torch.Generator().manual_seed(args.seed)) for k, ids in split.items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        config = {"experiment": name, "use_edge_attr": name not in ("E1", "E4"),
                  "fusion": "fixed_equal_logits" if name == "E2" else "adaptive_logits" if name in ("E3", "E4") else None,
                  "hidden_dim": 64, "batch_size": 16, "optimizer": "Adam", "lr": 0.001,
                  "epochs": args.epochs, "patience": 12, "seed": args.seed,
                  "auxiliary_losses": False, "calibration": False, "class_weighting": False,
                  "hard_negatives": False, "dataset_sha256": sha(args.data / "graphs.pt"),
                  "split_sha256": sha(args.data / "split.json"),
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "source_hashes": {p: sha(ROOT / p) for p in ["src/xgnid/model.py", "src/xgnid/edge_experiments.py", "third_party/GNN4ID/Utility/Model.py", "scripts/run_edge_chain.py"]},
                  "torch": torch.__version__, "device": "cpu", "scope": "pilot"}
        write(output / "config.json", config)
        best, patience = -1, 12
        started = time.perf_counter()
        with (output / "epochs.jsonl").open("x", encoding="utf-8") as log:
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses = []
                for batch in loaders["train"]:
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.nll_loss(model(batch), batch.y.view(-1))
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                val = evaluate(model, loaders["val"])
                entry = {"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_f1": val["macro_f1"]}
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(name, entry, flush=True)
                if val["macro_f1"] > best:
                    best, patience = val["macro_f1"], 12
                    torch.save({"state_dict": model.state_dict(), "config": config,
                                "best_epoch": epoch, "validation": val}, output / "best.pt")
                else:
                    patience -= 1
                    if patience == 0:
                        break
        elapsed = time.perf_counter() - started
        checkpoint = torch.load(output / "best.pt", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        metrics = evaluate(model, loaders["test"])
        metrics.update(training_seconds=elapsed, best_epoch=checkpoint["best_epoch"],
                       best_validation_macro_f1=best, parameter_count=sum(p.numel() for p in model.parameters()))
        write(output / "metrics.json", metrics)
        write(output / "status.json", {"status": "completed", "exit_code": 0, "scope": "pilot"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "smoke", "train", "eval"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--experiments", nargs="+", choices=["E0", "E1", "E2", "E3", "E4"], default=["E0", "E1", "E2", "E3"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    try:
        if args.command == "prepare":
            prepare(args)
        else:
            graphs = torch.load(args.data / "graphs.pt", weights_only=False)
            split = json.loads((args.data / "split.json").read_text(encoding="utf-8"))["split_indices"]
            if args.command == "eval":
                checkpoint = torch.load(args.checkpoint, weights_only=False)
                config = checkpoint["config"]
                if config["dataset_sha256"] != sha(args.data / "graphs.pt") or config["split_sha256"] != sha(args.data / "split.json"):
                    raise ValueError("Checkpoint dataset/split identity mismatch")
                model = build_experiment(config["experiment"], graphs[0])
                initialize(model, config["experiment"], Batch.from_data_list([graphs[i] for i in split["train"][:2]]))
                model.load_state_dict(checkpoint["state_dict"])
                loader = DataLoader([graphs[i] for i in split["test"]], batch_size=config["batch_size"], shuffle=False)
                write(args.output / "metrics.json", evaluate(model, loader))
            elif args.command == "smoke":
                smoke(graphs, args.output)
            else:
                smoke(graphs, args.output)
                run(args, graphs, split)
        write(args.output / "status.json", {"status": "completed", "exit_code": 0})
    except Exception:
        (args.output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write(args.output / "status.json", {"status": "failed", "exit_code": 1})
        raise


if __name__ == "__main__":
    main()
