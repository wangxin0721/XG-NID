"""R0-V: up to 100 epochs; early stopping; grouped TRAIN holdout; test once.

Server-only execution. Import and pure split/stopping helpers use stdlib only.
Never resumes an old run, changes the original split, or stops another job.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def within(root, value):
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def configuration(path):
    config = read(path)
    require(config["experiment"] == "R0-V" and config["schema"] == "xgnid.r0-validation.config.v1",
            "Not an R0-V configuration")
    expected = {"seed": 42, "val_fraction_of_source_train": .1, "group_by": "node_hash",
                "max_epochs": 100, "early_stopping": True, "patience": 12, "min_delta": 0.,
                "selection": "best_validation_macro_f1", "tie_break": "earliest_epoch",
                "batch_size": 64, "val_batch_size": 64, "test_batch_size": 1,
                "optimizer": "Adam", "lr": .01, "weight_decay": 0., "scheduler": None,
                "hidden_size": 64, "eps": 1., "test_passes": 1}
    require(all(key in config and config[key] == value for key, value in expected.items()),
            "Reviewed R0-V settings changed")
    return config


def source(repo):
    launch = read(repo / "outputs/r0_full_launch.json")
    code, manifest, original = [within(repo, launch[key]) for key in ("code_root", "manifest_dir", "output")]
    baseline = read(original / "summary.json")
    require((baseline["status"], baseline["exit_code"], baseline["epochs_completed"])
            == ("completed", 0, 30), "Original R0 is incomplete")
    binding = baseline["binding"]
    for name, digest in binding["code"].items():
        require(identity(within(code, name).read_text(encoding="utf-8-sig")) == digest,
                "Frozen source changed: " + name)
    require(sha(original / "last.pt") == baseline["last_checkpoint_sha256"], "Original checkpoint changed")
    meta = read(manifest / "manifest.json")
    payload = dict(meta)
    manifest_id = payload.pop("manifest_id")
    require(identity(payload) == manifest_id == binding["manifest_id"], "Original manifest changed")
    require(sha(manifest / "samples.jsonl") == meta["sample_manifest_sha256"], "Original samples changed")
    require(sha(manifest / "membership_evidence.txt") == meta["membership_evidence_sha256"], "Provenance changed")
    lock = read(code / "docs/r0_source_lock.json")
    model = code / "third_party/GNN4ID/Utility/Model.py"
    require(identity(model.read_text(encoding="utf-8-sig"))
            == lock["files"]["Utility/Model.py"]["normalized_text_identity"], "Official model changed")
    return code, manifest, meta, baseline


def grouped_holdout(rows, fraction=.1, seed=42):
    """Nearest-size prefix of a seeded per-class group permutation, TRAIN only.

    Retain all copies and edge variants. Never split identical No Edge inputs.
    Does not inspect labels, features or predictions from the test partition.
    """
    require(0 < fraction < 1, "Invalid validation fraction")
    require(rows and all(row["partition"] == "train" for row in rows), "Only source TRAIN may be split")
    require(len({row["sample_id"] for row in rows}) == len(rows), "Repeated sample ID")
    groups, labels = defaultdict(list), {}
    for row in rows:
        key, label = row["node_hash"], row["label"]
        require(isinstance(key, str) and key and type(label) is int and label in range(8), "Invalid input group/label")
        require(labels.get(key, label) == label, "Conflicting labels in a source-train input group")
        labels[key] = label
        groups[key].append(row)
    require(set(labels.values()) == set(range(8)), "Source train misses a class")
    assignment, details = {}, {}
    for label in range(8):
        keys = sorted(key for key in groups if labels[key] == label)
        require(len(keys) >= 2, "Need at least two independent input groups per class")
        random.Random(seed + label).shuffle(keys)
        target = sum(len(groups[key]) for key in keys) * fraction
        cumulative, candidates = 0, []
        for index, key in enumerate(keys[:-1], 1):
            cumulative += len(groups[key])
            candidates.append((abs(cumulative - target), index, cumulative))
        _, count, actual = min(candidates)
        chosen = set(keys[:count])
        assignment.update({key: "val" if key in chosen else "train" for key in keys})
        details[str(label)] = {"target_val_samples": target, "actual_val_samples": actual,
                               "val_groups": count, "train_groups": len(keys) - count}
    result = [dict(row, source_partition="train", partition=assignment[row["node_hash"]]) for row in rows]
    return result, details


def numeric_rows(rows):
    numbered = []
    for row in rows:
        match = re.fullmatch(r"data_(\d+)\.pt", Path(row["path"]).name)
        require(match and row["container_offset"] == 0, "Expected original numeric training filenames")
        numbered.append((int(match[1]), row))
    numbered.sort(key=lambda pair: pair[0])
    require([i for i, _ in numbered] == list(range(len(rows))), "Source train indices are not contiguous")
    return [row for _, row in numbered]


def prepare(args, config):
    _, manifest, meta, _ = source(args.repo)
    rows = [json.loads(line) for line in (manifest / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    train = numeric_rows([row for row in rows if row["partition"] == "train"])
    require(Counter(row["label"] for row in train) == Counter({i: 20000 for i in range(8)}),
            "Unexpected source train membership")
    split_rows, details = grouped_holdout(train, config["val_fraction_of_source_train"], config["seed"])
    directory = args.split_dir
    directory.relative_to(args.repo / "outputs")
    directory.mkdir(parents=True, exist_ok=False)
    samples = directory / "samples.jsonl"
    with samples.open("x", encoding="utf-8") as stream:
        for row in split_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    counts = {part: dict(sorted(Counter(str(r["label"]) for r in split_rows if r["partition"] == part).items()))
              for part in ("train", "val")}
    binding = {"source_manifest_id": meta["manifest_id"], "source_samples_sha256": meta["sample_manifest_sha256"],
               "config_id": identity(config), "samples_sha256": sha(samples),
               "builder_identity": identity(Path(__file__).read_text(encoding="utf-8-sig"))}
    report = {"status": "completed", "exit_code": 0, "schema": "xgnid.r0-validation.split.v1",
              "split_id": identity(binding), "binding": binding, "source_manifest_dir": str(manifest),
              "partition_label_counts": counts, "per_class_holdout": details,
              "source_test_counts_unchanged": meta["partition_label_counts"]["test"],
              "inherited_source_overlap": {k: meta.get(k) for k in ("edge_overlap", "node_overlap")},
              "no_train_val_node_hash_overlap": True, "all_source_train_rows_retained": len(split_rows) == 160000,
              "resampling": "none added; preserve all inherited source-train copies within their input group",
              "independence": "input groups, not proven PCAP/capture/flow independence; original test overlap is retained"}
    write(directory / "split.json", report)
    return report


class Selection:
    """Strict improvement; ties retain the earliest best epoch and count as stale."""
    def __init__(self, patience, min_delta=0.):
        require(patience is None or type(patience) is int and patience > 0, "Invalid patience")
        self.patience, self.min_delta = patience, min_delta
        self.best_score, self.best_epoch, self.stale = -math.inf, None, 0

    def observe(self, epoch, score):
        require(math.isfinite(score) and 0 <= score <= 1, "Invalid validation Macro-F1")
        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score, self.best_epoch, self.stale = score, epoch, 0
        else:
            self.stale += 1
        return improved, self.patience is not None and self.stale >= self.patience


def batches(order, size):
    """Merge final singleton into preceding batch for training Flow BatchNorm."""
    require(len(order) >= 2 and size >= 2, "Training requires at least two samples")
    result = [order[i:i + size] for i in range(0, len(order), size)]
    if len(result[-1]) == 1 and len(result) > 1:
        result[-2].extend(result.pop())
    return result


def no_training_processes():
    names = {"train_grouped.py", "run_r0_edge.py", "run_r0_aligned_server.py", "xgnid_r0_aligned.py",
             "run_r0_validation_server.py", "xgnid_r0_validation.py"}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            parts = (proc / "cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if parts and "python" in Path(parts[0]).name.lower():
            require(not names.intersection(Path(p).name for p in parts[1:] if p.endswith(".py")),
                    "Known training process still running, PID=" + proc.name + "; no process was stopped")


def verify_split(args, config, source_meta):
    report = read(args.split_dir / "split.json")
    binding = report["binding"]
    require(report["status"] == "completed" and report["exit_code"] == 0, "Split incomplete")
    require(report["split_id"] == identity(binding), "Split identity changed")
    require(binding["source_manifest_id"] == source_meta["manifest_id"]
            and binding["source_samples_sha256"] == source_meta["sample_manifest_sha256"]
            and binding["config_id"] == identity(config)
            and binding["builder_identity"] == identity(Path(__file__).read_text(encoding="utf-8-sig"))
            and binding["samples_sha256"] == sha(args.split_dir / "samples.jsonl"), "Split/source/config/code changed")
    return report, [json.loads(line) for line in (args.split_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()]


def verify_smoke(path, binding):
    require(binding["runtime"]["device_type"] == "cuda", "Full training requires CUDA smoke")
    require(path is not None, "Full training requires a new R0-V CUDA smoke report")
    smoke = read(path)
    require((smoke["status"], smoke["exit_code"], smoke["mode"], smoke["epochs_completed"])
            == ("completed", 0, "smoke", 2), "Incomplete R0-V smoke")
    require(smoke["binding"] == binding and smoke["optimizer_step_passed"]
            and smoke["test"] is None and smoke["test_passes"] == 0, "Smoke binding/optimizer/test mismatch")
    for name in ("best", "last"):
        require(sha(path.parent / (name + ".pt")) == smoke[name + "_checkpoint_sha256"], "Smoke checkpoint changed")


def train(args, config):
    no_training_processes()
    code, manifest, meta, baseline = source(args.repo)
    split, split_rows = verify_split(args, config, meta)
    sys.path.insert(0, str(code / "scripts"))
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv
    import r0_data
    import run_r0_edge as base
    require(r0_data.ROOT.resolve() == code, "Wrong frozen source imported")
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available(), "Real server CUDA is required")
    torch.set_num_threads(2)
    binding = {"experiment": config["experiment"], "config": config, "split_id": split["split_id"],
               "source_binding": baseline["binding"], "runtime": base.runtime(device),
               "entry_identity": identity(Path(__file__).read_text(encoding="utf-8-sig"))}
    if args.mode == "full":
        verify_smoke(args.smoke_report, binding)
    partitions = {}
    for part in ("train", "val"):
        dataset = r0_data.R0Graphs(manifest, "train")
        originals = {row["sample_id"]: row for row in dataset.rows}
        rows = [row for row in split_rows if row["partition"] == part]
        require(rows and len({row["sample_id"] for row in rows}) == len(rows), "Invalid selected membership")
        for row in rows:
            original = originals.get(row["sample_id"])
            require(original is not None and all(original[k] == v for k, v in row.items()
                    if k not in ("partition", "source_partition")), "Split row differs from original")
        dataset.rows = [originals[row["sample_id"]] for row in rows]
        if args.mode == "smoke":
            counts, selected = Counter(), []
            limit = 32 if part == "train" else 16
            for row in dataset.rows:
                if counts[row["label"]] < limit:
                    selected.append(row)
                    counts[row["label"]] += 1
            require(counts == Counter({label: limit for label in range(8)}), "Incomplete balanced smoke subset")
            dataset.rows = selected
        partitions[part] = dataset
    require(not {r["node_hash"] for r in partitions["train"].rows}
            .intersection(r["node_hash"] for r in partitions["val"].rows), "Train/val input leakage")
    output = args.output.resolve()
    require(output.parent == args.repo / "outputs" and output.name.startswith("r0_validation_"),
            "Use a new direct outputs/r0_validation_* directory")
    output.mkdir(exist_ok=False)
    write(output / "status.json", {"status": "running", "mode": args.mode, "pid": os.getpid()})
    started, test_passes = time.time(), 0
    try:
        write(output / "config.json", {"binding": binding, "mode": args.mode,
              "samples": {p: len(d) for p, d in partitions.items()}, "workers": 0,
              "singleton_policy": "merge final singleton into previous training batch"})
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        spec = importlib.util.spec_from_file_location("r0v_official", code / "third_party/GNN4ID/Utility/Model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.HeteroGNN_Edge(partitions["train"][0], config, aggr="mean").to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        selection = Selection(config["patience"], config["min_delta"])
        maximum = 2 if args.mode == "smoke" else config["max_epochs"]
        first_step, steps = {}, 0

        def save(path, payload):
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)

        def evaluate(dataset, batch_size, export=False):
            model.eval()
            truth, predictions, exports = [], [], []
            with torch.no_grad():
                for graph in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
                    graph = graph.to(device)
                    values = model(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict, graph)
                    require(torch.isfinite(values).all(), "Nonfinite evaluation output")
                    labels, predicted = graph.y.view(-1).tolist(), values.argmax(-1).tolist()
                    truth.extend(labels)
                    predictions.extend(predicted)
                    if export:
                        exports.extend({"sample_id": s, "label": y, "prediction": p}
                                       for s, y, p in zip(graph.sample_id, labels, predicted))
            require(len(truth) == len(dataset), "Incomplete evaluation")
            return base.metrics(truth, predictions), exports

        for epoch in range(1, maximum + 1):
            tick = time.time()
            model.train()
            order = torch.randperm(len(partitions["train"])).tolist()
            loader = DataLoader(partitions["train"], batch_sampler=batches(order, config["batch_size"]), num_workers=0)
            loss_sum, correct, seen = 0., 0, 0
            order_digest = hashlib.sha256()
            for index, graph in enumerate(loader, 1):
                graph = graph.to(device)
                optimizer.zero_grad()
                values = model(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict, graph)
                loss = model.loss(values, graph.y)
                require(torch.isfinite(loss), "Nonfinite training loss")
                loss.backward()
                if not first_step:
                    before = {k: p.detach().clone() for k, p in model.named_parameters()}
                    require(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),
                            "Nonfinite gradients")
                    gradients = {k: float(m.lin_edge.weight.grad.abs().max())
                                 if m.lin_edge.weight.grad is not None else None
                                 for k, m in model.named_modules() if isinstance(m, GATConv)}
                    require(len(gradients) == 6 and any(v is not None and v > 0 for v in gradients.values()),
                            "Missing GAT edge projection gradient")
                optimizer.step()
                if not first_step:
                    changed = sum(not torch.equal(before[k], p) for k, p in model.named_parameters())
                    require(changed > 0, "Optimizer did not update model")
                    first_step.update(updated_parameter_tensors=changed, edge_projection_gradients=gradients)
                    del before
                steps += 1
                seen += graph.num_graphs
                loss_sum += float(loss.detach()) * graph.num_graphs
                correct += int(values.argmax(-1).eq(graph.y).sum())
                order_digest.update((json.dumps(graph.sample_id) + "\n").encode())
                if index % 100 == 0:
                    print(f"epoch={epoch} batches={index}/{len(loader)}", flush=True)
            require(seen == len(partitions["train"]), "Incomplete training epoch")
            require(all(torch.isfinite(p).all() for p in model.parameters()), "Nonfinite epoch parameters")
            bn = {k: int(v) for k, v in model.state_dict().items() if k.endswith("num_batches_tracked")}
            require(len(bn) == 4 and all(value == steps for value in bn.values()), "Unexpected BN training counters")
            val, _ = evaluate(partitions["val"], config["val_batch_size"])
            improved, stop = selection.observe(epoch, val["macro_f1"])
            payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                       "epoch": epoch, "binding": binding, "val": val,
                       "best_epoch": selection.best_epoch, "best_val_macro_f1": selection.best_score,
                       "stale_epochs": selection.stale, "bn_counters": bn, "torch_rng_state": torch.get_rng_state(),
                       "cuda_rng_states": torch.cuda.get_rng_state_all(),
                       "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate()}
            save(output / "last.pt", payload)
            if improved:
                save(output / "best.pt", payload)
            entry = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
                     "train_samples": seen, "val": val, "best_epoch": selection.best_epoch,
                     "best_val_macro_f1": selection.best_score, "stale_epochs": selection.stale,
                     "improved": improved, "seconds": time.time() - tick,
                     "lr": optimizer.param_groups[0]["lr"], "optimizer_steps": steps,
                     "train_order_sha256": order_digest.hexdigest()}
            with (output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            write(output / "status.json", {"status": "running", "mode": args.mode,
                  "pid": os.getpid(), "epochs_completed": epoch, "best_epoch": selection.best_epoch,
                  "val_macro_f1": val["macro_f1"], "stale_epochs": selection.stale})
            print(json.dumps(entry), flush=True)
            if stop:
                break
        require(args.mode != "smoke" or epoch == 2, "Incomplete smoke")
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
        require(checkpoint["binding"] == binding and checkpoint["epoch"] == selection.best_epoch,
                "Selected checkpoint identity differs")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        result = {"status": "completed", "exit_code": 0, "mode": args.mode, "binding": binding,
                  "experiment": config["experiment"], "claim": config["claim"], "epochs_completed": epoch,
                  "stop_reason": "patience" if stop else "max_epochs",
                  "selection": config["selection"],
                  "best_epoch": selection.best_epoch, "best_val_macro_f1": selection.best_score,
                  "best_checkpoint_sha256": sha(output / "best.pt"),
                  "last_checkpoint_sha256": sha(output / "last.pt"),
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "optimizer_step_passed": bool(first_step), "first_step": first_step,
                  "partition_samples": {p: len(d) for p, d in partitions.items()},
                  "test": None, "test_passes": 0}
        if args.mode == "full":
            # Construct and load held-out TEST only after training and best checkpoint selection.
            write(output / "test_started.json", {"best_epoch": selection.best_epoch,
                  "best_checkpoint_sha256": result["best_checkpoint_sha256"]})
            test = r0_data.R0Graphs(manifest, "test")
            result["test"], exports = evaluate(test, config["test_batch_size"], export=True)
            test_passes += 1
            require(test_passes == config["test_passes"], "Repeated test evaluation")
            result["test_passes"] = test_passes
            write(output / "metrics.json", result["test"])
            with (output / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["sample_id", "label", "prediction"])
                writer.writeheader()
                writer.writerows(exports)
        result["wall_seconds"] = time.time() - started
        write(output / "summary.json", result)
        write(output / "status.json", {"status": "completed", "exit_code": 0, "test_passes": test_passes})
        return result
    except BaseException as exc:
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write(output / "status.json", {"status": "failed", "exit_code": 1,
              "error": str(exc), "test_passes_completed": test_passes})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["prepare", "smoke", "full"], required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-report", type=Path)
    args = parser.parse_args()
    require(sys.platform.startswith("linux"), "Run only on the Linux server; no local model execution")
    args.repo = args.repo.resolve()
    args.split_dir = (args.split_dir or args.repo / "outputs/r0_validation_split_s42_v1").resolve()
    config = configuration(args.config)
    if args.mode == "prepare":
        result = prepare(args, config)
    else:
        require(args.output is not None, "Training requires a new output directory")
        result = train(args, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
