"""Server-only R0-A: numeric membership order and official lazy initialization.

Reuses the completed R0's frozen worktree and graph manifest. Executes the locked
official Edge train/test functions, adding observation/checkpoint hooks only.
Never rebuilds data, resumes, selects checkpoints by test, or launches implicitly.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def within(root, value):
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def active_training():
    # Inspect Python processes, without NVML or assuming its warning means no CUDA.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            parts = (entry / "cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if parts and "python" in Path(parts[0]).name.lower():
            scripts = {Path(p).name for p in parts[1:] if p.endswith(".py")}
            require(not scripts.intersection({"train_grouped.py", "run_r0_edge.py", "run_r0_aligned_server.py",
                                             "xgnid_r0_aligned.py"}),
                    "Another known training process exists: " + entry.name)


def official_functions(path):
    """Keep official arithmetic and call order; append monitors without warmup."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    train, test = functions["train_with_edge_Att"], functions["test_edge"]
    epoch_loops = [n for n in train.body if isinstance(n, ast.For)]
    require(len(epoch_loops) == 1, "Unexpected official epoch loop")
    epoch_loop = epoch_loops[0]
    batches = [n for n in epoch_loop.body if isinstance(n, ast.For)]
    require(len(batches) == 1, "Unexpected official batch loop")
    body, insertions = [], 0
    for statement in batches[0].body:
        body.append(statement)
        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                and ast.unparse(statement.value.func) == "optimizer.step"):
            body.append(ast.parse("_step_monitor(model, batch, optimizer)").body[0])
            insertions += 1
    require(insertions == 1, "Official optimizer step structure changed")
    batches[0].body = body
    epoch_loop.body.append(ast.parse(
        "_epoch_monitor(epoch + 1, model, optimizer, total_loss, train_acc, num_graphs)"
    ).body[0])
    return ast.fix_missing_locations(ast.Module(body=[test, train], type_ignores=[]))


def run(args):
    require(sys.platform.startswith("linux"), "Run only on the Linux server; no local model execution")
    active_training()
    repo = args.repo.resolve()
    launch = read(repo / "outputs/r0_full_launch.json")
    code, manifest, original = [within(repo, launch[k]) for k in ("code_root", "manifest_dir", "output")]
    baseline = read(original / "summary.json")
    require((baseline["status"], baseline["exit_code"], baseline["epochs_completed"], baseline["selection"])
            == ("completed", 0, 30, "last_epoch"), "Original R0 is incomplete")
    require(sha(original / "last.pt") == baseline["last_checkpoint_sha256"], "Original checkpoint changed")
    binding0 = baseline["binding"]
    for name, digest in binding0["code"].items():
        require(identity(within(code, name).read_text(encoding="utf-8-sig")) == digest, "Frozen source changed: " + name)
    lock = read(code / "docs/r0_source_lock.json")
    for name in ("Model", "Training"):
        path = code / ("third_party/GNN4ID/Utility/" + name + ".py")
        require(identity(path.read_text(encoding="utf-8-sig"))
                == lock["files"]["Utility/" + name + ".py"]["normalized_text_identity"], "Official source differs")
    config = read(code / "configs/r0_edge.json")
    require(identity(config) == binding0["config_id"], "Frozen configuration changed")
    require((config["epochs"], config["batch_size"], config["test_batch_size"], config["lr"],
             config["seed"], config["weight_decay"], config["scheduler"])
            == (30, 64, 1, .01, 42, 0, None), "Unexpected fixed protocol")
    sys.path.insert(0, str(code / "scripts"))
    import r0_data
    import run_r0_edge as base
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv
    require(r0_data.ROOT.resolve() == code, "Incorrect frozen module loaded")
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available(), "R0-A requires real server CUDA")
    torch.set_num_threads(2)
    train = r0_data.R0Graphs(manifest, "train")
    require(train.meta["manifest_id"] == binding0["manifest_id"], "Manifest identity differs")
    require(train.meta["partition_label_counts"] == r0_data.table4_counts(), "Unexpected membership counts")

    def numeric_rows(dataset, partition):
        pattern = r"data_(\d+)\.pt" if partition == "train" else r"data_test_(\d+)\.pt"
        numbered = []
        for row in dataset.rows:
            match = re.fullmatch(pattern, Path(row["path"]).name)
            require(match is not None and row["container_offset"] == 0, "Not official single-graph numeric filenames")
            numbered.append((int(match[1]), row))
        numbered.sort(key=lambda pair: pair[0])
        require([i for i, _ in numbered] == list(range(len(numbered))), "Numeric membership is not contiguous/unique")
        return [row for _, row in numbered]

    train.rows = numeric_rows(train, "train")
    ordered_members = identity([r["sample_id"] for r in train.rows])
    # Fixed balanced smoke subset, selected from numeric order, never test errors.
    if args.mode == "smoke":
        counts, selected = Counter(), []
        for row in train.rows:
            if counts[row["label"]] < 32:
                selected.append(row)
                counts[row["label"]] += 1
        require(counts == Counter({i: 32 for i in range(8)}), "Incomplete smoke classes")
        train.rows = selected
    experiment = {"experiment": "R0-A", "base_config_id": binding0["config_id"],
                  "order": "contiguous_numeric_filename", "initialization": "first_training_forward_no_warmup",
                  "training": "locked_official_train_with_edge_Att_with_observation_hooks",
                  "seed": 42, "claim": "protocol_reimplementation", "joint_control_not_individual_causal_attribution": True}
    binding = {"manifest_id": train.meta["manifest_id"], "base_code": binding0["code"],
               "experiment": experiment, "tool_identity": identity(Path(__file__).read_text(encoding="utf-8-sig")),
               "training_source_identity": identity((code / "third_party/GNN4ID/Utility/Training.py").read_text(encoding="utf-8-sig")),
               "runtime": base.runtime(device), "full_numeric_train_members_id": ordered_members}
    if args.mode == "full":
        require(args.smoke_report is not None, "A successful matching R0-A CUDA smoke report is required")
        smoke = read(args.smoke_report)
        require((smoke["status"], smoke["exit_code"], smoke["mode"], smoke["epochs_completed"])
                == ("completed", 0, "smoke", 2), "R0-A smoke did not complete")
        require(smoke["binding"] == binding and smoke["optimizer_step_passed"] and smoke["test"] is None,
                "Smoke data/code/config/runtime binding differs")
        require(sha(args.smoke_report.parent / "last.pt") == smoke["last_checkpoint_sha256"], "Smoke checkpoint changed")
    output = args.output.resolve()
    output.relative_to((repo / "outputs").resolve())
    require(output.parent == (repo / "outputs").resolve() and output.name.startswith("r0_aligned_"),
            "Use a new direct outputs/r0_aligned_* directory")
    output.mkdir(exist_ok=False)
    write(output / "status.json", {"status": "running", "mode": args.mode, "pid": os.getpid()})
    started = time.time()
    try:
        write(output / "config.json", {"binding": binding, "base_config": config, "mode": args.mode,
              "source_run": str(original), "samples": len(train), "workers": 0,
              "smoke_report": str(args.smoke_report) if args.smoke_report else None})
        tree = official_functions(code / "third_party/GNN4ID/Utility/Training.py")
        (output / "executed_training.py").write_text(ast.unparse(tree) + "\n", encoding="utf-8")
        # No model execution, loader iteration or smoke probes before this seed.
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        spec = importlib.util.spec_from_file_location("r0a_official", code / "third_party/GNN4ID/Utility/Model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.HeteroGNN_Edge(train[0], {"hidden_size": 64, "eps": 1.0}, aggr="mean").to(device)
        loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=0)
        first_state, attrs, handles = {}, {}, []
        steps, epoch_steps, completed = 0, 0, 0
        epoch_order = hashlib.sha256()
        first_step = {}
        tick = time.time()

        def first_forward(current, inputs, result):
            if not first_state:
                require(tuple(result.shape) == (inputs[-1].num_graphs, 8), "Output dimensions differ")
                first_state.update({k: p.detach().clone() for k, p in current.named_parameters()})

        handles.append(model.register_forward_hook(first_forward))
        for name, child in model.named_modules():
            if isinstance(child, GATConv):
                def observe(current, inputs, kwargs, name=name):
                    attr = kwargs.get("edge_attr", inputs[2] if len(inputs) > 2 else None)
                    attrs[name] = None if attr is None else list(attr.shape)
                handles.append(child.register_forward_pre_hook(observe, with_kwargs=True))

        def step_monitor(current, batch, optimizer):
            nonlocal steps, epoch_steps
            steps += 1
            epoch_steps += 1
            epoch_order.update((json.dumps(batch.sample_id) + "\n").encode())
            if steps == 1:
                require(len(attrs) == 6 and all(v is not None for v in attrs.values()), "Missing actual GAT edge inputs")
                require(all(torch.isfinite(p.grad).all() for p in current.parameters() if p.grad is not None),
                        "Nonfinite first-step gradients")
                updated = sum(not torch.equal(first_state[k], p) for k, p in current.named_parameters())
                require(updated > 0, "First optimizer step did not update parameters")
                edge_grad = {k: float(m.lin_edge.weight.grad.abs().max()) if m.lin_edge.weight.grad is not None else None
                             for k, m in current.named_modules() if isinstance(m, GATConv)}
                require(any(v is not None and v > 0 for v in edge_grad.values()), "No nonzero edge projection gradient")
                first_step.update(updated_parameter_tensors=updated, gat_attribute_shapes=dict(attrs), edge_projection_gradients=edge_grad)
                for handle in handles:
                    handle.remove()
                first_state.clear()
            if epoch_steps % 100 == 0:
                print(f"epoch={completed + 1} batches={epoch_steps}/{len(loader)}", flush=True)

        def epoch_monitor(epoch, current, optimizer, loss, accuracy, count):
            nonlocal completed, epoch_steps, epoch_order, tick
            require(count == len(train) and epoch_steps == len(loader), "Epoch sample/step count differs")
            require(np.isfinite(loss) and all(torch.isfinite(p).all() for p in current.parameters()), "Nonfinite epoch state")
            bn = {k: int(v) for k, v in current.state_dict().items() if k.endswith("num_batches_tracked")}
            require(len(bn) == 4 and all(v == steps for v in bn.values()), "Unexpected BN training counters")
            entry = {"epoch": epoch, "train_loss": loss, "train_accuracy": accuracy, "train_samples": count,
                     "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - tick,
                     "train_order_sha256": epoch_order.hexdigest(), "optimizer_steps": steps}
            with (output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            torch.save({"model_state": current.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch, "binding": binding, "bn_counters": bn,
                        "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
                        "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate()}, output / "last.pt")
            completed, epoch_steps, epoch_order, tick = epoch, 0, hashlib.sha256(), time.time()
            write(output / "status.json", {"status": "running", "mode": args.mode, "pid": os.getpid(), "epochs_completed": epoch})
            print(json.dumps(entry), flush=True)

        namespace = {"torch": torch, "tqdm": lambda value: value,
                     "_step_monitor": step_monitor, "_epoch_monitor": epoch_monitor}
        exec(compile(tree, "<locked-official-edge-with-monitors>", "exec"), namespace)
        epochs = 2 if args.mode == "smoke" else 30
        namespace["train_with_edge_Att"](loader, model, {"lr": .01, "epochs": epochs}, device)
        payload = torch.load(output / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        result = {"status": "completed", "exit_code": 0, "mode": args.mode, "epochs_completed": completed,
                  "binding": binding, "claim": "protocol_reimplementation", "selection": "last_epoch",
                  "optimizer_step_passed": bool(first_step), "first_step": first_step,
                  "parameters": sum(p.numel() for p in model.parameters()), "test": None,
                  "last_checkpoint_sha256": sha(output / "last.pt")}
        if args.mode == "full":
            test = r0_data.R0Graphs(manifest, "test")
            test.rows = numeric_rows(test, "test")
            truth, predictions, rows = [], [], []
            model.eval()
            with torch.no_grad():
                for batch in DataLoader(test, batch_size=1, shuffle=False, num_workers=0):
                    batch = batch.to(device)
                    values = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
                    require(torch.isfinite(values).all(), "Nonfinite test output")
                    label, pred = int(batch.y.item()), int(values.argmax(-1).item())
                    truth.append(label)
                    predictions.append(pred)
                    rows.append({"sample_id": batch.sample_id[0], "label": label, "prediction": pred})
            result["test"] = base.metrics(truth, predictions)
            write(output / "metrics.json", result["test"])
            with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["sample_id", "label", "prediction"])
                writer.writeheader()
                writer.writerows(rows)
        require(sha(original / "last.pt") == baseline["last_checkpoint_sha256"], "Original checkpoint changed during control")
        result["wall_seconds"] = time.time() - started
        write(output / "summary.json", result)
        write(output / "status.json", {"status": "completed", "exit_code": 0})
        return result
    except BaseException as exc:
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write(output / "status.json", {"status": "failed", "exit_code": 1, "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
