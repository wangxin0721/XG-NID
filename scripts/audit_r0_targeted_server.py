"""Server-only R0 value checks and paired diagnostic optimizer steps.

CPU only. Never train a full run, load test graphs/predictions, overwrite source
files, or save model weights. Optional upstream replay removes only the official
CSV write node and requires an explicitly supplied, complete per-PCAP CSV.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
from types import MethodType, SimpleNamespace


ADDITIONAL_ID = "c8ec2eef2f5adc3572355f54ebd8423bfb5968fb31ea90971a1bdc155687d2db"
PACKET = ["udps." + n for n in ("payload_data", "delta_time", "packet_direction", "ip_size", "transport_size", "payload_size",
                               "syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin")]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def within(root, value):
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def added_columns(tree):
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "data"
                        and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str)
                        and target.slice.value.startswith(("Rolling", "Unique_", "packet_size"))):
                    result.append(target.slice.value)
    return result


def find_upstream(repo, graph_root):
    """Bounded header-only inventory; never select a candidate for replay."""
    root = repo / "data"
    candidates, inspected, truncated, directories = [], 0, False, 0
    if not root.is_dir():
        return {"candidates": [], "scope": str(root), "missing": True}
    for current, children, files in os.walk(root, followlinks=False):
        current = Path(current)
        directories += 1
        if current.resolve() == graph_root or len(current.relative_to(root).parts) >= 6:
            children[:] = []
            continue
        children[:] = sorted(d for d in children if d not in (".git", "__pycache__") and not (current / d).is_symlink()
                             and (current / d).resolve() != graph_root)
        if directories > 10000:
            truncated = True
            break
        for name in sorted(files):
            path = current / name
            if path.suffix.lower() != ".csv" or path.is_symlink():
                continue
            if inspected >= 300:
                truncated = True
                break
            inspected += 1
            try:
                with path.open(newline="", encoding="utf-8-sig") as stream:
                    header = next(csv.reader(stream))
                if {"src_ip", "dst_ip", "bidirectional_first_seen_ms"}.issubset(header):
                    candidates.append({"path": str(path), "bytes": path.stat().st_size,
                        "raw_protocol_available": "protocol" in header, "raw_expiration_available": "expiration_id" in header,
                        "mac_columns": [n for n in ("src_mac", "dst_mac") if n in header],
                        "already_has_temporal": any(n.startswith("Rolling_") for n in header),
                        "full_pre_sampling_context_verified": False})
            except (OSError, UnicodeError, StopIteration, csv.Error):
                continue
        if truncated:
            break
    return {"scope": str(root), "csv_headers_inspected": inspected, "truncated": truncated, "candidates": candidates,
            "note": "Headers only; absence in this scope does not establish global absence. Candidates may already be split/filtered."}


def context(repo):
    launch = read(repo / "outputs/r0_full_launch.json")
    code, directory, run = [within(repo, launch[k]) for k in ("code_root", "manifest_dir", "output")]
    summary = read(run / "summary.json")
    binding = summary["binding"]
    require((summary["status"], summary["exit_code"], summary["epochs_completed"]) == ("completed", 0, 30), "R0 completion differs")
    require(summary["selection"] == "last_epoch" and summary["mode"] == "full", "Checkpoint selection differs")
    require(read(run / "status.json") == {"status": "completed", "exit_code": 0}, "R0 final status differs")
    require(sha(run / "last.pt") == summary["last_checkpoint_sha256"], "Original checkpoint changed")
    for name, digest in binding["code"].items():
        require(identity(within(code, name).read_text(encoding="utf-8-sig")) == digest, "Frozen source changed: " + name)
    config = read(code / "configs/r0_edge.json")
    require(identity(config) == binding["config_id"] and config["seed"] == 42 and config["lr"] == .01 and config["weight_decay"] == 0, "Protocol differs")
    require(read(run / "config.json")["binding"] == binding and all(launch[k] == binding[k] for k in ("manifest_id", "config_id")), "Recorded run binding differs")
    lock = read(code / "docs/r0_source_lock.json")
    for name in ("Utility/Model.py", "Utility/Training.py"):
        text = (code / "third_party/GNN4ID" / name).read_text(encoding="utf-8-sig")
        require(identity(text) == lock["files"][name]["normalized_text_identity"], "Official source lock mismatch: " + name)
    additional = code / "third_party/GNN4ID/Utility/Additional_Features.py"
    require(identity(additional.read_text(encoding="utf-8-sig")) == ADDITIONAL_ID, "Official additional feature helper differs")
    evidence = read(directory.parent / "source_evidence.json")
    sys.path.insert(0, str(code / "scripts"))
    import r0_data
    require(r0_data.ROOT.resolve() == code, "Wrong data module imported")
    dataset = r0_data.R0Graphs(directory, "train", limit_per_class=8)
    # Select eight fixed interior ranks per class. No test-error-driven selection.
    counts = Counter()
    targets = {i: {1250 + j * 2500 for j in range(8)} for i in range(8)}
    selected = []
    with (directory / "samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["partition"] == "train":
                if counts[row["label"]] in targets[row["label"]]:
                    selected.append(row)
                counts[row["label"]] += 1
    require(len(selected) == 64 and counts == Counter({i: 20000 for i in range(8)}), "Training sample selection differs")
    dataset.rows = selected
    return SimpleNamespace(launch=launch, code=code, run=run, directory=directory, summary=summary, binding=binding,
        config=config, dataset=dataset, selected=selected, evidence=evidence,
        added_tree=ast.parse(additional.read_text(encoding="utf-8-sig"))), r0_data


def paired_step(ctx, output):
    import torch
    import numpy as np
    from torch_geometric.data import Batch
    from xgnid.edge_experiments import ReferenceEdge
    from torch.nn import functional as F
    graphs = [ctx.dataset[i] for i in range(64)]
    batch = Batch.from_data_list(graphs)
    before_inputs = [ctx.data_module.fingerprint(g, True) for g in graphs]
    official_module = load_module("targeted_official_model", ctx.code / "third_party/GNN4ID/Utility/Model.py")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    official = official_module.HeteroGNN_Edge(graphs[0], {"hidden_size": 64, "eps": 1.0}).eval()
    with torch.no_grad():
        official(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
    initial = {k: v.detach().clone() for k, v in official.state_dict().items()}
    wrapped = ReferenceEdge(graphs[0], use_edge_attr=True).eval()
    with torch.no_grad():
        wrapped(batch)
    official.load_state_dict(initial, strict=True)
    wrapped.load_state_dict({"reference." + k: v for k, v in initial.items()}, strict=True)
    require(sum(p.numel() for p in official.parameters()) == 431704, "Wrong official Edge parameter count")
    # Warm optimizer imports before the paired RNG resets; do not take a step.
    warm_optimizer = torch.optim.Adam(official.parameters(), lr=.01)
    del warm_optimizer
    ctx.training_graphs = graphs

    def name(k):
        return k[len("reference."):] if k.startswith("reference.") else k

    def snapshot_state(model):
        return {name(k): v.detach().clone() for k, v in model.state_dict().items()}

    def compare(left, right, title):
        require(left.keys() == right.keys(), title + " key sets differ")
        maximum, mismatches = 0., []
        for k in left:
            a, b = left[k], right[k]
            if a is None or b is None:
                require(a is None and b is None, title + " None gradient differs: " + k)
                continue
            require(a.shape == b.shape and a.dtype == b.dtype, title + " shape/dtype differs: " + k)
            require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), title + " nonfinite tensor: " + k)
            d = float((a.double()-b.double()).abs().max()) if a.numel() else 0.
            maximum = max(maximum, d)
            if not torch.allclose(a, b, rtol=1e-6, atol=1e-7):
                mismatches.append({"parameter": k, "max_abs_delta": d})
        require(not mismatches, title + " mismatch: " + json.dumps(mismatches[:8]))
        return maximum

    compare(snapshot_state(official), snapshot_state(wrapped), "initial state")
    state_hash = hashlib.sha256()
    for k, v in sorted(initial.items()):
        state_hash.update(json.dumps([k, str(v.dtype), list(v.shape)]).encode())
        state_hash.update(v.detach().contiguous().numpy().tobytes())
    results = {}

    def run_one(label, model, execute):
        observation = {"zero_grad_calls": 0, "optimizer_step_calls": 0}
        parameters = {name(k): p for k, p in model.named_parameters()}

        class ObservedAdam(torch.optim.Adam):
            def zero_grad(self, *a, **kw):
                observation["zero_grad_calls"] += 1
                return super().zero_grad(*a, **kw)

            def step(self, *a, **kw):
                observation["optimizer_step_calls"] += 1
                observation["gradients"] = {k: None if p.grad is None else p.grad.detach().clone() for k, p in parameters.items()}
                return super().step(*a, **kw)

        def loss(prediction, target):
            value = F.nll_loss(prediction, target)
            observation["loss"] = value.detach().clone()
            return value

        def forward_hook(module, a, value):
            require(module.training, "Diagnostic forward unexpectedly in eval mode")
            require("output" not in observation, "Unexpected second model forward")
            observation["output"] = value.detach().clone()

        handle = model.register_forward_hook(forward_hook)
        optimizer_box = []

        def adam(*a, **kw):
            optimizer = ObservedAdam(*a, **kw)
            optimizer_box.append(optimizer)
            return optimizer

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        try:
            returned_loss = execute(model, [batch.clone()], adam, loss)
            if returned_loss is not None:
                observation["loss"] = returned_loss
        finally:
            handle.remove()
        require(len(optimizer_box) == 1 and observation["zero_grad_calls"] == observation["optimizer_step_calls"] == 1, "Not exactly one zero_grad/step")
        optimizer = optimizer_box[0]
        observation["state_after"] = snapshot_state(model)
        observation["optimizer_state"] = {k + "/" + field: value.detach().clone() if torch.is_tensor(value) else torch.tensor(value)
            for k, p in parameters.items() for field, value in optimizer.state[p].items()}
        observation["optimizer_groups"] = [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups]
        observation["rng_after"] = torch.get_rng_state().clone()
        results[label] = observation

    official_tree = ast.parse((ctx.code / "third_party/GNN4ID/Utility/Training.py").read_text(encoding="utf-8-sig"))
    official_function = copy.deepcopy(next(n for n in official_tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_with_edge_Att"))
    r0_tree = ast.parse((ctx.code / "scripts/run_r0_edge.py").read_text(encoding="utf-8-sig"))
    run = next(n for n in r0_tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    loops = [n for n in ast.walk(run) if isinstance(n, ast.For) and ast.unparse(n.target) == "(step, data)"]
    require(len(loops) == 1, "R0 training loop structure differs; re-review extraction")
    optimizer_nodes = [n for n in ast.walk(run) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                       and ast.unparse(n.value.func) == "torch.optim.Adam"]
    require(len(optimizer_nodes) == 1, "R0 optimizer construction differs")
    epoch_loop = next(n for n in ast.walk(run) if isinstance(n, ast.For) and ast.unparse(n.target) == "epoch")
    mode_node = next(n for n in epoch_loop.body if isinstance(n, ast.Expr) and ast.unparse(n.value) == "model.train()")
    count_node = next(n for n in epoch_loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "(total, count)")
    r0_function = ast.parse("def r0_one_step(model, loader, device, config):\n pass\n").body[0]
    r0_function.body = [copy.deepcopy(optimizer_nodes[0]), copy.deepcopy(mode_node), copy.deepcopy(count_node)]
    r0_function.body.append(copy.deepcopy(loops[0]))
    require(all(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == a
                    for n in ast.walk(r0_function)) == 1 for a in ("zero_grad", "backward", "step")), "R0 step calls differ")
    snippets = {"official": ast.unparse(official_function), "r0": ast.unparse(r0_function)}
    write(output / "executed_step_source.json", {"scope": "Official function unchanged; R0 batch-loop AST unchanged, surrounding setup/IO omitted",
        "official_epoch_accuracy": "Stubbed out after optimizer step; not part of this diagnostic", **snippets})

    def official_execute(model, loader, adam, loss):
        # Keep the genuine official loss method, instrument only its returned scalar.
        original_loss = model.loss
        def observed_loss(self, pred, target):
            value = original_loss(pred, target)
            results_pending["loss"] = value.detach().clone()
            return value
        results_pending = {}
        model.loss = MethodType(observed_loss, model)
        namespace = {"torch": SimpleNamespace(optim=SimpleNamespace(Adam=adam)), "tqdm": lambda x: x,
                     "test_edge": lambda *a: 0., "print": lambda *a, **kw: None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[official_function], type_ignores=[])), "<locked-official-edge-training>", "exec"), namespace)
        namespace["train_with_edge_Att"](loader, model, {"lr": .01, "epochs": 1}, "cpu")
        require("loss" in results_pending, "Official loss was not called")
        return results_pending["loss"]

    run_one("official", official, official_execute)

    def r0_execute(model, loader, adam, loss):
        class TorchProxy:
            optim = SimpleNamespace(Adam=adam)
            def __getattr__(self, key):
                return getattr(torch, key)
        namespace = {"torch": TorchProxy(), "F": SimpleNamespace(nll_loss=loss), "time": time}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[r0_function], type_ignores=[])), "<frozen-r0-batch-loop>", "exec"), namespace)
        namespace["r0_one_step"](model, loader, torch.device("cpu"), ctx.config)

    run_one("r0", wrapped, r0_execute)
    a, b = results["official"], results["r0"]
    deltas = {"training_output": compare({"output": a["output"]}, {"output": b["output"]}, "training output"),
              "loss": compare({"loss": a["loss"]}, {"loss": b["loss"]}, "loss"),
              "gradients": compare(a["gradients"], b["gradients"], "gradients"),
              "parameters_and_buffers": compare(a["state_after"], b["state_after"], "updated state"),
              "adam_state": compare(a["optimizer_state"], b["optimizer_state"], "Adam state")}
    require(a["optimizer_groups"] == b["optimizer_groups"] and torch.equal(a["rng_after"], b["rng_after"]), "Optimizer configuration or post-step RNG differs")
    counters = {k: int(v) for k, v in a["state_after"].items() if k.endswith("num_batches_tracked")}
    require(len(counters) == 4 and set(counters.values()) == {1}, "BatchNorm did not update exactly once")
    require(before_inputs == [ctx.data_module.fingerprint(g, True) for g in graphs], "Diagnostic inputs changed")
    kinds = Counter("none" if v is None else "zero" if not torch.count_nonzero(v).item() else "nonzero" for v in a["gradients"].values())
    changed = sum(not torch.equal(initial[k], a["state_after"][k]) for k in a["gradients"])
    return {"status": "passed", "device": "cpu", "training_graphs": 64, "classes": 8, "seed": 42,
        "initialization": "Fresh shared weights and BN state; not the final trained checkpoint",
        "initial_state_sha256": state_hash.hexdigest(),
        "training_sample_ids_sha256": identity([row["sample_id"] for row in ctx.selected]),
        "optimizer_steps_each": 1, "loss": float(a["loss"]), "max_abs_deltas": deltas,
        "gradient_kinds": dict(kinds), "updated_parameter_tensors": changed, "batchnorm_counters": counters,
        "optimizer_groups_equal": True, "post_step_rng_equal": True,
        "source_snippets": str(output / "executed_step_source.json"),
        "limits": "One controlled CPU batch; not proof of full-run order/RNG or historical author CUDA behavior"}


def numeric_values(ctx, output):
    import numpy as np
    import pandas as pd
    entry = ctx.evidence["csv_sources"]["train"]
    path = Path(entry["path"])
    before = path.stat()
    require(before.st_size == entry["bytes"] and sha(path) == entry["sha256"], "Training CSV changed")
    header = list(pd.read_csv(path, nrows=0).columns)
    flow = [c for c in header if c not in PACKET + ["Label"]]
    added = added_columns(ctx.added_tree)
    require(len(flow) == 82 and len(added) == 29 and all(c in flow for c in added), "Feature schema differs")
    target_rows = {int(re.fullmatch(r"data_(\d+)\.pt", Path(r["path"]).name)[1]): r for r in ctx.selected}
    observed, stats, violations, rows = {}, {c: {"finite": 0, "nan_or_inf": 0, "zeros": 0, "min": None, "max": None} for c in flow}, Counter(), 0
    examples = []
    bounded = [c for c in added if any(s in c for s in ("UDP_Requests", "TCP_Requests", "ICMP_Requests", "http_port", "DNS_request", "vulnerable_port", "Unique_Ports"))]
    for chunk in pd.read_csv(path, chunksize=2048):
        values = chunk[flow].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        for i, col in enumerate(flow):
            v, valid = values[:, i], finite[:, i]
            s = stats[col]
            s["finite"] += int(valid.sum()); s["nan_or_inf"] += int((~valid).sum()); s["zeros"] += int((v[valid] == 0).sum())
            if valid.any():
                lo, hi = float(v[valid].min()), float(v[valid].max())
                s["min"] = lo if s["min"] is None else min(s["min"], lo)
                s["max"] = hi if s["max"] is None else max(s["max"], hi)
        violations["nonfinite_flow_cells"] += int((~finite).sum())
        expected = chunk[["src2dst_min_ps", "src2dst_max_ps", "dst2src_min_ps", "dst2src_max_ps"]].std(axis=1).to_numpy()
        actual = chunk["packet_size_variation"].to_numpy(dtype=float)
        bad = ~np.isclose(expected, actual, rtol=1e-6, atol=1e-6, equal_nan=False)
        violations["packet_size_variation_formula"] += int(bad.sum())
        for i in np.flatnonzero(bad)[:max(0, 5-len(examples))]:
            examples.append({"csv_row": rows + int(i), "stored": float(actual[i]) if np.isfinite(actual[i]) else None,
                             "recomputed": float(expected[i]) if np.isfinite(expected[i]) else None})
        temporal = chunk[added].to_numpy(dtype=float)
        violations["negative_added_values"] += int((temporal < -1e-8).sum())
        flags = chunk[["Exp_0", "Exp_-1", "proto_1", "proto_2", "proto_6", "proto_17", "proto_58"]].to_numpy(dtype=float)
        violations["nonbinary_onehot_cells"] += int((~np.isin(flags, [0., 1.])).sum())
        violations["multiple_protocol_flags"] += int((flags[:, 2:].sum(axis=1) > 1).sum())
        violations["multiple_expiration_flags"] += int((flags[:, :2].sum(axis=1) > 1).sum())
        violations["zero_protocol_flags_rows"] += int((flags[:, 2:].sum(axis=1) == 0).sum())
        violations["zero_expiration_flags_rows"] += int((flags[:, :2].sum(axis=1) == 0).sum())
        violations["boolean_or_unique_counts_over_default350_cells"] += int((chunk[bounded].to_numpy(dtype=float) > 350 + 1e-6).sum())
        violations["bipackets_below_source_packets_rows"] += int((chunk["Rolling_bipackets_destination"] + 1e-6 < chunk["Rolling_packets_destination"]).sum())
        for ordinal in target_rows:
            if rows <= ordinal < rows + len(chunk):
                observed[ordinal] = chunk.iloc[ordinal-rows].copy()
        rows += len(chunk)
        if rows % 32768 == 0:
            print(f"Numeric training CSV rows={rows}/160000", flush=True)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "CSV changed while profiling")
    require(rows == 160000 and len(observed) == 64, "Training row count or sample ordinals differ")
    # Independent value construction, not the official graph helper: byte arrays,
    # padding/truncation and edge attributes addressed by their actual endpoints.
    sample_checks = []
    for graph_index, row in enumerate(ctx.selected):
        ordinal = int(re.fullmatch(r"data_(\d+)\.pt", Path(row["path"]).name)[1])
        source = observed[ordinal]
        graph = ctx.training_graphs[graph_index] if hasattr(ctx, "training_graphs") else ctx.dataset[graph_index]
        decoded = {c: ast.literal_eval(source[c]) for c in PACKET}
        n = len(decoded["udps.payload_data"])
        differences = []
        if not all(isinstance(v, list) and len(v) == n for v in decoded.values()):
            differences.append("packet_list_lengths")
        expected_packet = np.zeros((n, 1500), dtype=np.float32)
        payload_size_disagreements = 0
        for i, payload in enumerate(decoded["udps.payload_data"]):
            raw = bytes.fromhex(payload)
            value = np.frombuffer(raw[:1500], dtype=np.uint8).astype(np.float32)
            expected_packet[i, :len(value)] = value
            size = int(decoded["udps.payload_size"][i])
            # Author plugin uses the one-zero-byte sentinel for zero payload.
            if size > 0 and len(raw) != size:
                payload_size_disagreements += 1
        if not np.array_equal(graph["packet"].x.numpy(), expected_packet):
            differences.append("packet_bytes_padding_truncation_or_normalization")
        if not np.array_equal(graph["flow"].x.numpy().reshape(-1), np.asarray([source[c] for c in flow], dtype=np.float32)):
            differences.append("flow_values_or_numeric_ordinal_mapping")
        contain = graph["flow", "contain", "packet"]
        attrs = np.asarray([[decoded["udps." + c][i] for c in ("packet_direction", "ip_size", "transport_size", "payload_size")]
                            for i in range(n)], dtype=np.int64).astype(np.float32)
        destinations = contain.edge_index[1].numpy()
        if not np.array_equal(contain.edge_attr.numpy(), attrs[destinations]):
            differences.append("contain_attribute_alignment")
        link = graph["packet", "link", "packet"]
        endpoints = link.edge_index.numpy()
        delta = np.asarray(decoded["udps.delta_time"], dtype=np.int64).astype(np.float32)
        expected_delta = delta[endpoints.max(axis=0)] if endpoints.shape[1] else np.zeros(0, dtype=np.float32)
        if not np.array_equal(link.edge_attr.numpy().reshape(-1), expected_delta):
            differences.append("link_delta_attribute_alignment")
        if int(source["Label"]) != graph.y.item():
            differences.append("candidate_numeric_csv_row_label")
        sample_checks.append({"sample_id": row["sample_id"], "candidate_csv_row": ordinal, "packets": n,
                              "differences": differences, "payload_size_disagreements": payload_size_disagreements})
    violations["independent_sample_numeric_mismatches"] = sum(bool(s["differences"]) for s in sample_checks)
    violations["sample_payload_size_disagreements"] = sum(s["payload_size_disagreements"] for s in sample_checks)
    hard = [k for k in violations if violations[k] and k not in ("zero_protocol_flags_rows", "zero_expiration_flags_rows", "boolean_or_unique_counts_over_default350_cells")]
    detail = {"training_csv_sha256": entry["sha256"], "training_rows": rows, "flow_columns": flow,
              "added_columns": added, "feature_statistics": stats, "violations_or_warnings": dict(violations),
              "formula_mismatch_examples": examples,
              "independent_sample_value_checks": sample_checks,
              "policy": "No normalization/imputation/fitting applied; numeric coercion is diagnostic only. Zero onehot rows may reflect out-of-category inputs. Default350 bounds are a hypothesis, not provenance."}
    write(output / "feature_values.json", detail)
    ctx.observed_csv_rows, ctx.flow_columns, ctx.added_columns = observed, flow, added
    return {"status": "findings_require_review" if hard else "value_constraints_passed_upstream_unverified",
            "training_rows": rows, "feature_columns": len(flow), "hard_findings": hard,
            "violations_or_warnings": dict(violations), "formula_checked": "packet_size_variation = std(four packet-size statistics, ddof=1)",
            "all_zero_columns": [c for c, s in stats.items() if s["zeros"] == rows],
            "independent_training_samples_checked": len(sample_checks),
            "detail": str(output / "feature_values.json"), "full_rolling_recalculation": "requires complete upstream CSV"}


def upstream_replay(ctx, path, output, max_mib):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    require(path.is_file() and not path.is_symlink(), "Explicit upstream CSV missing or symlinked")
    require(path.resolve() != Path(ctx.evidence["csv_sources"]["train"]["path"]).resolve(), "Final sampled CSV cannot replace upstream context")
    require(path.stat().st_size <= max_mib * 1024**2, "Upstream CSV exceeds memory guard; review size before raising --max-upstream-mib")
    header = set(pd.read_csv(path, nrows=0).columns)
    require({"src_ip", "dst_ip", "src_port", "dst_port", "bidirectional_first_seen_ms", "protocol", "expiration_id"}.issubset(header), "Upstream raw context missing; do not invent dropped IP/time/protocol fields")
    before = path.stat()
    digest = sha(path)
    function = copy.deepcopy(next(n for n in ctx.added_tree.body if isinstance(n, ast.FunctionDef) and n.name == "additional_features"))
    writers = [i for i, n in enumerate(function.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "to_csv"]
    require(len(writers) == 1 and writers[0] == len(function.body)-1, "Official final CSV write structure differs")
    function.body[writers[0]] = ast.Return(value=ast.Name(id="data", ctx=ast.Load()))
    namespace = {"pd": pd, "np": np, "LabelEncoder": LabelEncoder}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "<locked-official-features-without-write>", "exec"), namespace)
    replay = namespace["additional_features"](str(path))
    require(isinstance(replay, pd.DataFrame), "Official upstream feature computation did not complete")
    base = [c for c in ctx.flow_columns if c not in ctx.added_columns]
    require(set(base + PACKET).issubset(replay.columns), "Replayed file lacks matching base/packet columns")
    def key(row):
        numeric = np.asarray([row[c] for c in base], dtype=np.float32).tobytes()
        packets = json.dumps([str(row[c]) for c in PACKET]).encode()
        return hashlib.sha256(numeric + packets).hexdigest()
    targets = {}
    for ordinal, row in ctx.observed_csv_rows.items():
        targets.setdefault(key(row), []).append(ordinal)
    hits = {k: [] for k in targets}
    for _, row in replay.iterrows():
        k = key(row)
        if k in hits:
            hits[k].append(np.asarray([row[c] for c in ctx.added_columns], dtype=float))
    result = []
    for k, ordinals in targets.items():
        unique = {v.astype(np.float32).tobytes() for v in hits[k]}
        for ordinal in ordinals:
            observed = np.asarray([ctx.observed_csv_rows[ordinal][c] for c in ctx.added_columns], dtype=float)
            matches = [bool(np.allclose(v, observed, rtol=1e-6, atol=1e-6, equal_nan=False)) for v in hits[k]]
            result.append({"training_csv_row": ordinal, "source_occurrences": len(matches), "distinct_added_vectors": len(unique),
                           "any_match": any(matches), "unambiguous_match": len(unique) == 1 and bool(matches) and all(matches)})
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and sha(path) == digest, "Upstream source changed")
    write(output / "upstream_replay.json", {"source": str(path), "source_sha256": digest,
        "window": "official default350 only; not a window search", "rows_recomputed": len(replay), "matches": result,
        "limitations": "Source completeness is supplied by user, not established by headers; unmatched rows may belong to other per-PCAP files; collisions can be ambiguous"})
    found = [r for r in result if r["source_occurrences"]]
    return {"status": "matched_available_rows_only" if found and all(r["unambiguous_match"] for r in found) else "findings_or_insufficient_source_coverage",
            "upstream_file": str(path), "training_targets_found": len(found), "unambiguous_matches": sum(r["unambiguous_match"] for r in found),
            "detail": str(output / "upstream_replay.json"), "author_history_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--step-only", action="store_true", help="Skip training CSV profiling and upstream replay")
    parser.add_argument("--upstream-csv", type=Path, help="Explicit complete per-PCAP pre-sampling raw CSV; never auto-selected")
    parser.add_argument("--max-upstream-mib", type=int, default=256)
    args = parser.parse_args()
    require(os.name == "posix", "Execute this diagnostic on the Linux server, not locally")
    require(not (args.step_only and args.upstream_csv), "--step-only excludes upstream replay")
    require(args.max_upstream_mib > 0, "Invalid upstream memory guard")
    sys.dont_write_bytecode = True
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.nice(10)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    repo = args.repo.resolve()
    output = repo / "outputs" / ("r0_targeted_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "xgnid.r0.targeted-audit.v1", "tool_sha256": sha(Path(__file__)), "checks": [],
        "training_run_started": False, "gpu_used": False, "test_graphs_loaded": 0, "test_predictions_read": False,
        "root_cause_established": False, "upstream_csv_explicit": str(args.upstream_csv) if args.upstream_csv else None,
        "limits": ["Diagnostic CPU optimizer steps use fresh shared state, not a resumed formal checkpoint",
                   "Numerical constraints do not establish the original temporal context",
                   "Default replay requires complete pre-sampling file; cannot infer lost IP/time from balanced CSV",
                   "No validation/test performance selection or full retraining"]}
    def check(name, operation):
        print("Checking " + name, flush=True)
        write(output / "status.json", {"status": "running", "phase": name, "pid": os.getpid()})
        try:
            detail = operation()
            report["checks"].append({"name": name, "status": "completed", "detail": detail})
            return detail
        except Exception as exc:
            report["checks"].append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc()})
            return None
    box = {}
    def setup():
        ctx, data_module = context(repo)
        ctx.data_module = data_module
        box["ctx"] = ctx
        return {"manifest_id": ctx.binding["manifest_id"], "code_root": str(ctx.code),
                "sample_selection": "8 fixed interior ranks/class, 64 training graphs", "device": "cpu",
                "runtime": {"torch": torch.__version__, "python": sys.version.split()[0],
                            "pyg": metadata.version("torch-geometric"), "pandas": metadata.version("pandas")}}
    if check("binding_and_training_samples", setup) is not None:
        ctx = box["ctx"]
        check("paired_training_step", lambda: paired_step(ctx, output))
        if not args.step_only:
            check("training_feature_values", lambda: numeric_values(ctx, output))
            inventory = find_upstream(repo, Path(ctx.dataset.root).resolve())
            write(output / "upstream_candidates.json", inventory)
            report["upstream_candidates"] = {"detail": str(output / "upstream_candidates.json"),
                "count": len(inventory.get("candidates", [])), "auto_selected": False,
                "paths": [c["path"] for c in inventory.get("candidates", [])[:20]]}
            if args.upstream_csv and hasattr(ctx, "observed_csv_rows"):
                check("official_upstream_replay", lambda: upstream_replay(ctx, args.upstream_csv.resolve(), output, args.max_upstream_mib))
            else:
                report["official_upstream_replay"] = {"status": "not_performed_missing_complete_upstream_context"}
        def preserved():
            require(sha(ctx.run / "last.pt") == ctx.summary["last_checkpoint_sha256"], "Formal checkpoint changed during diagnostic")
            return {"original_checkpoint_unchanged": True}
        check("original_checkpoint_preserved", preserved)
    failed = [c["name"] for c in report["checks"] if c["status"] == "failed"]
    findings = [c["name"] for c in report["checks"] if c.get("detail", {}).get("status") in ("findings_require_review", "findings_or_insufficient_source_coverage")]
    report.update(status="findings_require_review" if failed or findings else "targeted_checks_complete_history_unresolved",
                  failed_checks=failed, findings=findings, output=str(output))
    write(output / "report.json", report)
    write(output / "status.json", {"status": report["status"], "exit_code": int(bool(failed or findings))})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return int(bool(failed or findings))


if __name__ == "__main__":
    raise SystemExit(main())
