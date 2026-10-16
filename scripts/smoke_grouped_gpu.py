"""One-batch E0--E3 device smoke on frozen train samples; no epoch training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
import torch_geometric

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xgnid.edge_experiments import build_experiment, validate_graph, RELATIONS
from xgnid.model import _PaperStyleHeteroEncoder
from grouped_graphs import FrozenGraphs
from build_grouped_split import sha


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)


def state_hash(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def initialize(model, name, batch):
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


def exercise(name, batch, graph, seed):
    seed_all(seed)
    model = build_experiment(name, graph).to(batch["flow"].x.device)
    initialize(model, name, batch)
    initial = state_hash(model.state_dict().items())
    snapshot = batch.clone()
    received, handles = {}, []
    for key, module in model.named_modules():
        if isinstance(module, GATConv):
            def capture(module, args, kwargs, key=key):
                attr = kwargs.get("edge_attr", args[2] if len(args) > 2 else None)
                received[key] = list(attr.shape) if attr is not None else None
            handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
    with torch.no_grad():
        baseline = model(batch).detach()
        for handle in handles:
            handle.remove()
        repeated = model(batch).detach()
        jitter = float((baseline - repeated).abs().max())
        scale = max(float(baseline.abs().max()), 1.)
        tolerance = max(1e-6, jitter * 20, 8 * torch.finfo(baseline.dtype).eps * scale)
        if not torch.isfinite(baseline).all() or not torch.isfinite(repeated).all():
            raise ValueError(f"{name}: nonfinite predictions")
        if len(received) != (6 if name in ("E0", "E1") else 12):
            raise ValueError("Not all expected GAT layers executed")
        if not all((shape is None) == (name in ("E1", "E4")) for shape in received.values()):
            raise ValueError("GAT attribute switch did not reach all layers")
        sensitivity = {}
        for relation in RELATIONS:
            changed = batch.clone()
            attr = changed[relation].edge_attr
            ordinal = torch.arange(attr.size(0), device=attr.device, dtype=attr.dtype)
            if attr.size(1) == 4:
                attr[:, 0] = ordinal.remainder(2)
                attr[:, 1] = 64 + ordinal.remainder(128)
                attr[:, 2] = attr[:, 1] - 20
                attr[:, 3] = attr[:, 1] - 28
            else:
                attr[:, 0] = ordinal * 0.25 + 0.01
            # Diagnostic isolates one relation; normal input mirror checks are
            # bypassed only here, while nodes/edges/parameters remain fixed.
            if name in ("E0", "E1"):
                attrs = {r: changed[r].edge_attr if name == "E0" else None for r in RELATIONS}
                prediction = model.reference(changed.x_dict, changed.edge_index_dict, attrs, changed)
            else:
                prediction = model(changed)
            if not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite attribute probe")
            sensitivity[str(relation)] = float((prediction - baseline).abs().max())
    if name in ("E1", "E4"):
        if max(sensitivity.values()) > tolerance:
            raise ValueError(f"No Edge attributes affect output beyond device jitter: {sensitivity}")
    elif max(sensitivity.values()) <= tolerance:
        raise ValueError(f"No measurable attribute effect on selected train batch: {sensitivity}; tolerance={tolerance}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    parameter_before = state_hash(model.named_parameters())
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(batch)
    loss = F.nll_loss(predictions, batch.y.view(-1))
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite loss")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().max() > 0 for g in grads):
        raise ValueError("Missing, nonfinite, or all-zero gradients")
    edge_gradients = {key: float(module.lin_edge.weight.grad.abs().max())
        if module.lin_edge.weight.grad is not None else None
        for key, module in model.named_modules() if isinstance(module, GATConv)}
    gate_gradient = None
    if name in ("E2", "E3", "E4"):
        gate_grads = [p.grad for p in model.gate.parameters() if p.grad is not None]
        gate_gradient = max((float(g.abs().max()) for g in gate_grads), default=0.)
        if name == "E2" and gate_grads:
            raise ValueError("Fixed-weight fusion unexpectedly backpropagated through gate")
        if name in ("E3", "E4") and gate_gradient <= 0:
            raise ValueError("Adaptive gate has no nonzero gradient on selected batch")
    optimizer.step()
    if parameter_before == state_hash(model.named_parameters()):
        raise ValueError("Optimizer did not change any model parameter")
    for node in ("flow", "packet"):
        if not torch.equal(batch[node].x, snapshot[node].x):
            raise ValueError("Model modified node features")
    for relation in RELATIONS:
        for key in ("edge_index", "edge_attr"):
            if not torch.equal(batch[relation][key], snapshot[relation][key]):
                raise ValueError("Model modified graph topology or attributes")
    if not torch.equal(batch.y, snapshot.y):
        raise ValueError("Model modified labels")
    return {"initial_state_sha256": initial, "finite_loss": float(loss.detach()),
            "actual_GAT_inputs": received, "attribute_sensitivity": sensitivity,
            "repeat_forward_jitter": jitter, "comparison_tolerance": tolerance,
            "edge_projection_gradient_max": edge_gradients, "gate_gradient_max": gate_gradient,
            "parameter_update_verified": True, "input_graph_unchanged": True,
            "parameter_count": sum(p.numel() for p in model.parameters())}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable; no CPU fallback")
        if device.type not in ("cuda", "cpu"):
            raise ValueError("Only explicit CPU or CUDA smoke is supported")
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        frozen = FrozenGraphs(args.split_dir, args.expected_split_id, args.data_root)
        rows = frozen.training_candidates(args.candidates_per_class)
        loaded = [(row, frozen.load(row)) for row in rows]
        for _, graph in loaded:
            validate_graph(graph)
        multi = [i for i, (_, graph) in enumerate(loaded) if graph["packet"].x.size(0) >= 3]
        if len(multi) < 2:
            raise ValueError("Need at least two train graphs with >=3 packets; increase candidates-per-class")
        order = multi[:2] + [i for i in range(len(loaded)) if i not in multi[:2]]
        chosen = [loaded[i] for i in order[:args.batch_size]]
        graphs = [graph for _, graph in chosen]
        batch = Batch.from_data_list(graphs).to(device)
        write(args.output / "config.json", {"split_id": frozen.metadata["split_id"],
            "sample_manifest_sha256": frozen.metadata["sample_manifest_sha256"], "seed": args.seed,
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch": torch.__version__, "pyg": torch_geometric.__version__, "epochs": 0,
            "optimizer_steps_per_model": 1, "optimizer": "Adam", "lr": 0.001,
            "batch_size": len(graphs), "hidden_dim": 64, "loss": "plain NLL",
            "auxiliary_losses": False, "calibration": False, "class_weighting": False,
            "sample_partition": "train only; validation/test graph files are not loaded",
            "link_adapter": "read-time [E] -> [E,1] view after raw fingerprint verification",
            "samples": [{k: row[k] for k in ("sample_id", "group_id", "label", "path", "container_offset")} for row, _ in chosen],
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_hashes": {path: sha(ROOT / path) for path in (
                "scripts/grouped_graphs.py", "scripts/smoke_grouped_gpu.py", "scripts/audit_graph_split.py",
                "src/xgnid/model.py", "src/xgnid/edge_experiments.py", "third_party/GNN4ID/Utility/Model.py")}})
        results = {}
        for name in args.experiments:
            print(f"{name}: one-batch forward / edge probes / backward / update on {device}", flush=True)
            results[name] = exercise(name, batch, graphs[0], args.seed)
            write(args.output / "smoke_partial.json", results)
        pairs = {}
        for left, right in (("E0", "E1"), ("E2", "E3")):
            if left in results and right in results:
                equal = results[left]["initial_state_sha256"] == results[right]["initial_state_sha256"]
                pairs[f"{left}/{right}"] = equal
                if not equal:
                    raise ValueError(f"Paired initial parameters/buffers differ: {left}/{right}")
        report = {"status": "completed", "exit_code": 0, "split_id": frozen.metadata["split_id"],
            "device": str(device), "models": results, "paired_initial_states_identical": pairs,
            "verified_candidate_graphs": len(loaded), "batch_graphs": len(chosen),
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "train-only one-batch smoke; no epochs, performance evaluation or formal training"}
        write(args.output / "smoke.json", report)
        write(args.output / "status.json", {"status": "completed", "exit_code": 0})
        print("Smoke completed. Report:", args.output / "smoke.json", flush=True)
        return report
    except Exception:
        (args.output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write(args.output / "status.json", {"status": "failed", "exit_code": 1})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--expected-split-id", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidates-per-class", type=int, default=4)
    parser.add_argument("--experiments", nargs="+", choices=["E0", "E1", "E2", "E3", "E4"], default=["E0", "E1", "E2", "E3"])
    args = parser.parse_args()
    if args.batch_size < 2 or args.candidates_per_class < 1:
        parser.error("batch-size >= 2 and candidates-per-class >= 1 are required")
    run(args)
