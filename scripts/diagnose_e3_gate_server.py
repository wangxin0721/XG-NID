"""Server-only validation diagnostics of completed E2/E3-B512 checkpoints.

Metadata mode imports no ML runtime. Diagnose restores verified checkpoints,
reproduces saved validation matrices, then exports aligned branch/gate caches.
No test graph tensors, backward pass, optimizer, calibration or new training.
"""
from __future__ import annotations

import argparse
import gzip
import math
import os
from pathlib import Path
import sys
import time
import traceback

from e3_gate_study import (activate_gpu, checked_gpu, choose_alpha, guard_graph_loads,
    identity, initial_model, inside, matrix_metrics, read, reference_metadata, require,
    runtime, sha, write)


class Probabilities:
    def __init__(self):
        self.n, self.brier, self.nll = 0, 0., 0.
        self.bins = [[0, 0., 0] for _ in range(10)]

    def add(self, probabilities, label):
        require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), "Invalid class probabilities")
        require(abs(sum(probabilities) - 1) < 1e-5, "Probabilities do not sum to one")
        predicted = max(range(8), key=lambda k: probabilities[k])
        confidence = probabilities[predicted]
        self.n += 1
        self.brier += sum((p - int(k == label)) ** 2 for k, p in enumerate(probabilities))
        self.nll -= math.log(max(probabilities[label], 1e-12))
        bucket = self.bins[min(9, int(confidence * 10))]
        bucket[0] += 1
        bucket[1] += confidence
        bucket[2] += int(predicted == label)

    def result(self):
        return {"samples": self.n, "brier_multiclass_sum": self.brier / self.n,
                "nll": self.nll / self.n,
                "ece_10_equal_width_bins": sum(abs(c - a) for n, c, a in self.bins if n) / self.n,
                "bins": [{"index": i, "samples": n, "mean_confidence": c / n if n else None,
                          "accuracy": a / n if n else None} for i, (n, c, a) in enumerate(self.bins)]}


def correctness_groups():
    return {key: {"samples": 0, "dynamic_correct": 0, "equal_correct": 0,
                  "gate_picks_correct_branch": 0, "raw_gate_picks_correct_branch": 0}
            for key in ("both_correct", "flow_only_correct", "packet_only_correct", "both_wrong")}


def webbased_focus(metrics, labels):
    """Describe validation errors for the already identified class, not select a model."""
    cm = metrics["confusion_matrix"]
    index = labels.index("WebBased")
    support, predicted = sum(cm[index]), sum(row[index] for row in cm)
    tp = cm[index][index]
    return {"partition": "val", "support": support, "correct": tp,
            "precision": tp / predicted if predicted else 0., "recall": tp / support if support else 0.,
            "f1": metrics["per_class_f1"][index],
            "true_WebBased_predicted_as": {label: cm[index][j] for j, label in enumerate(labels)},
            "true_other_predicted_WebBased": {label: cm[i][index] for i, label in enumerate(labels) if i != index}}


def diagnose_model(core, args, spec, reference, name, device):
    import json
    torch = core.torch
    frozen = core.FrozenGraphs(inside(args.repo, spec["split_directory"]), spec["split_id"])
    loads = guard_graph_loads(frozen, {"train", "val"})
    model, _ = initial_model(core, frozen, device)
    folder = inside(args.repo, spec["references"][name]) / name
    saved_cfg, saved_metrics = read(folder / "config.json"), read(folder / "metrics.json")
    checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=False)
    require(checkpoint["config"] == saved_cfg and checkpoint["epoch"] == reference["references"][name]["best_epoch"], "Checkpoint configuration/epoch differs")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    model.use_gate = name == "E3"
    model.eval()
    state_before = core.state_hash(model.state_dict().items())
    captured = {}
    handles = [model.flow_classifier.register_forward_pre_hook(lambda module, values: captured.update(hf=values[0])),
               model.packet_classifier.register_forward_pre_hook(lambda module, values: captured.update(hp=values[0])),
               model.gate.register_forward_hook(lambda module, values, result: captured.update(g=result))]
    rows = core.select_rows(frozen, "val")
    # Keep the guarded loader in the parent process so its forbidden-partition
    # assertion and exact graph-load counter also cover every graph read.
    loader = core.DataLoader(core.GraphRows(frozen, rows), batch_size=16, shuffle=False, num_workers=0)
    keys = ["dynamic", "flow_only", "packet_only", "equal"] + (["pure_gate"] if name == "E3" else []) + [f"alpha_{i:02d}" for i in range(21)]
    matrices = {key: [[0] * 8 for _ in range(8)] for key in keys}
    calibration = {key: Probabilities() for key in ("dynamic", "flow_only", "packet_only", "equal")}
    groups = correctness_groups()
    groups_by_class = {label: correctness_groups() for label in core.LABELS}
    proxies = {branch: {key: {"samples": 0, "sum_entropy_complement": 0., "sum_margin": 0.}
                       for key in ("correct", "incorrect")} for branch in ("flow", "packet")}
    proxy_bins = {branch: [{"samples": 0, "correct_Hz_predictions": 0, "sum_proxy": 0.} for _ in range(10)]
                  for branch in ("flow", "packet")}
    n, sum_x, sum_y = 0, torch.zeros(64, dtype=torch.float64), torch.zeros(64, dtype=torch.float64)
    xx, yy, xy = [torch.zeros((64, 64), dtype=torch.float64) for _ in range(3)]
    weight_values = []
    identity_rows = []
    directory = args.output / name
    directory.mkdir()
    cache_path = directory / "validation_cache.jsonl.gz"
    tick = time.perf_counter()
    try:
        with gzip.open(cache_path, "xt", encoding="utf-8") as stream, torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                captured.clear()
                dynamic = model(batch)
                require(bool(torch.isfinite(dynamic).all()), "Nonfinite original prediction")
                cache = model._last_cache
                zf, zp = cache["flow_logits"], cache["packet_logits"]
                w = cache["gate_weights"]
                g = captured.get("g", w)
                hf, hp = captured["hf"], captured["hp"]
                require(hf.shape[1] == hp.shape[1] == 64 and zf.shape[1] == zp.shape[1] == 8, "Unexpected representations")
                values = {"dynamic": dynamic.exp(), "flow_only": torch.softmax(model.shared_head(zf), -1),
                          "packet_only": torch.softmax(model.shared_head(zp), -1),
                          "equal": torch.softmax(model.shared_head(.5 * (zf + zp)), -1)}
                if name == "E3":
                    values["pure_gate"] = torch.softmax(model.shared_head(g[:, :1] * zf + g[:, 1:] * zp), -1)
                for i in range(21):
                    alpha = i / 20
                    values[f"alpha_{i:02d}"] = torch.softmax(model.shared_head(alpha * zf + (1 - alpha) * zp), -1)
                require(all(bool(torch.isfinite(p).all()) for p in values.values()), "Nonfinite intervention probabilities")
                probabilities = {key: value.cpu().tolist() for key, value in values.items()}
                predictions = {key: value.argmax(-1).cpu().tolist() for key, value in values.items()}
                fields = {"h_flow": hf.cpu().tolist(), "h_packet": hp.cpu().tolist(),
                          "z_flow": zf.cpu().tolist(), "z_packet": zp.cpu().tolist(),
                          "raw_gate": g.cpu().tolist() if name == "E3" else [None] * batch.num_graphs,
                          "final_weights": w.cpu().tolist(), "flow_proxy": cache["flow_conf"].cpu().tolist(),
                          "packet_proxy": cache["packet_conf"].cpu().tolist(),
                          "flow_margin": cache["flow_margin"].cpu().tolist(), "packet_margin": cache["packet_margin"].cpu().tolist(),
                          "z_cosine": cache["branch_agreement"].cpu().tolist()}
                truth = batch.y.view(-1).tolist()
                for j, sample_id in enumerate(batch.sample_id):
                    row = rows[n + j]
                    label = truth[j]
                    require(sample_id == row["sample_id"] and label == row["label"], "Validation identity/order changed")
                    identity_rows.append((sample_id, label))
                    for key in keys:
                        matrices[key][label][predictions[key][j]] += 1
                    for key, tracker in calibration.items():
                        tracker.add(probabilities[key][j], label)
                    fc, pc = predictions["flow_only"][j] == label, predictions["packet_only"][j] == label
                    group_key = "both_correct" if fc and pc else "flow_only_correct" if fc else "packet_only_correct" if pc else "both_wrong"
                    for group in (groups[group_key], groups_by_class[core.LABELS[label]][group_key]):
                        group["samples"] += 1
                        group["dynamic_correct"] += int(predictions["dynamic"][j] == label)
                        group["equal_correct"] += int(predictions["equal"][j] == label)
                        group["gate_picks_correct_branch"] += int(fc if fields["final_weights"][j][0] >= .5 else pc)
                        raw = fields["raw_gate"][j]
                        if raw is not None:
                            group["raw_gate_picks_correct_branch"] += int(fc if raw[0] >= .5 else pc)
                    for branch, correct in (("flow", fc), ("packet", pc)):
                        item = proxies[branch]["correct" if correct else "incorrect"]
                        item["samples"] += 1
                        item["sum_entropy_complement"] += fields[f"{branch}_proxy"][j]
                        item["sum_margin"] += fields[f"{branch}_margin"][j]
                        proxy_value = fields[f"{branch}_proxy"][j]
                        require(math.isfinite(proxy_value), "Nonfinite entropy proxy")
                        bucket = proxy_bins[branch][max(0, min(9, int(proxy_value * 10)))]
                        bucket["samples"] += 1
                        bucket["correct_Hz_predictions"] += int(correct)
                        bucket["sum_proxy"] += proxy_value
                    weights = fields["final_weights"][j]
                    require(all(math.isfinite(v) and 0 <= v <= 1 for v in weights) and abs(sum(weights) - 1) < 1e-5, "Invalid gate weights")
                    weight_values.append(weights[0])
                    stream.write(json.dumps({"sample_id": sample_id, "label": label, "node_hash": row["node_hash"],
                        **{key: value[j] for key, value in fields.items()},
                        "probabilities": {key: probabilities[key][j] for key in ("dynamic", "flow_only", "packet_only", "equal", "pure_gate") if key in probabilities}},
                        allow_nan=False) + "\n")
                x, y = hf.cpu().double(), hp.cpu().double()
                sum_x += x.sum(0)
                sum_y += y.sum(0)
                xx += x.T @ x
                yy += y.T @ y
                xy += x.T @ y
                n += batch.num_graphs
                if n % 1024 == 0:
                    print(json.dumps({"model": name, "validation_samples": n, "total": len(rows)}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    require(n == 13446 and loads["test"] == 0, "Validation incomplete or test graphs touched")
    result_metrics = {key: matrix_metrics(cm) for key, cm in matrices.items()}
    saved = saved_metrics["best_validation"]
    require(matrices["dynamic"] == saved["confusion_matrix"], "Saved validation matrix not reproduced; reject diagnosis")
    require(all(abs(result_metrics["dynamic"][key] - saved[key]) <= 1e-10 for key in ("accuracy", "macro_f1", "weighted_f1")), "Saved validation metrics not reproduced")
    require(core.state_hash(model.state_dict().items()) == state_before, "Evaluation changed model state")
    require(sha(folder / "best.pt") == reference["references"][name]["checkpoints"]["best.pt"], "Checkpoint file changed")
    alpha_scores = {i: result_metrics[f"alpha_{i:02d}"] for i in range(21)}
    chosen = choose_alpha(alpha_scores)
    centered_xx, centered_yy, centered_xy = xx - torch.outer(sum_x, sum_x) / n, yy - torch.outer(sum_y, sum_y) / n, xy - torch.outer(sum_x, sum_y) / n
    denominator = float((centered_xx.square().sum() * centered_yy.square().sum()).sqrt())
    ordered_weights = sorted(weight_values)
    result = {"samples": n, "saved_validation_reproduced": True, "model_state_unchanged": True,
              "metrics": result_metrics, "selected_alpha": chosen / 20, "selection_partition": "val",
              "interventions_are_not_independently_trained_models": True,
              "branch_correctness_groups": groups, "branch_correctness_groups_by_class": groups_by_class,
              "proxy_conditioned_on_Hz_correctness": proxies, "entropy_proxy_bins_against_Hz_correctness": proxy_bins,
              "raw_gate_applicable": name == "E3", "raw_gate_note": "E2 bypasses its untrained gate; no raw/pure-gate result is reported" if name == "E2" else "Original learned gate before proxy mixing",
              "webbased_validation_focus": {key: webbased_focus(result_metrics[key], list(core.LABELS)) for key in
                  ("dynamic", "flow_only", "packet_only", "equal", "pure_gate", f"alpha_{chosen:02d}") if key in result_metrics},
              "calibration_of_Hz_predictions": {key: tracker.result() for key, tracker in calibration.items()},
              "linear_cka_of_64d_branch_embeddings": float(centered_xy.square().sum()) / denominator if denominator else None,
              "final_flow_weight_quantiles": {str(q): ordered_weights[round(q * (n - 1))] for q in (0, .05, .25, .5, .75, .95, 1)},
              "fraction_final_weight_above_0_95": sum(v > .95 or v < .05 for v in weight_values) / n,
              "ordered_validation_identity_sha256": identity(identity_rows), "graph_loads_parent_process": loads,
              "cache_sha256": sha(cache_path), "wall_seconds": time.perf_counter() - tick,
              "gradients": "not collected; inference/no_grad only", "test_evaluated": False}
    write(directory / "diagnostics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("check", "diagnose"), default="check")
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1))
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(os.name == "posix", "Run on the Linux server; local checks must not execute models")
    args.repo = args.repo.resolve()
    spec = read(args.config)
    reference = reference_metadata(args.repo, spec)
    if args.mode == "check":
        print(__import__("json").dumps({"status": "metadata_checks_passed", "reference": reference,
            "gpu_touched": False, "model_loaded": False, "training_started": False}, indent=2), flush=True)
        return 0
    require(args.physical_gpu is not None and args.output is not None, "Diagnose needs an explicit physical GPU and new output")
    args.output = inside(args.repo / "outputs", args.output.resolve())
    require(args.output.name.startswith("e3_gate_diagnostic_") and not args.output.exists(), "Use a new outputs/e3_gate_diagnostic_* directory")
    gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
    activate_gpu(gpu)
    args.output.mkdir()
    write(args.output / "status.json", {"status": "running", "mode": "diagnose", "pid": os.getpid()})
    try:
        sys.path.insert(0, str(args.repo / "scripts"))
        import train_grouped as core
        device = runtime(core)
        models = {name: diagnose_model(core, args, spec, reference, name, device) for name in ("E2", "E3")}
        require(models["E2"]["ordered_validation_identity_sha256"] == models["E3"]["ordered_validation_identity_sha256"], "Paired validation identities differ")
        report = {"schema": "xgnid.e3-gate-diagnostic.v1", "status": "completed", "exit_code": 0, "mode": "diagnose",
                  "partition": "val", "reference": reference, "hardware": gpu, "models": models,
                  "entry_sha256": sha(Path(__file__)), "shared_helper_sha256": sha(Path(__file__).with_name("e3_gate_study.py")),
                  "test_graphs_loaded": 0, "test_evaluated": False, "training_started": False,
                  "loader_workers": 0, "reason": "parent-process partition guard/load counts; evaluation batch16 and original graph fingerprints retained"}
        write(args.output / "report.json", report)
        write(args.output / "status.json", {"status": "completed", "exit_code": 0, "mode": "diagnose"})
        print(__import__("json").dumps({"status": "completed", "report": str(args.output / "report.json"),
            "models": {name: {"dynamic_val_macro_f1": m["metrics"]["dynamic"]["macro_f1"],
                "equal_val_macro_f1": m["metrics"]["equal"]["macro_f1"], "pure_gate_val_macro_f1": m["metrics"].get("pure_gate", {}).get("macro_f1"),
                "selected_alpha": m["selected_alpha"]} for name, m in models.items()}, "test_evaluated": False}, indent=2), flush=True)
    except BaseException as exc:
        write(args.output / "status.json", {"status": "failed", "exit_code": 130 if isinstance(exc, KeyboardInterrupt) else 1, "mode": "diagnose"})
        (args.output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
