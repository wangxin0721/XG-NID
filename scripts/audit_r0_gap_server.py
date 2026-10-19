"""Server-only read-only R0 audit: artifacts, CSV/graph samples, CPU forwards.

No training, backward, optimization, graph-file generation, or new test
inference. Only a new diagnostic directory is written. CSV-to-graph ordinal
matching is a hypothesis checked on fixed samples, not assumed provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time


LABELS = ["Benign", "WebBased", "Spoofing", "Recon", "Mirai", "Dos", "DDos", "BruteForce"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(root, value):
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def near(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--metadata-only", action="store_true", help="Skip torch/CSV/graph/forward diagnostics")
    args = parser.parse_args()
    require(os.name == "posix", "Run this audit on the Linux server, not the local development machine")
    sys.dont_write_bytecode = True
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.nice(10)
    repo = args.repo.resolve()
    output = repo / "outputs" / ("r0_gap_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "xgnid.r0.gap-audit.v1", "tool_sha256": sha(Path(__file__)),
              "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
              "training_started": False, "gpu_used": False, "test_inference_performed": False,
              "test_access": "Read existing predictions and source feature samples; no new test predictions",
              "historical_author_sample_identity_verified": False, "root_cause_established": False,
              "limitations": ["CSV ordinal correspondence is tested only on 4 predetermined rows per class/partition",
                  "Matching samples does not prove the full CSV-to-graph mapping or author sample identity",
                  "Historical temporal window, MAC filtering, subclass composition and paper run RNG remain unverified",
                  "Public notebook defaults to SAGE; paper describes GAT; paper checkpoint/run association is missing",
                  "No hyperparameter selection, test-driven checkpoint selection, or automatic rerun"]}

    def check(name, function):
        print("Checking " + name, flush=True)
        write(output / "status.json", {"status": "running", "phase": name, "pid": os.getpid()})
        try:
            result = function()
            report["checks"].append({"name": name, "status": "passed", "detail": result})
            return result
        except Exception as exc:
            report["checks"].append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            return None

    context = {}

    def artifacts():
        launch = read(repo / "outputs/r0_full_launch.json")
        run = within(repo, launch["output"])
        summary, config = read(run / "summary.json"), read(run / "config.json")
        binding = summary["binding"]
        require(read(run / "status.json") == {"status": "completed", "exit_code": 0}, "R0 did not complete")
        require(run.with_suffix(".exitcode").read_text().strip() == "0", "Supervisor exit code is not zero")
        require((summary["status"], summary["exit_code"], summary["mode"], summary["epochs_completed"], summary["selection"])
                == ("completed", 0, "full", 30, "last_epoch"), "Unexpected final R0 status/protocol")
        require(summary["parameters"] == 431704 and summary["claim"] == "protocol_reimplementation", "Wrong R0 model/claim")
        require(config["binding"] == binding, "Saved config binding differs from summary")
        require(all(launch[k] == binding[k] for k in ("manifest_id", "config_id")), "Launch identity differs")
        require(sha(run / "last.pt") == summary["last_checkpoint_sha256"], "Final checkpoint SHA differs")
        require(read(run / "metrics.json") == summary["test"], "metrics.json differs from summary.test")
        code = within(repo, launch["code_root"])
        for name, expected in binding["code"].items():
            require(identity(within(code, name).read_text(encoding="utf-8-sig")) == expected, "Training source changed: " + name)
        canonical = read(code / "configs/r0_edge.json")
        require(identity(canonical) == binding["config_id"], "Protocol config identity differs")
        require(all(config[k] == v for k, v in canonical.items()), "Saved configuration differs from frozen protocol")
        with (run / "epochs.jsonl").open(encoding="utf-8") as stream:
            epochs = [json.loads(line) for line in stream]
        require([e["epoch"] for e in epochs] == list(range(1, 31)), "Epoch records missing or unordered")
        require(all(e["train_samples"] == 160000 and e["lr"] == .01 and math.isfinite(e["train_loss"]) for e in epochs), "Bad training counts/lr/loss")
        directory = within(repo, launch["manifest_dir"])
        meta = read(directory / "manifest.json")
        content = dict(meta)
        require(content.pop("manifest_id") == binding["manifest_id"] == identity(content), "Manifest identity differs")
        require(sha(directory / "samples.jsonl") == meta["sample_manifest_sha256"], "Manifest rows SHA differs")
        require(sha(directory / "membership_evidence.txt") == meta["membership_evidence_sha256"], "Membership evidence SHA differs")
        evidence = read(directory.parent / "source_evidence.json")
        require(read(directory / "membership_evidence.txt") == evidence, "Source evidence differs from frozen membership evidence")
        require(meta["labels"] == LABELS, "Class order differs")
        counts = {p: Counter() for p in ("train", "test")}
        targets = {(p, i): {0, int(n) // 3, 2 * int(n) // 3, int(n) - 1}
                   for p, classes in meta["partition_label_counts"].items() for i, n in ((int(k), v) for k, v in classes.items())}
        selected, test, names = {p: [] for p in counts}, [], {p: [] for p in counts}
        numeric_order = {p: True for p in counts}
        with (directory / "samples.jsonl").open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                part, label = row["partition"], row["label"]
                require(row["manifest_index"] == index and part in counts and label in range(8), "Invalid manifest row")
                filename = Path(row["path"]).name
                match = re.fullmatch(r"data_(test_)?(\d+)\.pt", filename)
                require(match and bool(match[1]) == (part == "test"), "Unexpected official-style filename")
                numeric_order[part] &= int(match[2]) == sum(counts[part].values())
                if len(names[part]) < 12:
                    names[part].append(filename)
                if counts[part][label] in targets[part, label]:
                    selected[part].append(row)
                counts[part][label] += 1
                if part == "test":
                    test.append(row)
        require({p: {str(i): n for i, n in c.items()} for p, c in counts.items()} == meta["partition_label_counts"], "Manifest class counts differ")
        require(sum(counts["train"].values()) == 160000 and len(test) == 25557, "Wrong partition sizes")
        require(all(len(selected[p]) == 32 for p in selected), "Fixed diagnostic sample selection failed")
        context.update(launch=launch, run=run, summary=summary, config=canonical, binding=binding,
                       code=code, directory=directory, meta=meta, evidence=evidence, selected=selected, test=test)
        return {"output": str(run), "checkpoint_sha256": summary["last_checkpoint_sha256"],
                "manifest_id": binding["manifest_id"], "runtime_recorded": binding["runtime"],
                "wall_seconds": summary["wall_seconds"], "epochs": len(epochs),
                "base_order_is_official_numeric": numeric_order, "first_filenames": names,
                "order_note": "Different base indexing changes seed/sample initialization and batch trajectories; causality for the paper gap is not established"}

    def predictions():
        import csv
        cm = [[0] * 8 for _ in range(8)]
        path = context["run"] / "predictions.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            require(rows.fieldnames == ["sample_id", "label", "prediction"], "Prediction CSV schema differs")
            count = 0
            for count, prediction in enumerate(rows, 1):
                require(count <= len(context["test"]), "Extra prediction rows")
                row = context["test"][count - 1]
                y, pred = int(prediction["label"]), int(prediction["prediction"])
                require(prediction["sample_id"] == row["sample_id"] and y == row["label"] and pred in range(8), "Prediction identity/order/label differs")
                cm[y][pred] += 1
        expected = context["summary"]["test"]
        require(count == expected["samples"] == 25557, "Prediction count differs")
        require(cm == expected["confusion_matrix_true_rows_predicted_columns"], "Predictions do not reconstruct saved confusion matrix")
        support = [sum(row) for row in cm]
        predicted = [sum(row[i] for row in cm) for i in range(8)]
        f1 = [2 * cm[i][i] / (support[i] + predicted[i]) for i in range(8)]
        require(near(sum(cm[i][i] for i in range(8)) / count, expected["accuracy"]) and
                near(sum(f1) / 8, expected["macro_f1"]) and
                near(sum(f * n for f, n in zip(f1, support)) / count, expected["weighted_f1"]), "Saved metric arithmetic differs")
        for i, label in enumerate(LABELS):
            item = expected["classification_report"][label]
            require(item["support"] == support[i] and near(item["f1-score"], f1[i]) and
                    near(item["precision"], cm[i][i] / predicted[i] if predicted[i] else 0) and
                    near(item["recall"], cm[i][i] / support[i]), "Per-class metrics differ: " + label)
        return {"prediction_sha256": sha(path), "rows": count, "sample_identity_and_order": "matched frozen test manifest",
                "reconstructed_macro_f1": sum(f1) / 8, "new_test_inference": False}

    def cpu_diagnostics():
        import torch
        import torch_geometric
        import pandas as pd
        from torch_geometric.data import HeteroData, Batch
        from torch_geometric.transforms import ToUndirected
        from types import SimpleNamespace
        from importlib import metadata
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(context["code"] / "scripts"))
        import r0_data
        from xgnid.edge_experiments import ReferenceEdge
        require(r0_data.ROOT.resolve() == context["code"], "Wrong worktree imported")
        lock = read(context["code"] / "docs/r0_source_lock.json")
        utility_path = context["code"] / "third_party/GNN4ID/Utility/Functions.py"
        require(identity(utility_path.read_text(encoding="utf-8-sig")) == lock["files"]["Utility/Functions.py"]["normalized_text_identity"], "Graph helper differs from locked official source")
        utility = import_file("r0_gap_official_functions", utility_path)
        helpers = SimpleNamespace(single_file=True, include_packetflag=False, include_packetpayload=True)
        graphs, csv_results = {p: [] for p in ("train", "test")}, {}
        root = Path(context["meta"]["source_data_root"]).resolve()
        for part in graphs:
            entry = context["evidence"]["csv_sources"][part]
            path = Path(entry["path"])
            print(f"Hashing candidate {part} CSV: {path}", flush=True)
            before = path.stat()
            require(before.st_size == entry["bytes"] and sha(path) == entry["sha256"], "Candidate CSV changed after source freezing: " + part)
            targets = {int(re.fullmatch(r"data_(?:test_)?(\d+)\.pt", Path(r["path"]).name)[1]): r for r in context["selected"][part]}
            found, offset, classes, columns = {}, 0, Counter(), None
            for chunk in pd.read_csv(path, chunksize=2048):
                if columns is None:
                    columns = list(chunk.columns)
                classes.update(int(x) for x in chunk["Label"])
                for ordinal in targets:
                    if offset <= ordinal < offset + len(chunk):
                        found[ordinal] = chunk.iloc[ordinal - offset].copy()
                offset += len(chunk)
                if offset % 32768 == 0:
                    print(f"Read {part} CSV rows={offset}", flush=True)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Candidate CSV changed while reading")
            require({str(k): v for k, v in classes.items()} == context["meta"]["partition_label_counts"][part], "Candidate CSV class counts differ")
            packet_columns = ["udps.payload_data", "udps.delta_time", "udps.packet_direction", "udps.ip_size", "udps.transport_size", "udps.payload_size",
                              "udps.syn", "udps.cwr", "udps.ece", "udps.urg", "udps.ack", "udps.psh", "udps.rst", "udps.fin"]
            flow_columns = [name for name in columns if name not in packet_columns + ["Label"]]
            require(len(flow_columns) == 82, "Candidate CSV does not yield Flow82")
            compared = []
            for ordinal, row in sorted(targets.items()):
                require(ordinal in found, "Candidate CSV has no numeric ordinal for selected graph")
                source = found[ordinal]
                for name in packet_columns[:6]:
                    source[name] = source[name].strip("][").replace("'", "").split(", ")
                expected = HeteroData()
                expected["flow"].x = utility.NIDSDataset._get_flow_node_features(helpers, source)
                expected["packet"].x = utility.NIDSDataset._get_packet_node_features(helpers, source)
                n = expected["packet"].x.size(0)
                contain, link = ("flow", "contain", "packet"), ("packet", "link", "packet")
                expected[contain].edge_index = utility.NIDSDataset._get_contain_edge_index(helpers, n)
                expected[link].edge_index = utility.NIDSDataset._get_link_edge_index(helpers, n)
                expected[contain].edge_attr = utility.NIDSDataset._get_contain_edge_features(helpers, source)
                expected[link].edge_attr = utility.NIDSDataset._get_link_edge_features(helpers, source)
                expected.y = torch.tensor(int(source["Label"]), dtype=torch.long)
                expected = ToUndirected()(expected)
                original = r0_data.load_container(r0_data.location(root, row), root)[row["container_offset"]]
                r0_data.check(original)
                require((r0_data.fingerprint(original, True), r0_data.fingerprint(original, False), original.y.item())
                        == (row["edge_hash"], row["node_hash"], row["label"]), "Source graph changed after R0 freeze")
                # Compare logical link tensors after the same view-only normalization as R0.
                actual = original.clone()
                for graph in (expected, actual):
                    if graph[link].edge_attr.ndim == 1:
                        graph[link].edge_attr = graph[link].edge_attr.unsqueeze(-1)
                differences = []
                if actual.node_types != expected.node_types or actual.edge_types != expected.edge_types:
                    differences.append({"field": "metadata_insertion_order", "actual": actual.metadata(), "expected": expected.metadata()})
                def compare(name, a, b):
                    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                        d = {"field": name, "actual_shape": list(a.shape), "expected_shape": list(b.shape)}
                        if a.shape == b.shape and a.numel():
                            d["max_abs_delta"] = float((a.double() - b.double()).abs().max())
                        differences.append(d)
                for node in ("flow", "packet"):
                    compare(node + ".x", actual[node].x, expected[node].x)
                for relation in expected.edge_types:
                    for key in ("edge_index", "edge_attr"):
                        compare(str(relation) + "." + key, actual[relation][key], expected[relation][key])
                if actual.y.item() != expected.y.item():
                    differences.append({"field": "label", "actual": actual.y.item(), "expected": expected.y.item()})
                compared.append({"sample_id": row["sample_id"], "path": row["path"], "candidate_csv_row": ordinal,
                                 "label": row["label"], "differences": differences})
                graphs[part].append(actual)
            csv_results[part] = {"csv_sha256": entry["sha256"], "csv_rows": offset, "flow_feature_order": flow_columns,
                "temporal_columns": [k for k in flow_columns if k.startswith(("Rolling", "Unique_", "packet_size"))],
                "mac_columns_present": [k for k in columns if k in ("src_mac", "dst_mac")],
                "sampled_graphs": compared, "matching_samples": sum(not x["differences"] for x in compared),
                "numeric_row_correspondence_is_a_hypothesis": True}
        write(output / "csv_graph_samples.json", csv_results)
        require(csv_results["train"]["flow_feature_order"] == csv_results["test"]["flow_feature_order"], "Train/test candidate CSV feature orders differ")
        # A failed candidate ordinal mapping must not be described as a proven graph-generation bug.
        mismatches = sum(len(v["sampled_graphs"]) - v["matching_samples"] for v in csv_results.values())
        payload = torch.load(context["run"] / "last.pt", map_location="cpu", weights_only=False)
        require(payload["epoch"] == 30 and payload["binding"] == context["binding"] and payload["config"] == context["config"], "Checkpoint internal identity differs")
        state = payload["model_state"]
        require(all(k.startswith("reference.") for k in state) and all(torch.isfinite(t).all().item() for t in state.values()), "Nonreference or nonfinite model state")
        steps = sorted({int(v["step"].item()) for v in payload["optimizer_state"]["state"].values()})
        counters = {k: int(v.item()) for k, v in state.items() if k.endswith("num_batches_tracked")}
        expected_steps = 30 * (160000 // 64)
        require(steps == [expected_steps] and len(counters) == 4 and set(counters.values()) == {expected_steps}, "Unexpected optimizer/BatchNorm training counters")
        groups = payload["optimizer_state"]["param_groups"]
        require(len(groups) == 1 and groups[0]["lr"] == .01 and groups[0]["weight_decay"] == 0, "Saved optimizer settings differ")
        batch = Batch.from_data_list(graphs["train"][:16])
        wrapped = ReferenceEdge(graphs["train"][0], use_edge_attr=True).eval()
        model_file = context["code"] / "third_party/GNN4ID/Utility/Model.py"
        official = import_file("r0_gap_official_model", model_file).HeteroGNN_Edge(graphs["train"][0], {"hidden_size": 64, "eps": 1.0}).eval()
        with torch.no_grad():
            wrapped(batch)
            official(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
        wrapped.load_state_dict(state, strict=True)
        official.load_state_dict({k[len("reference."):]: v for k, v in state.items()}, strict=True)
        require(sum(p.numel() for p in wrapped.parameters()) == 431704, "Checkpoint model parameter count differs")
        with torch.no_grad():
            a = wrapped(batch)
            b = official(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
            singles = torch.cat([wrapped(Batch.from_data_list([g])) for g in graphs["train"][:16]], dim=0)
            loss = official.loss(b, batch.y.reshape(-1))
            wrapped_loss = torch.nn.functional.nll_loss(a, batch.y.reshape(-1))
        delta, batch_delta = float((a-b).abs().max()), float((a-singles).abs().max())
        require(torch.isfinite(a).all().item() and torch.allclose(a, b, rtol=1e-5, atol=1e-5) and near(float(loss), float(wrapped_loss)), "Wrapper differs from official checkpoint forward/loss")
        require(torch.allclose(a, singles, rtol=1e-5, atol=1e-5), "Batch size changes evaluation beyond tolerance")
        detail = {"runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "pyg": torch_geometric.__version__,
                  "pandas": pd.__version__, "numpy": metadata.version("numpy"), "sklearn": metadata.version("scikit-learn")},
                  "device": "cpu", "source_graphs_read": 64, "new_test_inference": False,
                  "csv_graph_report": str(output / "csv_graph_samples.json"),
                  "csv_graph_matching_samples": {p: r["matching_samples"] for p, r in csv_results.items()},
                  "candidate_mapping_mismatches": mismatches, "optimizer_step_counts": steps,
                  "batchnorm_counters": counters, "wrapper_official_max_logprob_delta": delta,
                  "batch_vs_single_max_logprob_delta": batch_delta, "training_sample_nll": float(loss)}
        require(sha(context["run"] / "last.pt") == context["summary"]["last_checkpoint_sha256"], "Checkpoint changed during audit")
        context["candidate_mapping_mismatches"] = mismatches
        return detail

    if check("formal_run_artifacts_and_manifest", artifacts) is not None:
        check("saved_predictions_and_metric_reconstruction", predictions)
        if not args.metadata_only:
            check("csv_graph_checkpoint_and_cpu_forward", cpu_diagnostics)
    failed = [c["name"] for c in report["checks"] if c["status"] == "failed"]
    review = bool(failed or context.get("candidate_mapping_mismatches"))
    report.update(status="findings_require_review" if review else "checks_passed_with_unresolved_protocol_differences",
                  failed_checks=failed, metadata_only=args.metadata_only,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    write(output / "report.json", report)
    write(output / "status.json", {"status": report["status"], "exit_code": int(review)})
    print(json.dumps({"status": report["status"], "failed_checks": failed,
        "checks": report["checks"], "report": str(output / "report.json"),
        "root_cause_established": False, "training_started": False, "gpu_used": False,
        "test_inference_performed": False}, ensure_ascii=False, indent=2), flush=True)
    return int(review)


if __name__ == "__main__":
    raise SystemExit(main())
