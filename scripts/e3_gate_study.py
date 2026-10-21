"""Standard-library metadata, provenance and pure logic for the E3 gate study.

Model imports/loads are restricted to the explicitly invoked Linux entries.
This module never imports torch, opens a graph or deserializes a checkpoint.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import textwrap
from types import MethodType

SPLIT_ID = "a8ae9c934eec9fee970cc725e60051d0c709e0f16c7dda14749350ff2006975a"
INITIAL = "0270349d613f11b453b92e4e928a3c2e7b67d062b5e3ba73c8ba4fb58dbd521a"
CONTROLS = {"seed": 42, "batch_size": 512, "lr": .001, "hidden_dim": 64,
            "optimizer": "Adam", "early_stop_patience": 12, "early_stop_min_delta": 0.,
            "auxiliary_losses": False, "calibration": False, "class_weighting": False,
            "hard_negatives": False, "loss": "plain NLL", "device_family": "cuda"}
SOURCES = {"scripts/train_grouped.py", "scripts/grouped_graphs.py", "scripts/smoke_grouped_gpu.py",
           "scripts/audit_graph_split.py", "scripts/build_grouped_split.py", "src/xgnid/__init__.py",
           "src/xgnid/model.py", "src/xgnid/edge_experiments.py", "third_party/GNN4ID/Utility/Model.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inside(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def check_spec(spec):
    require(spec["schema"] == "xgnid.e3-gate-study.v1", "Wrong gate study schema")
    require(spec["kind"] == "exploratory_mechanism_study_after_prior_test_results"
            and spec["prior_test_results_available"] is True, "Exploratory registration missing")
    require(spec["split_id"] == SPLIT_ID and spec["reference_initial_state_sha256"] == INITIAL, "Reference identity changed")
    require(spec["controls"] == CONTROLS, "Controls changed; register a separate study")
    require((spec["epochs"], spec["workers"], spec["initialization_batch_size"], spec["evaluation_batch_size"]) ==
            (100, 2, 16, 16), "Training/evaluation budget changed")
    require(spec["diagnostics"]["partition"] == "val"
            and spec["diagnostics"]["alpha_grid_numerators_over_20"] == list(range(21)), "Validation-only grid changed")
    require(spec["diagnostics"]["saved_validation_metric_tolerance"] == 1e-10
            and spec["diagnostics"]["temperature_scaling"] is False, "Diagnostic evaluation changed")
    require(set(spec["variants"]) == {"E3a", "E3a_equal_init"}, "Unregistered variant")
    require(spec["screening"]["probe_warmup_updates"] == 1
            and spec["screening"]["probe_measured_updates"] == 3, "Probe budget changed")
    require(spec["budget"]["screening_round_configuration_ceiling"] == 2, "Screening scope changed")


def reference_metadata(repo, spec):
    """Validate original file bytes before importing model/data code."""
    check_spec(spec)
    references = {}
    for name in ("E2", "E3"):
        root = inside(repo, spec["references"][name])
        folder = root / name
        status, summary, metrics, cfg = [read(p) for p in
            (root / "status.json", root / "summary.json", folder / "metrics.json", folder / "config.json")]
        require((status.get("status"), status.get("exit_code")) == ("completed", 0), f"{name} reference not completed")
        require((summary["status"], summary["exit_code"], summary["mode"]) == ("completed", 0, "full"), "Reference summary incomplete")
        require(summary["models"][name] == metrics, f"{name} metrics/summary differ")
        require(cfg["controls"] == CONTROLS and (cfg["epochs"], cfg["workers"], cfg["initialization_batch_size"],
                cfg["evaluation_batch_size"]) == (100, 2, 16, 16), "Reference controls differ")
        require(cfg["experiment"] == name and cfg["mode"] == "full", "Wrong reference model")
        if name == "E3":
            require(cfg["protocol"] == "xgnid.e3-batch512-comparison.v1" and cfg["use_edge_attr"] is True
                    and cfg["use_gate"] is True, "Reference is not the original Edge adaptive E3")
        else:
            # The original E2 batch wrapper predates explicit use_edge_attr /
            # use_gate config fields. Its bound, frozen build_experiment('E2')
            # selects Edge and bypasses the gate; do not invent missing fields.
            require(cfg["protocol"] == "xgnid.e2-batch-comparison.v1", "Wrong E2 reference protocol")
        require(sha(folder / "config.json") == metrics["config_sha256"], "Reference config hash differs")
        binding = summary["binding"]
        require(cfg["binding"] == binding and binding["split_id"] == SPLIT_ID, "Reference binding differs")
        require(set(binding["sources"]) == SOURCES, "Required frozen sources missing")
        for path, digest in binding["sources"].items():
            require(sha(inside(repo, path)) == digest, f"Frozen source changed: {path}")
        require(metrics["parameter_count"] == 846800 and metrics["initial_state_sha256"] == INITIAL, "Wrong model/init identity")
        checkpoint_hashes = {}
        for file in ("best.pt", "last.pt"):
            checkpoint_hashes[file] = sha(folder / file)
            require(checkpoint_hashes[file] == metrics["checkpoint_sha256"][file], f"{name}/{file} SHA mismatch")
        references[name] = {"directory": root.relative_to(repo).as_posix(), "summary_sha256": sha(root / "summary.json"),
                            "config_sha256": sha(folder / "config.json"), "metrics_sha256": sha(folder / "metrics.json"),
                            "checkpoints": checkpoint_hashes, "sources": binding["sources"],
                            "samples_sha256": binding["samples_sha256"], "best_epoch": metrics["best_epoch"]}
    require(references["E2"]["sources"] == references["E3"]["sources"]
            and references["E2"]["samples_sha256"] == references["E3"]["samples_sha256"], "Paired references differ")
    split_dir = inside(repo, spec["split_directory"])
    split = read(split_dir / "split.json")
    require((split["status"], split["split_id"]) == ("completed", SPLIT_ID), "Wrong split")
    require(split["partition_sizes"] == {"train": 107621, "val": 13446, "test": 13448}, "Partition sizes differ")
    manifest = inside(split_dir, split["sample_manifest"])
    require(manifest.parent == split_dir and sha(manifest) == references["E3"]["samples_sha256"] == split["sample_manifest_sha256"], "Sample manifest differs")
    return {"schema": "xgnid.e3-gate-study-reference.v1", "specification_id": identity(spec),
            "split_id": SPLIT_ID, "references": references, "split_sha256": sha(split_dir / "split.json")}


def gpu_query(*args):
    result = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True)
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def compute_processes(uuid):
    rows = gpu_query("--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits")
    return [{"pid": int(r[1]), "name": r[2].strip()} for r in rows
            if len(r) == 3 and r[0].strip() == uuid and int(r[1]) != os.getpid()]


def checked_gpu(index, shared):
    rows = gpu_query("-i", str(index), "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits")
    require(len(rows) == 1 and len(rows[0]) == 6, "GPU identification failed")
    number, uuid, name, total, used, free = [v.strip() for v in rows[0]]
    require(int(number) == index and uuid.startswith("GPU-"), "GPU UUID/index mismatch")
    processes = compute_processes(uuid)
    require(shared or not processes, f"Physical GPU {index} is occupied: {processes}; explicit sharing needs --allow-shared-gpu")
    return {"physical_index": index, "uuid": uuid, "name": name, "total_mib": int(total),
            "used_mib_at_query": int(used), "free_mib_at_query": int(free), "allow_shared_gpu": shared,
            "other_processes_at_query": processes, "timing_scope": "sharing permitted; no isolated speed claim" if shared else "query only; not a reservation"}


def activate_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"


def runtime(core):
    require(platform.python_version() == "3.9.23" and core.torch.__version__ == "2.3.1+cu118"
            and core.torch_geometric.__version__ == "2.5.3", "Use original Linux wx runtime")
    require(core.torch.cuda.is_available(), "CUDA required; no CPU fallback")
    core.torch.cuda.set_device(0)
    core.torch.backends.cuda.matmul.allow_tf32 = False
    core.torch.backends.cudnn.allow_tf32 = False
    return core.torch.device("cuda:0")


def guard_graph_loads(frozen, allowed):
    original = frozen.load
    counts = {"train": 0, "val": 0, "test": 0}
    def guarded(row):
        part = row["partition"]
        require(part in allowed, f"Graph loading forbidden for partition {part}")
        counts[part] += 1
        return original(row)
    frozen.load = guarded
    return counts


def initial_model(core, frozen, device):
    core.seed_all(42)
    warm = [frozen.load(row) for row in frozen.training_candidates()]
    multi = [i for i, graph in enumerate(warm) if graph["packet"].x.size(0) >= 3]
    require(len(multi) >= 2, "Missing original initialization candidates")
    order = multi[:2] + [i for i in range(len(warm)) if i not in multi[:2]]
    batch = core.Batch.from_data_list([warm[i] for i in order[:16]]).to(device)
    model = core.build_experiment("E3", warm[0]).to(device)
    core.initialize(model, "E3", batch)
    require(core.state_hash(model.state_dict().items()) == INITIAL, "Original initialization changed")
    require(sum(p.numel() for p in model.parameters()) == 846800, "Model parameter count changed")
    return model, batch


def pure_gate_ast(source):
    tree = ast.parse(textwrap.dedent(source))
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
            and tree.body[0].name == "_graph_logits", "Unexpected forward definition")
    expected = [ast.dump(ast.parse(s).body[0], include_attributes=False) for s in (
        "gate_weights = 0.9 * gate_weights + 0.1 * torch.stack([flow_conf, packet_conf], dim=-1)",
        "gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)")]
    removed = []
    class RemoveMix(ast.NodeTransformer):
        def visit_Assign(self, node):
            key = ast.dump(node, include_attributes=False)
            if key in expected:
                removed.append(key)
                return None
            return self.generic_visit(node)
    result = RemoveMix().visit(tree)
    require(removed == expected, "Expected exactly two frozen E3 proxy-mixing statements")
    ast.fix_missing_locations(result)
    return result


def apply_variant(core, model, variant):
    require(variant in ("E3a", "E3a_equal_init"), "Unregistered variant")
    original = model._graph_logits.__func__
    source = inspect.getsource(original)
    transformed = pure_gate_ast(source)
    namespace = dict(original.__globals__)
    exec(compile(transformed, "<independent-e3-pure-gate-forward>", "exec"), namespace)
    model._graph_logits = MethodType(namespace["_graph_logits"], model)
    before = {name: core.state_hash([(name, value)]) for name, value in model.state_dict().items()}
    if variant == "E3a_equal_init":
        with core.torch.no_grad():
            model.gate.net[-1].weight.zero_()
            model.gate.net[-1].bias.zero_()
    after = {name: core.state_hash([(name, value)]) for name, value in model.state_dict().items()}
    changed = sorted(name for name in before if before[name] != after[name])
    expected = ["gate.net.2.bias", "gate.net.2.weight"] if variant == "E3a_equal_init" else []
    require(changed == expected, "Unexpected initialization tensor change")
    return {"variant": variant, "forward_ast_sha256": identity(ast.dump(transformed, include_attributes=False)),
            "removed_statements": 2, "initialization_changed_tensors": changed,
            "original_initial_state_sha256": INITIAL, "variant_initial_state_sha256": core.state_hash(model.state_dict().items())}


def matrix_metrics(cm):
    counts = [sum(row) for row in cm]
    n = sum(counts)
    require(n > 0 and len(cm) == 8 and all(len(row) == 8 for row in cm), "Invalid 8-class confusion matrix")
    fs = [2 * cm[i][i] / (counts[i] + sum(row[i] for row in cm))
          if counts[i] + sum(row[i] for row in cm) else 0. for i in range(8)]
    return {"samples": n, "accuracy": sum(cm[i][i] for i in range(8)) / n,
            "macro_f1": sum(fs) / 8, "weighted_f1": sum(f * c for f, c in zip(fs, counts)) / n,
            "per_class_f1": fs, "confusion_matrix": cm}


def choose_alpha(items):
    require(set(items) == set(range(21)), "All 21 registered validation alpha scores required")
    return max(items, key=lambda i: (items[i]["macro_f1"], -abs(i - 10), -i))


def check_diagnostic(report, reference):
    require((report.get("status"), report.get("exit_code"), report.get("mode"), report.get("partition")) ==
            ("completed", 0, "diagnose", "val"), "Completed validation diagnostic required")
    require(report.get("reference") == reference and report.get("test_graphs_loaded") == 0
            and report.get("test_evaluated") is False, "Diagnostic provenance/test boundary differs")
    require(set(report.get("models", {})) == {"E2", "E3"}
            and all(m.get("saved_validation_reproduced") is True and m.get("model_state_unchanged") is True
                    and m.get("samples") == 13446 for m in report["models"].values()), "Diagnostic validation not accepted")
