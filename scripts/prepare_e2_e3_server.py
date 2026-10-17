"""Server-only metadata checks for frozen E2/E3; never import torch or train.

Reads JSON/CSV, hashes source/checkpoint files and scans the sample manifest.
No .pt deserialization, graph-content checks, package installation, GPU query,
test inference, subprocess launch, or edits to existing experiment artifacts.
Only a new, exclusive preparation/report.json is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shlex
import sys


TRAIN_SOURCES = {
    "scripts/train_grouped.py", "scripts/grouped_graphs.py",
    "scripts/smoke_grouped_gpu.py", "src/xgnid/model.py",
    "src/xgnid/edge_experiments.py", "third_party/GNN4ID/Utility/Model.py",
}
SMOKE_SOURCES = TRAIN_SOURCES - {"scripts/train_grouped.py"} | {"scripts/audit_graph_split.py"}
LABELS = ["Benign", "WebBased", "Spoofing", "Recon", "Mirai", "Dos", "DDos", "BruteForce"]
SPLIT_ID = "a8ae9c934eec9fee970cc725e60051d0c709e0f16c7dda14749350ff2006975a"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(root, value):
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def check_config(config):
    expected_controls = {
        "seed": 42, "batch_size": 16, "lr": .001, "hidden_dim": 64,
        "optimizer": "Adam", "early_stop_patience": 12, "early_stop_min_delta": 0.,
        "auxiliary_losses": False, "calibration": False, "class_weighting": False,
        "hard_negatives": False, "loss": "plain NLL", "device_family": "cuda",
    }
    require(config["schema"] == "xgnid.e2-e3-preparation.v1", "Wrong preparation schema")
    require(config["split_id"] == SPLIT_ID and config["labels"] == LABELS, "Frozen split/labels changed")
    require(config["controls"] == expected_controls, "Frozen E0/E1 training controls changed")
    require((config["epochs"], config["workers"], config["device"]) == (100, 2, "cuda:0"), "Wrong training budget/device/workers")
    require(config["partition_sizes"] == {"train": 107621, "val": 13446, "test": 13448}, "Wrong frozen partition sizes")
    return {"controls": expected_controls, "epochs": 100, "workers": 2}


def runtime(config):
    require(os.name == "posix", "Execute this preparation on the Linux server")
    versions = {"python": platform.python_version(), "torch": metadata.version("torch"),
                "pyg": metadata.version("torch-geometric"),
                "numpy": metadata.version("numpy"), "scikit-learn": metadata.version("scikit-learn")}
    for name, expected in config["expected_runtime"].items():
        require(versions.get(name) == expected, f"Runtime version changed: {name}={versions.get(name)}, expected {expected}")
    return {"executable": sys.executable, "versions": versions,
            "verification": "distribution metadata only; CUDA availability and binary imports not tested",
            "historical_numpy_sklearn_versions": "not pinned by old training configs; current versions recorded only"}


def manifest(repo, config):
    directory = within(repo, config["paths"]["split_dir"])
    meta = read(directory / "split.json")
    require((meta["schema"], meta["status"], meta["split_id"]) ==
            ("xgnid.grouped-input.v1", "completed", SPLIT_ID), "Split metadata identity mismatch")
    path = within(directory, meta["sample_manifest"])
    require(path.parent == directory and sha(path) == meta["sample_manifest_sha256"], "Sample manifest path/SHA mismatch")
    identity = {"protocol": meta["protocol"], "samples_sha256": meta["sample_manifest_sha256"],
                "source_manifest_sha256": meta["source_manifest_sha256"]}
    require(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == SPLIT_ID, "Recomputed split ID mismatch")
    require(meta["protocol"]["label_mapping"] == dict(enumerate(LABELS)) or
            meta["protocol"]["label_mapping"] == {str(i): label for i, label in enumerate(LABELS)}, "Label mapping mismatch")
    for key in ("no_cross_partition_node_hash", "no_cross_partition_edge_hash", "no_repeated_edge_hash", "all_groups_assigned_once"):
        require(meta["checks"].get(key) is True, f"Split invariant not recorded: {key}")
    root = Path(meta["source_data_root"]).resolve()
    require(root.is_dir(), f"Source graph root missing: {root}; do not silently rewrite paths")
    rows = {part: [] for part in config["partition_sizes"]}
    groups, edges = {}, set()
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            part, label, edge, node = row["partition"], row["label"], row["edge_hash"], row["node_hash"]
            require(part in rows and label in range(8) and row["manifest_index"] == index, f"Invalid sample at {index}")
            require(row["sample_id"] == "edge_sha256:" + edge and row["group_id"] == "node_sha256:" + node, f"Invalid identity at {index}")
            require(edge not in edges and groups.get(node, (part, label)) == (part, label), "Repeated Edge input or cross-partition/conflicting group")
            location = Path(row["path"])
            require(not location.is_absolute() and ".." not in location.parts and row["container_offset"] >= 0, "Invalid graph location")
            groups[node] = (part, label)
            edges.add(edge)
            rows[part].append(row)
    sizes = {part: len(values) for part, values in rows.items()}
    counts = {part: dict(Counter(str(row["label"]) for row in values)) for part, values in rows.items()}
    require(sizes == meta["partition_sizes"] == config["partition_sizes"], "Manifest partition sizes mismatch")
    require(counts == meta["partition_label_counts"], "Manifest class counts mismatch")
    require(len(edges) == meta["selected_graph_count"] and len(groups) == meta["independent_input_groups"], "Manifest sample/group count mismatch")
    # A bounded existence check only; every graph is hash-checked by the frozen loader at training time.
    for label in range(8):
        candidates = [row for row in rows["train"] if row["label"] == label][:4]
        require(len(candidates) == 4, "Missing training initialization candidates")
        for row in candidates:
            require(within(root, row["path"]).is_file(), "Training candidate graph missing: " + row["path"])
    return meta, rows


def source_hashes(repo, config, required):
    values = config["source_hashes"]
    require(set(values) >= required, "Historical report omits required frozen source hashes")
    for name, expected in values.items():
        require(sha(within(repo, name)) == expected, f"Frozen source changed: {name}")
    return values


def smoke(repo, config, meta):
    path = within(repo, config["paths"]["smoke_report"])
    report, saved = read(path), read(path.parent / "config.json")
    require((report["status"], report["exit_code"], report["split_id"]) == ("completed", 0, SPLIT_ID), "GPU smoke not completed on frozen split")
    require(report["device"].split(":")[0] == "cuda" and saved["device"].split(":")[0] == "cuda", "Not a CUDA smoke")
    require(saved["split_id"] == SPLIT_ID and saved["sample_manifest_sha256"] == meta["sample_manifest_sha256"], "Smoke manifest changed")
    require(saved["seed"] == 42 and saved["batch_size"] == 16, "Smoke initialization seed/batch changed")
    require(saved["torch"] == config["expected_runtime"]["torch"] and saved["pyg"] == config["expected_runtime"]["pyg"], "Historical smoke runtime differs")
    for pair in ("E0/E1", "E2/E3"):
        require(report["paired_initial_states_identical"].get(pair) is True, f"Unverified paired initialization: {pair}")
        left, right = pair.split("/")
        require(report["models"][left]["initial_state_sha256"] == report["models"][right]["initial_state_sha256"], f"Paired hashes differ: {pair}")
    for name in ("E0", "E1", "E2", "E3"):
        result = report["models"][name]
        require(result.get("parameter_update_verified") is True and result.get("input_graph_unchanged") is True, f"Smoke checks missing for {name}")
        if name in ("E2", "E3"):
            require(result["parameter_count"] == 846800, f"Unexpected {name} parameter count")
    require(report["models"]["E2"]["gate_gradient_max"] == 0, "E2 equal fusion unexpectedly trained the gate")
    require(report["models"]["E3"]["gate_gradient_max"] > 0, "E3 adaptive gate gradient not verified")
    source_hashes(repo, saved, SMOKE_SOURCES)
    return report


def training_artifacts(repo, config, meta, rows, smoke_report, key, names, mode):
    path = within(repo, config["paths"][key])
    report = read(path)
    require((report["status"], report["exit_code"], report["mode"], report["split_id"]) ==
            ("completed", 0, mode, SPLIT_ID), f"Wrong status/identity: {key}")
    require(report["controls"] == config["controls"] and report["epochs"] == (2 if mode == "pilot" else 100), f"Frozen budget/control mismatch: {key}")
    require(read(path.parent / "status.json") == {"status": "completed", "exit_code": 0}, f"Root status mismatch: {key}")
    result = {"report_sha256": sha(path), "models": {}}
    for name in names:
        saved = report["models"][name]
        directory = path.parent / name
        cfg = read(directory / "config.json")
        require(sha(directory / "config.json") == saved["config_sha256"], f"{key}/{name} config SHA mismatch")
        require(read(directory / "metrics.json") == saved, f"{key}/{name} metrics differ from summary")
        require(read(directory / "status.json") == {"status": "completed", "exit_code": 0}, f"{key}/{name} model status mismatch")
        require(cfg["controls"] == config["controls"] and cfg["epochs"] == report["epochs"], f"{key}/{name} config controls differ")
        require(cfg["experiment"] == name and cfg["mode"] == mode and cfg["split_id"] == SPLIT_ID, f"{key}/{name} model identity differs")
        require(cfg["torch"] == config["expected_runtime"]["torch"] and cfg["pyg"] == config["expected_runtime"]["pyg"], f"{key}/{name} historical runtime differs")
        require(cfg["sample_manifest_sha256"] == meta["sample_manifest_sha256"] and
                cfg["source_manifest_sha256"] == meta["source_manifest_sha256"], f"{key}/{name} data identity differs")
        initial = smoke_report["models"][name]["initial_state_sha256"]
        require(cfg["initial_state_sha256"] == saved["initial_state_sha256"] == initial, f"{key}/{name} initialization differs")
        source_hashes(repo, cfg, TRAIN_SOURCES)
        files = {}
        for filename in ("best.pt", "last.pt", "epochs.jsonl"):
            artifact = directory / filename
            require(artifact.is_file() and artifact.stat().st_size > 0, f"Missing artifact: {artifact}")
            files[filename] = {"bytes": artifact.stat().st_size, "sha256_now": sha(artifact)}
        with (directory / "epochs.jsonl").open(encoding="utf-8") as stream:
            epochs = [json.loads(line) for line in stream]
        require([entry["epoch"] for entry in epochs] == list(range(1, saved["stopped_epoch"] + 1)), f"{name} missing/disordered epoch records")
        require(0 < saved["stopped_epoch"] <= report["epochs"], f"{name} invalid stopped epoch")
        require(all(entry["train_samples"] == cfg["selected_partition_sizes"]["train"] for entry in epochs), f"{name} per-epoch training count mismatch")
        selected = max(epochs, key=lambda entry: entry["val_macro_f1"])
        require(selected["epoch"] == saved["best_epoch"] and selected["val_macro_f1"] == saved["best_validation"]["macro_f1"], f"{name} validation selection mismatch")
        if mode == "full":
            require(cfg["selected_partition_sizes"] == config["partition_sizes"], f"{name} selected sizes differ")
            for part in ("train", "val"):
                digest = hashlib.sha256(json.dumps([r["sample_id"] for r in rows[part]]).encode()).hexdigest()
                require(cfg[part + "_sample_ids_sha256"] == digest, f"{name} {part} sample ordering differs")
            require(saved["test"]["samples"] == len(rows["test"]), f"{name} test count mismatch")
            prediction_path = directory / "predictions.csv"
            with prediction_path.open(newline="", encoding="utf-8") as stream:
                predictions = csv.DictReader(stream)
                require(set(predictions.fieldnames or []) == {"sample_id", "label", "prediction", "flow_weight", "packet_weight"}, "Unexpected prediction CSV schema")
                count = 0
                for count, prediction in enumerate(predictions, 1):
                    require(count <= len(rows["test"]), "Extra prediction rows")
                    row = rows["test"][count - 1]
                    require(prediction["sample_id"] == row["sample_id"] and int(prediction["label"]) == row["label"]
                            and int(prediction["prediction"]) in range(8), f"{name} prediction identity/order mismatch at {count}")
                require(count == len(rows["test"]), "Missing prediction rows")
            files["predictions.csv"] = {"bytes": prediction_path.stat().st_size, "sha256_now": sha(prediction_path)}
        else:
            require(saved["test"] is None, "Pilot unexpectedly includes test metrics")
        result["models"][name] = {"best_epoch": saved["best_epoch"], "stopped_epoch": saved["stopped_epoch"], "files": files}
    return result


def command(repo, config, name):
    paths, controls = config["paths"], config["controls"]
    args = [sys.executable, "-u", str(repo / "scripts/train_grouped.py"), "--mode", "full",
            "--split-dir", str(within(repo, paths["split_dir"])), "--expected-split-id", SPLIT_ID,
            "--smoke-report", str(within(repo, paths["smoke_report"])),
            "--pilot-report", str(within(repo, paths["pilot_report"])),
            "--baseline-report", str(within(repo, paths["baseline_report"])),
            "--device", config["device"], "--seed", str(controls["seed"]),
            "--batch-size", str(controls["batch_size"]), "--lr", str(controls["lr"]),
            "--epochs", str(config["epochs"]), "--patience", str(controls["early_stop_patience"]),
            "--workers", str(config["workers"]), "--experiments", name,
            "--output", str(within(repo, paths[name.lower() + "_output"]))]
    if name == "E3":
        args += ["--fusion-control-report", str(within(repo, paths["e2_output"]) / "summary.json")]
    return {"argv": args, "foreground_command_preview": shlex.join(args),
            "execute_now": False, "note": "Preview only; launch after dependency review and GPU availability confirmation"}


def r0_snapshot(repo):
    pointer = repo / "outputs/r0_full_launch.json"
    if not pointer.is_file():
        return {"observed": "no launch pointer; does not prove no active R0"}
    value = read(pointer)
    directory = within(repo, value["output"])
    status = directory / "status.json"
    return {"output": str(directory), "status_at_read": read(status) if status.is_file() else None,
            "note": "Informational snapshot only; no process/GPU query, no start/stop action"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New report directory under repository outputs only")
    args = parser.parse_args()
    repo = args.repo.resolve()
    target = args.output or repo / "outputs" / ("e2_e3_preflight_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output = within(repo / "outputs", target.resolve())
    require(output != repo / "outputs", "Use a new preparation subdirectory")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "xgnid.e2-e3-preparation-report.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "repo": str(repo), "tool_sha256": sha(Path(__file__)), "checks": [], "training_started": False, "gpu_touched": False,
              "graph_tensors_loaded": False, "test_inference_performed": False,
              "limitations": ["Historical checkpoint SHA references absent: recorded hashes establish an inventory now, not past immutability",
                              "Graph contents/dimensions are not retested; frozen training loader verifies each graph before use",
                              "Binary import compatibility, live GPU availability and other users' GPU jobs not checked",
                              "E3 remains pending completed E2; no training command is executed by this tool"]}

    def check(name, action):
        try:
            result = action()
            report["checks"].append({"name": name, "status": "passed", "detail": result})
            return result
        except Exception as exc:
            report["checks"].append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            return None

    def load_config():
        value = read(args.config)
        require(isinstance(value, dict), "Configuration must be a JSON object")
        return value

    config = check("configuration_read", load_config)
    if config is not None and check("frozen_configuration", lambda: check_config(config)) is not None:
        report["configuration"] = config
        report["configuration_sha256"] = sha(args.config)
        check("server_runtime_metadata", lambda: runtime(config))
        dataset = check("frozen_manifest", lambda: manifest(repo, config))
        # Dataset rows are needed only internally; do not duplicate the large manifest in the report.
        if dataset is not None:
            meta, rows = dataset
            report["checks"][-1]["detail"] = {"split_id": meta["split_id"], "sample_manifest_sha256": meta["sample_manifest_sha256"],
                                                    "partition_sizes": meta["partition_sizes"], "partition_label_counts": meta["partition_label_counts"]}
            smoke_report = check("historical_gpu_smoke", lambda: smoke(repo, config, meta))
            if smoke_report is not None:
                report["checks"][-1]["detail"] = {"report_sha256": sha(within(repo, config["paths"]["smoke_report"])),
                    "paired_initial_states_identical": smoke_report["paired_initial_states_identical"],
                    "E2_E3_initial_state_sha256": smoke_report["models"]["E2"]["initial_state_sha256"]}
                for key, names, mode in (("pilot_report", ("E0", "E1"), "pilot"),
                                         ("baseline_report", ("E0",), "full"), ("e1_report", ("E1",), "full")):
                    check(key, lambda key=key, names=names, mode=mode: training_artifacts(repo, config, meta, rows, smoke_report, key, names, mode))
        def unused_outputs():
            paths = [within(repo, config["paths"][name]) for name in ("e2_output", "e3_output")]
            for path in paths:
                require(not path.exists() and not path.with_suffix(".log").exists(), f"Output or log already exists: {path}; inspect instead of overwriting")
            return [str(path) for path in paths]
        check("unused_training_outputs", unused_outputs)
        check("r0_informational_snapshot", lambda: r0_snapshot(repo))
    failed = [item["name"] for item in report["checks"] if item["status"] == "failed"]
    report["status"] = "blocked" if failed else "metadata_checks_passed"
    report["exit_code"] = 1 if failed else 0
    report["failed_checks"] = failed
    report["gpu_launch_readiness"] = "not_checked"
    if not failed:
        report["future_commands"] = {name: command(repo, config, name) for name in ("E2", "E3")}
        report["next_gates"] = ["R0 completed with saved results and no training process remains",
                                "Confirm live GPU availability; metadata checks do not certify it",
                                "Recheck dependencies at launch; E2 must complete successfully before E3"]
    path = output / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact_checks = [{k: item[k] for k in ("name", "status", "error") if k in item} for item in report["checks"]]
    print(json.dumps({"status": report["status"], "checks": compact_checks, "failed_checks": failed, "report": str(path),
                      "training_started": False, "gpu_touched": False}, ensure_ascii=False, indent=2))
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
