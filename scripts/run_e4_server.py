"""Independent E4 server entry: check -> real CUDA paired smoke -> full.

Reuses frozen E3/E4 constructors and the original full training implementation.
Only E4 config/provenance annotations and the in-memory E4 smoke lookup are
extended. Never edits E0-E3, trains locally, installs dependencies, picks a GPU,
loads trained E3 weights, or chains smoke into full automatically.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys


SPLIT_ID = "a8ae9c934eec9fee970cc725e60051d0c709e0f16c7dda14749350ff2006975a"
CONTROLS = {"seed": 42, "batch_size": 16, "lr": .001, "hidden_dim": 64,
            "optimizer": "Adam", "early_stop_patience": 12, "early_stop_min_delta": 0.,
            "auxiliary_losses": False, "calibration": False, "class_weighting": False,
            "hard_negatives": False, "loss": "plain NLL", "device_family": "cuda"}
TRAIN_SOURCES = {"scripts/train_grouped.py", "scripts/grouped_graphs.py", "scripts/smoke_grouped_gpu.py",
                 "src/xgnid/model.py", "src/xgnid/edge_experiments.py", "third_party/GNN4ID/Utility/Model.py"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def path_in(repo, value):
    path = (repo / value).resolve()
    path.relative_to(repo)
    return path


def source_check(repo, hashes):
    for name, digest in hashes.items():
        require(sha(path_in(repo, name)) == digest, f"Frozen source changed: {name}")


def reference(repo, spec):
    require(spec["schema"] == "xgnid.e4-no-edge.v1" and spec["experiment"] == "E4", "Wrong E4 configuration")
    require(spec["controls"] == CONTROLS and spec["split_id"] == SPLIT_ID, "Frozen controls/split differ")
    require((spec["epochs"], spec["workers"], spec["device"], spec["parameter_count"]) == (100, 2, "cuda:0", 846800), "Wrong budget/device/model size")
    parent = path_in(repo, spec["parent_e3_run"])
    status, summary = read(parent / "status.json"), read(parent / "summary.json")
    require(status.get("status") == "completed" and status.get("exit_code") == 0, "E3 has not completed successfully")
    require((summary["status"], summary["exit_code"], summary["mode"], summary["split_id"]) ==
            ("completed", 0, "full", SPLIT_ID), "Wrong E3 summary identity")
    require(summary["controls"] == CONTROLS and summary["epochs"] == 100, "Parent E3 budget/controls differ")
    metrics = read(parent / "E3/metrics.json")
    require(summary["models"]["E3"] == metrics, "E3 metrics/summary differ")
    cfg_path = parent / "E3/config.json"
    cfg = read(cfg_path)
    require(sha(cfg_path) == metrics["config_sha256"], "E3 config SHA differs")
    require(cfg["experiment"] == "E3" and cfg["mode"] == "full" and cfg["controls"] == CONTROLS, "Wrong E3 model config")
    require(cfg["use_edge_attr"] is True and cfg["fusion"] == "adaptive_logits", "Parent is not the Edge adaptive E3")
    require(cfg["epochs"] == 100 and cfg["workers"] == 2 and cfg["split_id"] == SPLIT_ID, "Parent execution settings differ")
    require(set(cfg["source_hashes"]) >= TRAIN_SOURCES, "Parent omits required source hashes")
    source_check(repo, cfg["source_hashes"])
    split_dir = path_in(repo, spec["split_dir"])
    split = read(split_dir / "split.json")
    require(split["status"] == "completed" and split["split_id"] == SPLIT_ID, "Wrong frozen split")
    require(split["partition_sizes"] == cfg["selected_partition_sizes"] == {"train": 107621, "val": 13446, "test": 13448}, "Partition counts differ")
    manifest = path_in(split_dir, split["sample_manifest"])
    require(manifest.parent == split_dir and sha(manifest) == split["sample_manifest_sha256"] == cfg["sample_manifest_sha256"], "Sample manifest changed")
    require(split["source_manifest_sha256"] == cfg["source_manifest_sha256"], "Graph source manifest changed")
    require(Path(cfg["data_root"]).is_dir(), "Original graph directory missing")
    old_path = path_in(repo, spec["historical_smoke"])
    old, old_cfg = read(old_path), read(old_path.parent / "config.json")
    require((old["status"], old["exit_code"], old["split_id"]) == ("completed", 0, SPLIT_ID), "Historical GPU smoke identity differs")
    require(old["device"].split(":")[0] == "cuda" and old_cfg["seed"] == 42 and old_cfg["batch_size"] == 16, "Wrong historical smoke settings")
    source_check(repo, old_cfg["source_hashes"])
    initial = metrics["initial_state_sha256"]
    require(initial == cfg["initial_state_sha256"] == old["models"]["E3"]["initial_state_sha256"], "E3 initialization identity differs")
    require(metadata.version("torch") == cfg["torch"] and metadata.version("torch-geometric") == cfg["pyg"], "Use the same torch/PyG environment as E3")
    binding = {"schema": "xgnid.e4-binding.v1", "definition_id": identity(spec), "wrapper_sha256": sha(Path(__file__)),
               "parent_summary_sha256": sha(parent / "summary.json"), "parent_config_sha256": sha(cfg_path),
               "parent_initial_state_sha256": initial, "historical_smoke_sha256": sha(old_path),
               "historical_smoke_config_sha256": sha(old_path.parent / "config.json"),
               "split_id": SPLIT_ID, "sample_manifest_sha256": split["sample_manifest_sha256"],
               "source_hashes": cfg["source_hashes"], "controls": CONTROLS, "epochs": 100}
    return cfg, old, old_cfg, binding


def smoke_args(repo, spec, cfg):
    return argparse.Namespace(output=path_in(repo, spec["smoke_output"]), split_dir=path_in(repo, spec["split_dir"]),
        expected_split_id=SPLIT_ID, data_root=Path(cfg["data_root"]), device="cuda:0", seed=42,
        batch_size=16, candidates_per_class=4, experiments=["E3", "E4"])


def verify_smoke(repo, spec, cfg, old_cfg, binding):
    directory = path_in(repo, spec["smoke_output"])
    report, saved, receipt = read(directory / "smoke.json"), read(directory / "config.json"), read(directory / "e4_binding.json")
    require(read(directory / "status.json").get("status") == "completed", "E4 raw smoke not completed")
    require(receipt["status"] == "completed" and receipt["binding"] == binding, "E4 smoke bound to different code/config/reference")
    require(receipt["smoke_sha256"] == sha(directory / "smoke.json") and receipt["config_sha256"] == sha(directory / "config.json"), "E4 smoke artifacts changed")
    check_paired_report(report, saved, cfg, old_cfg, binding)
    return report, saved, receipt


def check_paired_report(report, saved, cfg, old_cfg, binding):
    require((report["status"], report["exit_code"], report["split_id"]) == ("completed", 0, SPLIT_ID), "Paired smoke failed")
    require(report["device"].split(":")[0] == "cuda" and saved["device"].split(":")[0] == "cuda", "CUDA smoke required; no CPU substitution")
    require(saved["torch"] == cfg["torch"] and saved["pyg"] == cfg["pyg"] and saved["device_name"] == cfg["device_name"], "Smoke runtime/hardware differs from E3")
    require(saved["seed"] == 42 and saved["batch_size"] == 16 and saved["samples"] == old_cfg["samples"], "Paired smoke initialization samples/settings differ")
    require(saved["source_hashes"] == old_cfg["source_hashes"] and saved["sample_manifest_sha256"] == cfg["sample_manifest_sha256"], "Paired smoke input/code provenance differs")
    for name in ("E3", "E4"):
        result = report["models"][name]
        require(result["initial_state_sha256"] == binding["parent_initial_state_sha256"], f"{name} initialization not paired with completed E3")
        require(result["parameter_count"] == 846800 and result["parameter_update_verified"] is True and result["input_graph_unchanged"] is True,
                f"{name} model size/update/input check failed")
        require(result["gate_gradient_max"] > 0, f"{name} adaptive gate has no gradient")
    e4 = report["models"]["E4"]
    require(len(e4["actual_GAT_inputs"]) == 12 and all(shape is None for shape in e4["actual_GAT_inputs"].values()), "E4 attributes reached a GAT layer")
    require(all(value is None or value == 0 for value in e4["edge_projection_gradient_max"].values()), "E4 edge projection unexpectedly trains")
    require(max(e4["attribute_sensitivity"].values()) <= e4["comparison_tolerance"], "E4 output depends on disabled attributes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["check", "smoke", "full"], default="check")
    cli = parser.parse_args()
    require(os.name == "posix", "Execute on the Linux server only")
    repo, spec = cli.repo.resolve(), read(cli.config)
    cfg, old, old_cfg, binding = reference(repo, spec)
    if cli.mode == "check":
        print(json.dumps({"status": "metadata_checks_passed", "experiment": "E4", "parent": spec["parent_e3_run"],
            "controls": CONTROLS, "epochs": 100, "workers": 2, "use_edge_attr": False,
            "fresh_initialization_sha256_expected": binding["parent_initial_state_sha256"],
            "training_started": False, "gpu_touched": False, "server_gpu_smoke": "validated only by smoke/full modes"}, ensure_ascii=False, indent=2))
        return
    sys.path.insert(0, str(repo / "scripts"))
    if cli.mode == "smoke":
        import smoke_grouped_gpu as smoke
        args = smoke_args(repo, spec, cfg)
        report = smoke.run(args)
        try:
            saved = read(args.output / "config.json")
            check_paired_report(report, saved, cfg, old_cfg, binding)
            receipt = {"status": "completed", "exit_code": 0, "binding": binding,
                "smoke_sha256": sha(args.output / "smoke.json"), "config_sha256": sha(args.output / "config.json"),
                "paired_E3_E4_initialization": True, "E4_all_GAT_attributes_disabled": True,
                "E4_attribute_intervention_invariant": True, "E4_gate_gradient_verified": True,
                "runtime": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "device_name": saved["device_name"], "torch": saved["torch"], "pyg": saved["pyg"]},
                "test_inference_performed": False, "formal_training_started": False,
                "created_utc": datetime.now(timezone.utc).isoformat()}
            smoke.write(args.output / "e4_binding.json", receipt)
            print(json.dumps({"status": "e4_cuda_smoke_passed", "binding_report": str(args.output / "e4_binding.json"),
                              "paired_initial_state_sha256": binding["parent_initial_state_sha256"],
                              "formal_training_started": False}, ensure_ascii=False, indent=2), flush=True)
        except Exception as exc:
            smoke.write(args.output / "e4_binding.json", {"status": "failed", "exit_code": 1, "error": str(exc)})
            raise
        return
    fresh, fresh_cfg, receipt = verify_smoke(repo, spec, cfg, old_cfg, binding)
    import train_grouped as g
    require(g.torch.cuda.is_available(), "CUDA unavailable; no CPU fallback")
    require(g.torch.cuda.get_device_name(0) == fresh_cfg["device_name"], "Full run GPU model differs from passed E4 smoke")
    args = argparse.Namespace(mode="full", experiments=["E4"], output=path_in(repo, spec["full_output"]),
        split_dir=path_in(repo, spec["split_dir"]), expected_split_id=SPLIT_ID, data_root=Path(cfg["data_root"]),
        smoke_report=path_in(repo, spec["historical_smoke"]), pilot_report=path_in(repo, spec["pilot_report"]),
        baseline_report=path_in(repo, spec["baseline_report"]), fusion_control_report=None,
        device="cuda:0", seed=42, batch_size=16, lr=.001, epochs=100, patience=12, workers=2,
        pilot_train_per_class=32, pilot_val_per_class=16)
    evidence = {"definition": spec, "binding": binding, "e4_smoke_sha256": receipt["smoke_sha256"],
                "e4_smoke_binding_sha256": sha(path_in(repo, spec["smoke_output"]) / "e4_binding.json"),
                "smoke_runtime": receipt["runtime"], "full_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "scope": "Disable GAT edge_attr only; adaptive gate/topology/node features unchanged; fresh initialization"}
    original_readiness, original_write = g.readiness, g.write

    def e4_readiness(current_args, frozen, controls):
        verified, verified_cfg = original_readiness(current_args, frozen, controls)
        require(controls == binding["controls"], "Full E4 controls differ from E3/E4 smoke")
        require(frozen.metadata["sample_manifest_sha256"] == binding["sample_manifest_sha256"], "Full E4 manifest differs")
        # Extend the returned lookup only; historical smoke files stay unchanged.
        result = copy.deepcopy(verified)
        result["models"]["E4"] = fresh["models"]["E4"]
        return result, verified_cfg

    def e4_write(path, value):
        # The frozen writer's E0-E3-only metadata expressions do not describe E4.
        # Correct annotations only; leave model, loop, loss and selection intact.
        if path.name == "config.json" and value.get("experiment") == "E4":
            for field in ("train_sample_ids_sha256", "val_sample_ids_sha256", "selected_partition_sizes",
                          "sample_manifest_sha256", "source_manifest_sha256"):
                require(value[field] == cfg[field], f"E4 sample set/order differs from E3: {field}")
            value["use_edge_attr"] = False
            value["fusion"] = "adaptive_logits"
            value["edge_attribute_ablation"] = evidence
        if path.name == "metrics.json" or (path.name == "summary.json" and "E4" in value.get("models", {})):
            value["edge_attribute_ablation"] = evidence
        return original_write(path, value)

    g.readiness, g.write = e4_readiness, e4_write
    g.run(args)


if __name__ == "__main__":
    main()
