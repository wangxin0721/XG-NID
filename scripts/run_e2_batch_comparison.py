"""Independent, server-only E2 training-batch comparison (256 / 512 / 1024).

The frozen trainer/model files stay unchanged. Reuse its verified loader,
initialization, sampler, metrics and checkpoints; hold initialization/evaluation
batch at 16. Probe uses only real training graphs. Full requires a matching
successful probe on the chosen GPU and never auto-retries or resumes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


SOURCES = (
    "scripts/train_grouped.py", "scripts/grouped_graphs.py", "scripts/smoke_grouped_gpu.py",
    "scripts/audit_graph_split.py", "scripts/build_grouped_split.py", "src/xgnid/__init__.py",
    "src/xgnid/model.py", "src/xgnid/edge_experiments.py", "third_party/GNN4ID/Utility/Model.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(repo, name):
    path = (repo / name).resolve()
    path.relative_to(repo)
    return path


def gpu_query(*args):
    result = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True)
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def compute_processes(uuid):
    rows = gpu_query("--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits")
    return [{"pid": int(row[1]), "name": row[2].strip()} for row in rows
            if len(row) == 3 and row[0].strip() == uuid and int(row[1]) != os.getpid()]


def other_processes_snapshot(gpu):
    try:
        return {"processes": compute_processes(gpu["uuid"]), "scope": "point-in-time observation only"}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"query_error": str(exc), "scope": "unknown; do not infer timing isolation"}


def checked_gpu(index, allow_shared=False):
    rows = gpu_query("-i", str(index), "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits")
    require(len(rows) == 1 and len(rows[0]) == 6, "Cannot identify the selected physical GPU")
    number, uuid, name, memory, used, free = [value.strip() for value in rows[0]]
    require(int(number) == index and uuid.startswith("GPU-"), "GPU identity mismatch")
    active = compute_processes(uuid)
    require(allow_shared or not active, f"Physical GPU {index} has compute processes {active}; use --allow-shared-gpu only for an intentional shared run")
    return {"physical_index": index, "uuid": uuid, "name": name, "memory_mib": int(memory),
            "used_mib_at_query": int(used), "free_mib_at_query": int(free),
            "sharing_allowed": allow_shared, "existing_compute_processes": active,
            "availability": "explicit sharing allowed" if allow_shared else "no compute process at query time; not a reservation",
            "timing_scope": "shared execution; not a standalone speed benchmark" if allow_shared else "isolation requires observation; query does not reserve GPU"}


def check_config(config, batch_size):
    require(config["schema"] == "xgnid.e2-batch-comparison.v1" and config["experiment"] == "E2", "Wrong comparison protocol")
    registered = config["train_batch_sizes"]
    require(registered in ([256, 512], [1024]) and batch_size in registered, "Batch must match its separate comparison registration")
    if registered == [1024]:
        registration = config.get("registration", {})
        require(registration.get("kind") == "exploratory_extension_after_prior_results"
                and registration.get("prior_test_results_available") is True,
                "B1024 requires its explicit exploratory-extension config; do not edit the original B256/B512 registration")
    require((config["initialization_batch_size"], config["evaluation_batch_size"], config["epochs"], config["workers"]) ==
            (16, 16, 100, 2), "Fixed initialization/evaluation batch or budget changed")
    require(config["probe"] == {"warmup_steps": 1, "measured_steps": 3}, "Probe budget changed")
    expected = {"seed": 42, "batch_size": 16, "lr": .001, "hidden_dim": 64,
                "optimizer": "Adam", "early_stop_patience": 12, "early_stop_min_delta": 0.,
                "auxiliary_losses": False, "calibration": False, "class_weighting": False,
                "hard_negatives": False, "loss": "plain NLL", "device_family": "cuda"}
    require(config["reference_controls"] == expected, "Reference controls changed; no automatic lr scaling")
    require(config["split_id"] == "a8ae9c934eec9fee970cc725e60051d0c709e0f16c7dda14749350ff2006975a", "Wrong frozen split")


def setup(core, args, config, gpu):
    torch = core.torch
    require(torch.cuda.is_available(), "CUDA unavailable; no CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(platform.python_version() == "3.9.23" and torch.__version__ == "2.3.1+cu118"
            and core.torch_geometric.__version__ == "2.5.3", "Server training runtime changed")
    frozen = core.FrozenGraphs(local_path(args.repo, config["split_directory"]), config["split_id"])
    require(frozen.metadata["partition_sizes"] == {"train": 107621, "val": 13446, "test": 13448}, "Frozen partition sizes changed")
    reference_args = SimpleNamespace(mode="full", device="cuda:0", epochs=100, experiments=["E2"],
        smoke_report=local_path(args.repo, config["smoke_report"]),
        pilot_report=local_path(args.repo, config["pilot_report"]),
        baseline_report=local_path(args.repo, config["baseline_report"]))
    smoke, smoke_config = core.readiness(reference_args, frozen, config["reference_controls"])
    reference_path = local_path(args.repo, config["reference_e2"])
    reference = read(reference_path)
    require(read(reference_path.parent / "status.json") == {"status": "completed", "exit_code": 0}, "Reference E2 has not completed")
    require((reference["status"], reference["exit_code"], reference["mode"], reference["split_id"]) ==
            ("completed", 0, "full", config["split_id"]), "Reference E2 identity/status mismatch")
    require(reference["controls"] == config["reference_controls"] and reference["epochs"] == 100, "Reference E2 budget mismatch")
    core.verify_training_artifact(reference_path, reference, "E2")
    reference_cfg = read(reference_path.parent / "E2/config.json")
    expected = reference["models"]["E2"]
    require(reference_cfg["controls"] == config["reference_controls"] and reference_cfg["experiment"] == "E2", "Reference E2 config differs")
    require(reference_cfg["sample_manifest_sha256"] == frozen.metadata["sample_manifest_sha256"], "Reference E2 data differs")
    require(expected["parameter_count"] == 846800 and expected["test"]["samples"] == 13448, "Reference E2 result incomplete")
    require(expected["initial_state_sha256"] == smoke["models"]["E2"]["initial_state_sha256"], "Reference E2 and verified smoke initialization differ")
    binding = {"config_sha256": sha(args.config), "entry_sha256": sha(Path(__file__)),
               "sources": {name: sha(local_path(args.repo, name)) for name in SOURCES},
               "reference_e2_sha256": sha(reference_path), "reference_config_sha256": sha(reference_path.parent / "E2/config.json"),
               "smoke_sha256": sha(reference_args.smoke_report), "pilot_sha256": sha(reference_args.pilot_report),
               "baseline_sha256": sha(reference_args.baseline_report),
               "split_id": config["split_id"], "samples_sha256": frozen.metadata["sample_manifest_sha256"],
               "train_batch_size": args.batch_size, "initial_state_sha256": expected["initial_state_sha256"],
               "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                           "pyg": core.torch_geometric.__version__, "cuda": torch.version.cuda},
               "gpu_uuid": gpu["uuid"], "allow_shared_gpu": args.allow_shared_gpu}
    return frozen, device, binding


def initial_model(core, frozen, device, binding):
    core.seed_all(42)
    warm = [frozen.load(row) for row in frozen.training_candidates()]
    multi = [i for i, graph in enumerate(warm) if graph["packet"].x.size(0) >= 3]
    require(len(multi) >= 2, "Missing frozen initialization candidates")
    order = multi[:2] + [i for i in range(len(warm)) if i not in multi[:2]]
    batch = core.Batch.from_data_list([warm[i] for i in order[:16]]).to(device)
    model = core.build_experiment("E2", warm[0]).to(device)
    core.initialize(model, "E2", batch)
    require(core.state_hash(model.state_dict().items()) == binding["initial_state_sha256"], "Initial state differs from completed E2; comparison rejected")
    require(sum(p.numel() for p in model.parameters()) == 846800 and model.use_gate is False, "E2 structure changed")
    return model


def update(core, model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = core.F.nll_loss(model(batch), batch.y.view(-1))
    require(bool(core.torch.isfinite(loss)), "Nonfinite training loss")
    loss.backward()
    core.torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def probe(core, args, config, frozen, device, binding, gpu):
    model = initial_model(core, frozen, device, binding)
    optimizer = core.torch.optim.Adam(model.parameters(), lr=.001)
    parameters_before = core.state_hash(model.named_parameters())
    rows = core.select_rows(frozen, "train")
    batches = list(islice(core.TrainBatches(len(rows), args.batch_size, 42), 3))
    loader = core.DataLoader(core.GraphRows(frozen, rows), batch_sampler=batches, num_workers=0)
    measured = []
    for index, batch in enumerate(loader):
        require(batch.num_graphs == args.batch_size, "Probe did not use a full requested batch")
        batch = batch.to(device)
        if index == 0:
            update(core, model, optimizer, batch)  # One disposable warmup update, no checkpoint.
        core.sync(device)
        core.torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        loss = update(core, model, optimizer, batch)
        core.sync(device)
        elapsed = time.perf_counter() - tick
        require(bool((model._last_cache["gate_weights"] == .5).all()), "E2 equal weights changed")
        item = {"batch": index + 1, "graphs": batch.num_graphs, "loss": loss,
                "resident_step_seconds_excluding_graph_io_and_transfer": elapsed,
                "nodes": {name: batch[name].x.size(0) for name in ("flow", "packet")},
                "edges": {str(name): batch[name].edge_index.size(1) for name in batch.edge_types},
                "peak_cuda_allocated_bytes": core.torch.cuda.max_memory_allocated(device),
                "peak_cuda_reserved_bytes": core.torch.cuda.max_memory_reserved(device),
                "other_compute_processes_snapshot": other_processes_snapshot(gpu)}
        measured.append(item)
        print("E2 batch probe", json.dumps(item), flush=True)
    require(len(measured) == 3, "Probe incomplete")
    require(core.state_hash(model.named_parameters()) != parameters_before, "Probe optimizer did not update parameters")
    return {"status": "completed", "exit_code": 0, "mode": "probe", "binding": binding,
            "hardware": gpu, "warmup_updates": 1, "measured_updates": 3, "test": None,
            "optimizer_update_verified": True,
            "batches": measured, "scope": "First three seeded training batches only; no worst-case capacity guarantee or epoch-throughput claim"}


def full(core, args, config, frozen, device, binding, gpu):
    probe_path = args.probe_report.resolve()
    probe_report = read(probe_path)
    require((probe_report["status"], probe_report["exit_code"], probe_report["mode"]) == ("completed", 0, "probe"), "Successful server probe required")
    probe_status = read(probe_path.parent / "status.json")
    require((probe_status.get("status"), probe_status.get("exit_code"), probe_status.get("mode"), probe_status.get("batch_size")) ==
            ("completed", 0, "probe", args.batch_size), "Probe did not finish normally")
    require(probe_report["binding"] == binding and probe_report["test"] is None
            and probe_report.get("optimizer_update_verified") is True
            and probe_report["measured_updates"] == 3 and probe_report["warmup_updates"] == 1
            and len(probe_report["batches"]) == 3, "Probe identity/runtime/GPU or budget differs; do not reuse another batch's probe")
    require(all(item["graphs"] == args.batch_size for item in probe_report["batches"]), "Probe batch size differs")
    directory = args.output / "E2"
    directory.mkdir()
    core.write(directory / "status.json", {"status": "running"})
    model = initial_model(core, frozen, device, binding)
    optimizer = core.torch.optim.Adam(model.parameters(), lr=.001)
    rows = {part: core.select_rows(frozen, part) for part in ("train", "val", "test")}
    sampler = core.TrainBatches(len(rows["train"]), args.batch_size, 42)
    kwargs = {"num_workers": 2, "pin_memory": True, "persistent_workers": True}
    train_loader = core.DataLoader(core.GraphRows(frozen, rows["train"]), batch_sampler=sampler, **kwargs)
    val_loader = core.DataLoader(core.GraphRows(frozen, rows["val"]), batch_size=16, shuffle=False, **kwargs)
    controls = dict(config["reference_controls"], batch_size=args.batch_size)
    run_config = {"experiment": "E2", "mode": "full", "protocol": config["schema"], "controls": controls,
                  "epochs": 100, "workers": 2, "initialization_batch_size": 16, "evaluation_batch_size": 16,
                  "binding": binding, "hardware": gpu, "probe_report": str(probe_path), "probe_sha256": sha(probe_path),
                  "comparison": config["comparison"], "partition_sizes": frozen.metadata["partition_sizes"],
                  "optimizer_updates_per_full_epoch": len(sampler)}
    if "registration" in config:
        run_config["registration"] = config["registration"]
    core.write(directory / "config.json", run_config)
    best, best_epoch, stale, updates, history = -1., 0, 0, 0, []
    started = time.perf_counter()
    with (directory / "epochs.jsonl").open("x", encoding="utf-8") as stream:
        for epoch in range(1, 101):
            core.sync(device)
            tick = time.perf_counter()
            core.torch.cuda.reset_peak_memory_stats(device)
            loss_sum, count, epoch_updates = 0., 0, 0
            for batch in train_loader:
                batch = batch.to(device)
                loss_sum += update(core, model, optimizer, batch) * batch.num_graphs
                count += batch.num_graphs
                epoch_updates += 1
            core.sync(device)
            train_seconds = time.perf_counter() - tick
            require(count == len(rows["train"]), "Incomplete training epoch")
            updates += epoch_updates
            validation, _ = core.evaluate(model, val_loader, device)
            entry = {"epoch": epoch, "train_loss": loss_sum / count, "train_samples": count,
                     "train_seconds": train_seconds, "train_order_sha256": sampler.last_order_sha256,
                     "optimizer_updates": epoch_updates, "cumulative_optimizer_updates": updates,
                     "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"],
                     "val_weighted_f1": validation["weighted_f1"], "val_wall_seconds": validation["wall_seconds_including_graph_io"],
                     "peak_cuda_allocated_bytes": core.torch.cuda.max_memory_allocated(device),
                     "peak_cuda_reserved_bytes": core.torch.cuda.max_memory_reserved(device),
                     "other_compute_processes_snapshot": other_processes_snapshot(gpu)}
            history.append(entry)
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            improved = validation["macro_f1"] > best
            if improved:
                best, best_epoch, stale = validation["macro_f1"], epoch, 0
            else:
                stale += 1
            payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                       "config": run_config, "epoch": epoch, "best_epoch": best_epoch,
                       "best_val_macro_f1": best, "validation": validation,
                       "sampler_rng_state": sampler.generator.get_state(), "torch_rng_state": core.torch.get_rng_state(),
                       "cuda_rng_states": core.torch.cuda.get_rng_state_all(), "optimizer_updates": updates}
            core.checkpoint(directory / "last.pt", payload)
            if improved:
                core.checkpoint(directory / "best.pt", payload)
            print(f"E2 batch={args.batch_size}", json.dumps(entry), flush=True)
            if stale >= 12:
                break
    selected = core.torch.load(directory / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state"])
    training_wall = time.perf_counter() - started
    test_loader = core.DataLoader(core.GraphRows(frozen, rows["test"]), batch_size=16, shuffle=False, **kwargs)
    test, predictions = core.evaluate(model, test_loader, device, export_rows=True)
    with (directory / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "label", "prediction", "flow_weight", "packet_weight"])
        writer.writeheader()
        writer.writerows(predictions)
    metrics = {"mode": "full", "protocol": config["schema"], "best_epoch": best_epoch,
               "stopped_epoch": history[-1]["epoch"], "best_validation": selected["validation"], "test": test,
               "parameter_count": 846800, "initial_state_sha256": binding["initial_state_sha256"],
               "train_batch_size": args.batch_size, "evaluation_batch_size": 16,
               "total_optimizer_updates": updates, "training_samples_seen": len(rows["train"]) * len(history),
               "training_validation_checkpoint_wall_seconds": training_wall, "run_wall_seconds": time.perf_counter() - started,
               "config_sha256": sha(directory / "config.json"),
               "checkpoint_sha256": {name: sha(directory / name) for name in ("best.pt", "last.pt")}}
    if "registration" in config:
        metrics["registration"] = config["registration"]
    core.write(directory / "metrics.json", metrics)
    core.write(directory / "status.json", {"status": "completed", "exit_code": 0})
    return {"status": "completed", "exit_code": 0, "mode": "full", "protocol": config["schema"],
            "binding": binding, "hardware": gpu, "models": {"E2": metrics}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("probe", "full"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(256, 512, 1024), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--allow-shared-gpu", action="store_true", help="Explicitly allow other compute processes; record shared timing, preserve existing jobs")
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.name == "posix", "Execute on the Linux training server only")
    require(args.mode != "full" or args.probe_report is not None, "Full requires --probe-report")
    args.repo = args.repo.resolve()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    args.output.relative_to(args.repo / "outputs")
    require(args.output.name.startswith("e2_batch_"), "Use an independent outputs/e2_batch_* directory")
    require(not args.output.exists(), "Output already exists; no overwrite, automatic retry or resume")
    config = read(args.config)
    check_config(config, args.batch_size)
    gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
    print("GPU execution policy", json.dumps(gpu), flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output / "status.json"

    def write_status(value):
        status_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    write_status({"status": "running", "mode": args.mode, "batch_size": args.batch_size, "pid": os.getpid()})
    try:
        sys.path.insert(0, str(args.repo / "scripts"))
        import train_grouped as core
        frozen, device, binding = setup(core, args, config, gpu)
        started = time.perf_counter()
        result = probe(core, args, config, frozen, device, binding, gpu) if args.mode == "probe" else full(core, args, config, frozen, device, binding, gpu)
        result["wall_seconds_after_setup"] = time.perf_counter() - started
        if "registration" in config:
            result["registration"] = config["registration"]
        core.write(args.output / "summary.json", result)
        write_status({"status": "completed", "exit_code": 0, "mode": args.mode, "batch_size": args.batch_size})
        print(json.dumps({"status": "completed", "mode": args.mode, "batch_size": args.batch_size,
                          "output": str(args.output), "physical_gpu": gpu["physical_index"],
                          "test_evaluated": args.mode == "full"}), flush=True)
    except BaseException as exc:
        rc = 130 if isinstance(exc, KeyboardInterrupt) else 1
        (args.output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        failed = {"status": "failed", "exit_code": rc, "mode": args.mode, "batch_size": args.batch_size}
        write_status(failed)
        if (args.output / "E2").is_dir():
            (args.output / "E2/status.json").write_text(json.dumps(failed, indent=2) + "\n", encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
