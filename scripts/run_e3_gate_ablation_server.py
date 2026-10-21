"""Independent server-only E3a probe and train/validation screening.

Requires an accepted E2/E3 validation diagnostic. Preserves all frozen source
files. E3a removes only direct post-gate proxy mixing; E3a_equal_init additionally
zeros the final gate Linear after the original paired initialization. Screening
never builds a test loader or evaluates test graphs. No automatic chaining.
"""
from __future__ import annotations

import argparse
from itertools import islice
import json
import os
from pathlib import Path
import sys
import time
import traceback

from e3_gate_study import (activate_gpu, apply_variant, check_diagnostic, checked_gpu,
    compute_processes, guard_graph_loads, initial_model, inside, read, reference_metadata,
    require, runtime, sha, write)


def snapshot(gpu):
    try:
        return {"processes": compute_processes(gpu["uuid"]), "scope": "point-in-time observation only"}
    except Exception as exc:
        return {"query_error": str(exc), "scope": "unknown; no isolated timing claim"}


def diagnostic_receipt(path, reference):
    path = path.resolve()
    report = read(path)
    check_diagnostic(report, reference)
    status = read(path.parent / "status.json")
    require((status.get("status"), status.get("exit_code"), status.get("mode")) == ("completed", 0, "diagnose"), "Diagnostic process did not finish")
    require(report["shared_helper_sha256"] == sha(Path(__file__).with_name("e3_gate_study.py")), "Diagnostic helper version differs")
    for name, result in report["models"].items():
        require(read(path.parent / name / "diagnostics.json") == result, "Diagnostic report/details differ")
        require(sha(path.parent / name / "validation_cache.jsonl.gz") == result["cache_sha256"], "Diagnostic cache changed")
    return {"report_sha256": sha(path), "entry_sha256": report["entry_sha256"],
            "helper_sha256": report["shared_helper_sha256"]}


def parent_receipt(args, spec, reference, diagnostic):
    if args.variant == "E3a":
        require(args.parent_screen_report is None, "E3a starts fresh and has no parent screening run")
        return None
    require(args.parent_screen_report is not None, "Equal-init screening requires completed E3a --parent-screen-report")
    path = args.parent_screen_report.resolve()
    parent = read(path)
    status = read(path.parent / "status.json")
    require((parent.get("status"), parent.get("exit_code"), parent.get("mode"), parent.get("variant")) ==
            ("completed", 0, "screen", "E3a"), "Complete pure-gate screening first")
    require((status.get("status"), status.get("exit_code")) == ("completed", 0), "Parent screening process incomplete")
    require(parent["binding"]["reference"] == reference and parent["binding"]["diagnostic"] == diagnostic,
            "Parent screening used another diagnostic/reference")
    require(parent["binding"]["spec_sha256"] == sha(args.config)
            and parent["binding"]["entry_sha256"] == sha(Path(__file__))
            and parent["binding"]["helper_sha256"] == sha(Path(__file__).with_name("e3_gate_study.py")), "Parent screening implementation differs")
    require(parent["model"]["test"] is None and parent["test_evaluated"] is False, "Parent screening touched test")
    require(parent["wall_seconds"] < spec["budget"]["screening_round_gpu_hour_ceiling"] * 3600,
            "E3a already exhausted the round budget; do not add another job")
    folder = path.parent / "E3a"
    require(sha(folder / "config.json") == parent["model"]["config_sha256"], "Parent config changed")
    for file, digest in parent["model"]["checkpoint_sha256"].items():
        require(sha(folder / file) == digest, "Parent screening checkpoint changed")
    return {"summary_sha256": sha(path), "wall_seconds": parent["wall_seconds"],
            "initialization": "reference E3 initialization, never parent trained weights"}


def new_model(core, frozen, device, variant):
    model, batch = initial_model(core, frozen, device)
    certificate = apply_variant(core, model, variant)
    captured = {}
    handle = model.gate.register_forward_hook(lambda module, values, value: captured.update(g=value))
    before = core.state_hash(model.state_dict().items())
    try:
        with core.torch.no_grad():
            output = model(batch)
            weights = model._last_cache["gate_weights"]
            require(core.torch.equal(weights, captured["g"]), "Pure-gate output is still mixed with a proxy")
            require(bool(core.torch.isfinite(output).all()), "Nonfinite initial output")
            if variant == "E3a_equal_init":
                require(core.torch.equal(weights, core.torch.full_like(weights, .5)), "Equal-init actual weights are not exactly .5/.5")
    finally:
        handle.remove()
    require(core.state_hash(model.state_dict().items()) == before, "Initial eval changed model state")
    certificate["actual_initial_weight_mean"] = weights.mean(0).cpu().tolist()
    return model, certificate


def update(core, model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = core.F.nll_loss(model(batch), batch.y.view(-1))
    require(bool(core.torch.isfinite(loss)), "Nonfinite training loss")
    loss.backward()
    core.torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def probe(core, args, spec, frozen, device, binding, gpu):
    model, certificate = new_model(core, frozen, device, args.variant)
    optimizer = core.torch.optim.Adam(model.parameters(), lr=.001)
    before = core.state_hash(model.named_parameters())
    rows = core.select_rows(frozen, "train")
    batches = list(islice(core.TrainBatches(len(rows), 512, 42), 3))
    loader = core.DataLoader(core.GraphRows(frozen, rows), batch_sampler=batches, num_workers=0)
    measured = []
    captured = {}
    handle = model.gate.register_forward_hook(lambda module, values, value: captured.update(g=value))
    try:
        for index, batch in enumerate(loader):
            require(batch.num_graphs == 512, "Probe requires complete 512-graph batches")
            batch = batch.to(device)
            if index == 0:
                update(core, model, optimizer, batch)
            core.sync(device)
            core.torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            loss = update(core, model, optimizer, batch)
            core.sync(device)
            step_seconds = time.perf_counter() - tick
            weights = model._last_cache["gate_weights"].detach()
            require(core.torch.equal(weights, captured["g"].detach()), "Unexpected proxy mix in training")
            require(bool(core.torch.isfinite(weights).all()) and bool((weights >= 0).all())
                    and bool((weights <= 1).all()) and bool(core.torch.allclose(weights.sum(-1), weights.new_ones(weights.size(0)), atol=1e-6)), "Invalid pure-gate weights")
            gate_grads = {name: float(p.grad.abs().max()) if p.grad is not None else None
                          for name, p in model.gate.named_parameters()}
            require(max((v for v in gate_grads.values() if v is not None), default=0.) > 0, "Gate has no nonzero gradient")
            item = {"batch": index + 1, "graphs": 512, "loss": loss, "gate_gradient_max_by_parameter": gate_grads,
                    "gate_weight_mean": weights.mean(0).cpu().tolist(),
                    "resident_step_seconds_excluding_graph_io_and_transfer": step_seconds,
                    "peak_cuda_allocated_bytes": core.torch.cuda.max_memory_allocated(device),
                    "peak_cuda_reserved_bytes": core.torch.cuda.max_memory_reserved(device),
                    "other_compute_processes_snapshot": snapshot(gpu)}
            measured.append(item)
            print(json.dumps({"variant": args.variant, "probe": item}), flush=True)
    finally:
        handle.remove()
    require(len(measured) == 3 and core.state_hash(model.named_parameters()) != before, "Probe incomplete or optimizer made no update")
    return {"status": "completed", "exit_code": 0, "mode": "probe", "variant": args.variant,
            "binding": binding, "hardware": gpu, "initialization": certificate,
            "warmup_updates": 1, "measured_updates": 3, "batches": measured,
            "optimizer_update_verified": True, "pure_gate_verified": True, "test": None, "test_evaluated": False}


def screen(core, args, spec, frozen, device, binding, gpu):
    require(args.probe_report is not None, "Screen needs a successful new --probe-report")
    probe_path = args.probe_report.resolve()
    report, status = read(probe_path), read(probe_path.parent / "status.json")
    require((report.get("status"), report.get("exit_code"), report.get("mode"), report.get("variant")) ==
            ("completed", 0, "probe", args.variant), "Probe incomplete or wrong variant")
    require((status.get("status"), status.get("exit_code"), status.get("mode"), status.get("variant")) ==
            ("completed", 0, "probe", args.variant), "Probe process did not finish")
    require(report["binding"] == binding and report["optimizer_update_verified"] is True
            and report["pure_gate_verified"] is True and report["test"] is None
            and (report["warmup_updates"], report["measured_updates"]) == (1, 3), "Probe provenance or budget differs")
    require(len(report["batches"]) == 3 and all(item["graphs"] == 512 for item in report["batches"]), "Probe batch size differs")
    folder = args.output / args.variant
    folder.mkdir()
    write(folder / "status.json", {"status": "running", "mode": "screen"})
    model, certificate = new_model(core, frozen, device, args.variant)
    require(certificate == report["initialization"], "Probe and screening initialization differ")
    optimizer = core.torch.optim.Adam(model.parameters(), lr=.001)
    rows = {part: core.select_rows(frozen, part) for part in ("train", "val")}
    sampler = core.TrainBatches(len(rows["train"]), 512, 42)
    kwargs = {"num_workers": 2, "pin_memory": True, "persistent_workers": True, "multiprocessing_context": "fork"}
    train_loader = core.DataLoader(core.GraphRows(frozen, rows["train"]), batch_sampler=sampler, **kwargs)
    val_loader = core.DataLoader(core.GraphRows(frozen, rows["val"]), batch_size=16, shuffle=False, **kwargs)
    cfg = {"schema": spec["schema"], "mode": "screen", "variant": args.variant, "controls": spec["controls"],
           "epochs": 100, "workers": 2, "initialization_batch_size": 16, "evaluation_batch_size": 16,
           "binding": binding, "initialization": certificate, "probe_sha256": sha(probe_path),
           "test": "disabled", "same_epoch_ceiling_not_equal_optimizer_update_total": True}
    write(folder / "config.json", cfg)
    best, best_epoch, stale, updates, stopped = -1., 0, 0, 0, 0
    started = time.perf_counter()
    with (folder / "epochs.jsonl").open("x", encoding="utf-8") as log:
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
            require(count == 107621 and epoch_updates == len(sampler) == 211, "Incomplete training epoch")
            updates += epoch_updates
            validation, _ = core.evaluate(model, val_loader, device)
            improved = validation["macro_f1"] > best
            if improved:
                best, best_epoch, stale = validation["macro_f1"], epoch, 0
            else:
                stale += 1
            elapsed = time.perf_counter() - started
            prior_seconds = (binding.get("parent_screen") or {}).get("wall_seconds", 0.)
            entry = {"epoch": epoch, "train_loss": loss_sum / count, "train_samples": count,
                     "train_seconds": train_seconds, "train_order_sha256": sampler.last_order_sha256,
                     "optimizer_updates": epoch_updates, "cumulative_optimizer_updates": updates,
                     "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"],
                     "val_weighted_f1": validation["weighted_f1"], "val_wall_seconds": validation["wall_seconds_including_graph_io"],
                     "gate_weights": validation.get("gate_weights"), "best_epoch": best_epoch, "stale_epochs": stale,
                     "peak_cuda_allocated_bytes": core.torch.cuda.max_memory_allocated(device),
                     "peak_cuda_reserved_bytes": core.torch.cuda.max_memory_reserved(device),
                     "other_compute_processes_snapshot": snapshot(gpu),
                     "elapsed_screen_seconds": elapsed, "round_elapsed_including_parent_seconds": prior_seconds + elapsed}
            if epoch in (2, 3):
                entry["projected_max_100_epoch_round_gpu_hours"] = (prior_seconds + elapsed / epoch * 100) / 3600
                entry["do_not_add_jobs_if_projection_exceeds_round_budget"] = entry["projected_max_100_epoch_round_gpu_hours"] > 8
            payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                       "config": cfg, "epoch": epoch, "best_epoch": best_epoch, "best_val_macro_f1": best,
                       "validation": validation, "sampler_rng_state": sampler.generator.get_state(),
                       "torch_rng_state": core.torch.get_rng_state(), "cuda_rng_states": core.torch.cuda.get_rng_state_all(),
                       "optimizer_updates": updates, "stale_epochs": stale}
            core.checkpoint(folder / "last.pt", payload)
            if improved:
                core.checkpoint(folder / "best.pt", payload)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps({"variant": args.variant, **entry}), flush=True)
            stopped = epoch
            if stale >= 12:
                break
    selected = core.torch.load(folder / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state"], strict=True)
    metrics = {"mode": "screen", "variant": args.variant, "best_epoch": best_epoch, "stopped_epoch": stopped,
               "best_validation": selected["validation"], "test": None, "test_evaluated": False,
               "parameter_count": 846800, "initialization": certificate, "train_batch_size": 512,
               "evaluation_batch_size": 16, "total_optimizer_updates": updates, "training_samples_seen": 107621 * stopped,
               "run_wall_seconds": time.perf_counter() - started, "config_sha256": sha(folder / "config.json"),
               "checkpoint_sha256": {file: sha(folder / file) for file in ("best.pt", "last.pt")}}
    write(folder / "metrics.json", metrics)
    write(folder / "status.json", {"status": "completed", "exit_code": 0, "mode": "screen"})
    return {"status": "completed", "exit_code": 0, "mode": "screen", "variant": args.variant,
            "binding": binding, "hardware": gpu, "model": metrics, "test_evaluated": False,
            "note": "Validation screening completed; no test score. No automatic next run or final test."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("check", "probe", "screen"), default="check")
    parser.add_argument("--variant", choices=("E3a", "E3a_equal_init"), default="E3a")
    parser.add_argument("--diagnostic-report", type=Path)
    parser.add_argument("--parent-screen-report", type=Path)
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1))
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(os.name == "posix", "Execute on the Linux training server only")
    args.repo, args.config = args.repo.resolve(), args.config.resolve()
    spec = read(args.config)
    reference = reference_metadata(args.repo, spec)
    if args.mode == "check":
        print(json.dumps({"status": "metadata_checks_passed", "reference": reference, "variant": args.variant,
            "diagnostic_required_before_probe": True, "gpu_touched": False, "training_started": False}, indent=2), flush=True)
        return 0
    require(args.diagnostic_report is not None, "Probe/screen requires --diagnostic-report")
    diagnostic = diagnostic_receipt(args.diagnostic_report, reference)
    parent = parent_receipt(args, spec, reference, diagnostic)
    require(args.physical_gpu is not None and args.output is not None, "Explicit GPU and new output required")
    args.output = inside(args.repo / "outputs", args.output.resolve())
    require(args.output.name.startswith("e3_gate_") and not args.output.exists(), "Use a new outputs/e3_gate_* directory")
    gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
    activate_gpu(gpu)
    binding = {"reference": reference, "diagnostic": diagnostic, "parent_screen": parent,
               "spec_sha256": sha(args.config), "variant": args.variant,
               "entry_sha256": sha(Path(__file__)), "helper_sha256": sha(Path(__file__).with_name("e3_gate_study.py")),
               "gpu_uuid": gpu["uuid"], "allow_shared_gpu": args.allow_shared_gpu}
    args.output.mkdir()
    write(args.output / "status.json", {"status": "running", "mode": args.mode, "variant": args.variant, "pid": os.getpid()})
    started = time.perf_counter()
    try:
        sys.path.insert(0, str(args.repo / "scripts"))
        import train_grouped as core
        device = runtime(core)
        frozen = core.FrozenGraphs(inside(args.repo, spec["split_directory"]), spec["split_id"])
        guard_graph_loads(frozen, {"train"} if args.mode == "probe" else {"train", "val"})
        result = probe(core, args, spec, frozen, device, binding, gpu) if args.mode == "probe" else screen(core, args, spec, frozen, device, binding, gpu)
        result["wall_seconds"] = time.perf_counter() - started
        result["test_graphs_loaded"] = 0
        write(args.output / "summary.json", result)
        write(args.output / "status.json", {"status": "completed", "exit_code": 0, "mode": args.mode, "variant": args.variant})
        print(json.dumps({"status": "completed", "mode": args.mode, "variant": args.variant,
            "output": str(args.output), "test_evaluated": False}), flush=True)
    except BaseException as exc:
        failed = {"status": "failed", "exit_code": 130 if isinstance(exc, KeyboardInterrupt) else 1, "mode": args.mode, "variant": args.variant}
        write(args.output / "status.json", failed)
        if (args.output / args.variant).is_dir():
            write(args.output / args.variant / "status.json", failed)
        (args.output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
