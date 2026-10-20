"""Resume the paused original E3 in a NEW output directory after reboot.

The frozen training file is never edited. A validated AST copy of train_one
adds checkpoint restoration, epoch/log continuation and recovery accounting;
the original training, validation, checkpoint and final-test bodies are reused.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.loads(json.dumps(value))


def patch_function(source):
    """Fail closed if any frozen syntax hook is absent or duplicated."""
    tree = ast.parse(source, feature_version=9)
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_one")
    function = copy.deepcopy(original)
    counts = {key: 0 for key in ("config", "state", "loop", "log", "metrics", "test")}

    class ResumeHooks(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name == "config":
                    counts["config"] += 1
                    node.value = ast.Name(id="__e3_saved_config", ctx=ast.Load())
                if name == "best_checkpoint":
                    counts["test"] += 1
                    marker = ast.parse("__e3_test_access(output)").body[0]
                    return [ast.copy_location(marker, node), node]
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
                names = [item.id for item in node.targets[0].elts if isinstance(item, ast.Name)]
                if names == ["best", "best_epoch", "stale", "history"]:
                    counts["state"] += 1
                    node.value = ast.parse("__e3_restore(model, optimizer, sampler, output)", mode="eval").body
            return self.generic_visit(node)

        def visit_For(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == "epoch":
                counts["loop"] += 1
                node.iter = ast.parse("__e3_remaining_epochs(args)", mode="eval").body
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "open" and node.args:
                if any(isinstance(item, ast.Constant) and item.value == "epochs.jsonl" for item in ast.walk(node.func.value)):
                    require(isinstance(node.args[0], ast.Constant) and node.args[0].value == "x", "Frozen log mode changed")
                    counts["log"] += 1
                    node.args[0] = ast.Constant(value="a")
            return self.generic_visit(node)

        def visit_Expr(self, node):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "write" and value.args:
                if any(isinstance(item, ast.Constant) and item.value == "metrics.json" for item in ast.walk(value.args[0])):
                    counts["metrics"] += 1
                    annotation = ast.parse("__e3_annotate_metrics(metrics)").body[0]
                    return [ast.copy_location(annotation, node), node]
            return self.generic_visit(node)

    function = ResumeHooks().visit(function)
    require(all(value == 1 for value in counts.values()), f"Frozen resume hooks do not match: {counts}")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, "<isolated E3 epoch-boundary resume>", "exec"), counts


def load_snapshot(repo):
    original = repo / "outputs/grouped_full_E3_s42_v1"
    record = read(original.with_suffix(".restart.json"))
    require(record["status"] == "snapshot_verified_process_stopped" and Path(record["original_run"]).resolve() == original,
            "Wrong pause receipt or experiment")
    directory = Path(record["snapshot"]).resolve()
    directory.relative_to((repo / "outputs").resolve())
    for name in ("config.json", "last.pt", "best.pt", "epochs.jsonl"):
        require(sha(directory / name) == record["files_sha256"][name], f"Snapshot SHA changed: {name}")
    config = read(directory / "config.json")
    require(config["experiment"] == "E3" and config["mode"] == "full", "Expected full E3 checkpoint")
    require(config["controls"]["seed"] == 42 and config["controls"]["batch_size"] == 16 and config["epochs"] == 100,
            "Not the original batch16/seed42/100-epoch E3")
    for name, digest in config["source_hashes"].items():
        path = (repo / name).resolve()
        path.relative_to(repo)
        require(sha(path) == digest, f"Frozen source differs after reboot: {name}")
    require(metadata.version("torch") == config["torch"] and metadata.version("torch-geometric") == config["pyg"],
            "Restore the same torch/PyG environment; do not silently change versions")
    import torch
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    required = {"model_state", "optimizer_state", "sampler_rng_state", "torch_rng_state", "cuda_rng_states",
                "config", "epoch", "best_epoch", "best_val_macro_f1", "validation"}
    require(required <= set(last), "Incomplete last checkpoint")
    require(canonical(last["config"]) == config and canonical(best["config"]) == config, "Checkpoint config differs")
    require(last["epoch"] == record["saved_epoch"] and last["best_epoch"] == record["best_epoch"], "Pause receipt epoch differs")
    if best["epoch"] != last["best_epoch"]:
        require(last["epoch"] == last["best_epoch"], "Cannot recover recorded best weights")
        best = last
    history, discarded = [], []
    lines = (directory / "epochs.jsonl").read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            require(index == len(lines) - 1, "Invalid non-trailing epoch record")
            discarded.append("partial trailing log line")
            continue
        if entry["epoch"] <= last["epoch"]:
            history.append(entry)
        else:
            discarded.append(entry["epoch"])
    require([entry["epoch"] for entry in history] == list(range(1, last["epoch"] + 1)), "Checkpoint history is not contiguous")
    selected = max(history, key=lambda entry: entry["val_macro_f1"])
    require(selected["epoch"] == last["best_epoch"] and selected["val_macro_f1"] == last["best_val_macro_f1"], "Historical best selection differs")
    require(best["validation"]["macro_f1"] == last["best_val_macro_f1"], "Best checkpoint score differs")
    require(history[-1]["val_macro_f1"] == last["validation"]["macro_f1"], "Last checkpoint and log differ")
    code, hooks = patch_function((repo / "scripts/train_grouped.py").read_text(encoding="utf-8"))
    return record, directory, config, last, best, history, discarded, code, hooks


def refuse_live_original(repo, original, output):
    # Only inspect our own processes, never signal them or unrelated jobs.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.geteuid():
                continue
            argv = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if "--output" not in argv or not any(Path(value).name in ("train_grouped.py", "resume_e3_server.py", "xgnid_resume_e3.py") for value in argv):
                continue
            destination = ((proc / "cwd").resolve() / argv[argv.index("--output") + 1]).resolve()
            if int(proc.name) != os.getpid() and destination in (original, output):
                raise RuntimeError(f"Original/resumed run process still exists: PID {proc.name}; no duplicate resume")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/grouped_full_E3_s42_resume_v1"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--resume", action="store_true")
    cli = parser.parse_args()
    require(os.name == "posix", "Run only on the Linux server")
    repo = cli.repo.resolve()
    output = (repo / cli.output).resolve()
    output.relative_to((repo / "outputs").resolve())
    require(not output.exists(), "Continuation directory already exists; inspect it, never overwrite/repeat")
    record, snapshot, config, last, best, history, discarded, code, hooks = load_snapshot(repo)
    stale = last["epoch"] - last["best_epoch"]
    receipt = {"schema": "xgnid.e3-resume.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "tool_sha256": sha(Path(__file__)), "snapshot": str(snapshot), "original_run": record["original_run"],
               "output": str(output), "snapshot_files_sha256": record["files_sha256"], "saved_epoch": last["epoch"],
               "start_epoch": last["epoch"] + 1, "best_epoch": last["best_epoch"], "stale_epochs": stale,
               "epoch_budget": config["epochs"], "discarded_uncheckpointed_log_records": discarded,
               "ast_hooks": hooks, "original_files_modified": False,
               "limitations": ["Epoch-boundary restoration only; any partial next epoch is repeated",
                               "Original checkpoints did not save Python/NumPy/worker RNG or elapsed wall time",
                               "Workers restart and CUDA can be nondeterministic; bitwise equivalence is not guaranteed",
                               "If original final testing had begun, recovery repeats inference with the same fixed best weights; no test-driven selection"]}
    print(json.dumps({"status": "checkpoint_checks_passed", "saved_epoch": last["epoch"], "next_epoch": last["epoch"] + 1,
                      "best_epoch": last["best_epoch"], "stale_epochs": stale, "epoch_budget": config["epochs"],
                      "check_only": cli.check_only, "training_started": False, "output": str(output)}, ensure_ascii=False, indent=2), flush=True)
    if cli.check_only:
        return
    refuse_live_original(repo, Path(record["original_run"]), output)
    sys.path.insert(0, str(repo / "scripts"))
    import train_grouped as g
    require(g.torch.cuda.is_available(), "CUDA unavailable; no CPU fallback")
    require(g.torch.cuda.device_count() == len(last["cuda_rng_states"]), "Visible CUDA device count changed; use CUDA_VISIBLE_DEVICES=0")
    require(g.torch.cuda.get_device_name(0) == config["device_name"], "GPU model changed; do not silently change hardware")
    old_argv = record["argv"]

    def saved_path(flag):
        return (repo / old_argv[old_argv.index(flag) + 1]).resolve()

    controls = config["controls"]
    args = argparse.Namespace(output=output, mode="full", experiments=["E3"], device="cuda:0", seed=controls["seed"],
        batch_size=controls["batch_size"], lr=controls["lr"], epochs=config["epochs"], patience=controls["early_stop_patience"],
        workers=config["workers"], split_dir=Path(config["split_directory"]), data_root=Path(config["data_root"]),
        expected_split_id=config["split_id"], smoke_report=saved_path("--smoke-report"), pilot_report=saved_path("--pilot-report"),
        baseline_report=saved_path("--baseline-report"), fusion_control_report=saved_path("--fusion-control-report"),
        pilot_train_per_class=32, pilot_val_per_class=16)
    require(g.sha(args.smoke_report) == config["smoke_report_sha256"], "Original smoke report changed")

    def restore(model, optimizer, sampler, directory):
        require(g.sha(directory / "config.json") == record["files_sha256"]["config.json"], "Continuation config bytes differ")
        model.load_state_dict(last["model_state"], strict=True)
        optimizer.load_state_dict(last["optimizer_state"])
        sampler.generator.set_state(last["sampler_rng_state"].cpu())
        shutil.copy2(snapshot / "last.pt", directory / "last.pt")
        shutil.copy2(snapshot / ("last.pt" if record["best_checkpoint_from_last"] else "best.pt"), directory / "best.pt")
        with (directory / "epochs.jsonl").open("x", encoding="utf-8") as stream:
            for entry in history:
                stream.write(json.dumps(entry) + "\n")
        g.write(output / "restart.json", receipt)
        g.torch.set_rng_state(last["torch_rng_state"].cpu())
        g.torch.cuda.set_rng_state_all([state.cpu() for state in last["cuda_rng_states"]])
        print(f"Restored E3 epoch={last['epoch']}, best_epoch={last['best_epoch']}, stale={stale}; budget unchanged", flush=True)
        return last["best_val_macro_f1"], last["best_epoch"], stale, list(history)

    def remaining_epochs(current_args):
        if stale >= current_args.patience or last["epoch"] >= current_args.epochs:
            return range(0)
        return range(last["epoch"] + 1, current_args.epochs + 1)

    def test_access(directory):
        selected = g.torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        g.write(output / "test_access.json", {"checkpoint_epoch": selected["epoch"],
                                               "scope": "recovery final inference; no parameter selection",
                                               "created_utc": datetime.now(timezone.utc).isoformat()})

    def annotate_metrics(metrics):
        metrics["restart"] = receipt
        metrics["wall_time_scope"] = "Resume session only; pre-reboot elapsed time and downtime excluded"

    g.__dict__.update(__e3_saved_config=last["config"], __e3_restore=restore, __e3_remaining_epochs=remaining_epochs,
                      __e3_annotate_metrics=annotate_metrics, __e3_test_access=test_access)
    exec(code, g.__dict__)
    g.run(args)


if __name__ == "__main__":
    main()
