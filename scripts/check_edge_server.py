"""Read-only server inventory; stdlib-only unless a trusted graph is selected.

Does not rewrite paths, rebuild splits, import training, or launch experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=15)
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def inventory(root):
    root = root.resolve()
    result = {"root": str(root), "exists": root.exists(), "counts": {}, "bytes": {},
              "csv_files": [], "pt_examples": [], "identity_candidates": []}
    counts, sizes = Counter(), Counter()
    if not root.is_dir():
        return result
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in (".csv", ".pt", ".pcap", ".pcapng", ".json", ".parquet"):
            continue
        counts[suffix] += 1
        sizes[suffix] += path.stat().st_size
        if suffix == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                columns = next(csv.reader(handle), [])
            result["csv_files"].append({"path": str(path), "bytes": path.stat().st_size,
                                        "columns": columns})
        if suffix == ".pt" and len(result["pt_examples"]) < 30:
            result["pt_examples"].append(str(path))
        if (suffix in (".pcap", ".pcapng") or any(word in path.name.lower()
            for word in ("manifest", "flow_id", "capture", "provenance"))) and len(result["identity_candidates"]) < 50:
            result["identity_candidates"].append(str(path))
    result["counts"], result["bytes"] = dict(counts), dict(sizes)
    return result


def split_check(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    result = {"path": str(path.resolve()), "sha256": sha(path),
              "recorded_data_paths": {k: {"value": record[k], "exists_on_this_host": Path(record[k]).exists()}
                 for k in ("data_path", "data_path_resolved") if isinstance(record.get(k), str)},
              "total_graphs": record.get("total_graphs"),
              "recorded_label_counts": record.get("split_label_counts"),
              "sample_identity_present": bool(record.get("samples") or record.get("sample_ids")),
              "limitation": "Index disjointness does not establish flow disjointness or match graph ordering."}
    splits = record.get("split_indices", {})
    if not isinstance(splits, dict) or not all(name in splits for name in ("train", "val", "test")):
        raise ValueError("Expected explicit split_indices with train/val/test; no schema guessed")
    sets = {name: set(ids) for name, ids in splits.items()}
    result["sizes"] = {name: len(ids) for name, ids in splits.items()}
    result["repeated_indices"] = {name: len(ids) - len(sets[name]) for name, ids in splits.items()}
    result["cross_split_index_overlap"] = {f"{a}/{b}": len(sets[a] & sets[b])
        for a in sets for b in sets if a < b}
    if isinstance(record.get("total_graphs"), int):
        result["invalid_index_count"] = sum(not isinstance(i, int) or i < 0 or i >= record["total_graphs"]
            for ids in splits.values() for i in ids)
    return result


def graph_check(path, limit):
    import torch
    from torch_geometric.data import HeteroData
    sys.path.insert(0, str(ROOT / "src"))
    from xgnid.edge_experiments import validate_graph
    # Explicit --trusted-graph-file input only; never auto-load arbitrary .pt.
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, HeteroData):
        graphs = [obj]
    elif isinstance(obj, list):
        graphs = obj
    elif isinstance(obj, dict) and isinstance(obj.get("graphs"), list):
        graphs = obj["graphs"]
    else:
        raise TypeError(f"Unsupported container {type(obj).__name__}; no path indirection followed")
    result = {"path": str(path.resolve()), "sha256": sha(path), "graph_count": len(graphs),
              "sample_only": True, "checked_graphs": []}
    for i, graph in enumerate(graphs[:limit]):
        entry = {"index": i}
        if not isinstance(graph, HeteroData):
            entry["error"] = f"Expected HeteroData, got {type(graph).__name__}"
        else:
            entry["nodes"] = {n: list(graph[n].x.shape) if getattr(graph[n], "x", None) is not None else None for n in graph.node_types}
            entry["edges"] = {str(r): {"edge_index": list(graph[r].edge_index.shape),
                "edge_attr": list(graph[r].edge_attr.shape) if getattr(graph[r], "edge_attr", None) is not None else None} for r in graph.edge_types}
            relation = ("packet", "link", "packet")
            if relation in graph.edge_types:
                index = graph[relation].edge_index
                if index.ndim == 2 and index.size(0) == 2:
                    pairs = set(map(tuple, index.t().tolist()))
                    entry["link_topology"] = {"edges": index.size(1),
                        "unique_edges_with_reverse": sum((b, a) in pairs for a, b in pairs),
                        "max_in_degree": int(torch.bincount(index[1]).max()) if index.numel() else 0}
            try:
                validate_graph(graph)
                entry["edge_contract"] = "passed"
            except (ValueError, AttributeError, KeyError) as exc:
                entry["edge_contract"] = "failed"
                entry["error"] = str(exc)
        result["checked_graphs"].append(entry)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, action="append")
    parser.add_argument("--split", type=Path, action="append", default=[])
    graph_source = parser.add_mutually_exclusive_group()
    graph_source.add_argument("--trusted-graph-file", type=Path)
    graph_source.add_argument("--trusted-graph-directory", type=Path)
    parser.add_argument("--graph-limit", type=int, default=8)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new report name")
    if args.graph_limit < 1:
        parser.error("graph-limit must be positive")
    packages = {}
    for name in ("torch", "torch-geometric", "pandas", "numpy", "scikit-learn", "tqdm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
              "python": sys.executable, "python_version": platform.python_version(), "packages": packages,
              "git": {}, "inventory": [], "splits": [], "errors": [],
              "assessment": "Inventory only; formal readiness requires source identity, split/graph mapping, leakage audit and Edge smoke evidence."}
    for name, argv in (("head", ["git", "rev-parse", "HEAD"]), ("status", ["git", "status", "--short"])):
        try:
            report["git"][name] = command(argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["git"][name] = {"error": str(exc)}
    for root in args.data_root or [ROOT / "data"]:
        try:
            report["inventory"].append(inventory(root))
        except (OSError, ValueError, csv.Error) as exc:
            report["errors"].append({"path": str(root), "error": str(exc)})
    for path in args.split:
        try:
            report["splits"].append(split_check(path))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report["errors"].append({"path": str(path), "error": str(exc)})
    if args.check_runtime:
        try:
            import torch
            import torch_geometric
            report["runtime"] = {"torch": torch.__version__, "pyg": torch_geometric.__version__,
                "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        except Exception as exc:
            report["errors"].append({"stage": "runtime", "error": str(exc)})
    if args.trusted_graph_file:
        try:
            report["graph"] = graph_check(args.trusted_graph_file, args.graph_limit)
            for entry in report["graph"]["checked_graphs"]:
                if "error" in entry:
                    report["errors"].append({"stage": "graph", "index": entry["index"], "error": entry["error"]})
        except Exception as exc:
            report["errors"].append({"stage": "graph", "error": str(exc)})
    if args.trusted_graph_directory:
        report["graph_samples"] = []
        checked = 0
        root = args.trusted_graph_directory
        if not root.is_dir():
            report["errors"].append({"stage": "graph", "error": f"Missing directory: {root}"})
        else:
            # Explicitly named trusted directory; select shards, not metadata.
            for path in sorted(root.rglob("*.pt")):
                if path.name in ("pre_filter.pt", "pre_transform.pt"):
                    continue
                try:
                    result = graph_check(path, args.graph_limit - checked)
                    report["graph_samples"].append(result)
                    checked += len(result["checked_graphs"])
                    for entry in result["checked_graphs"]:
                        if "error" in entry:
                            report["errors"].append({"stage": "graph", "path": str(path), "error": entry["error"]})
                except Exception as exc:
                    report["errors"].append({"stage": "graph", "path": str(path), "error": str(exc)})
                    break
                if checked >= args.graph_limit:
                    break
            if checked == 0:
                report["errors"].append({"stage": "graph", "error": "No graph samples checked"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["exit_code"] = 1 if report["errors"] else 0
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Report: {args.output.resolve()}; inspection errors: {len(report['errors'])}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
