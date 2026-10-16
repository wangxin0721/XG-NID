"""Stream trusted graph shards to audit a historical split by model-input hash.

Uses the repository loader's lexicographic recursive order. This reconstructs
the current order; old order is not provable without its historical manifest.
Only reads source graphs/split; writes a new SQLite manifest and JSON report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import torch
from torch_geometric.data import HeteroData


RELATION_DIMS = {("flow", "contain", "packet"): 4,
                 ("packet", "rev_contain", "flow"): 4,
                 ("packet", "link", "packet"): 1}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(graph, with_attributes):
    digest = hashlib.sha256()

    def add(name, value):
        digest.update(json.dumps(name).encode() + b"\n")
        if value is None:
            digest.update(b"null\n")
            return
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode() + b"\n")
        # View bytes supports dtypes which NumPy cannot represent directly.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())

    for node in sorted(graph.node_types):
        add([node, "x"], getattr(graph[node], "x", None))
    for relation in sorted(graph.edge_types):
        add([relation, "edge_index"], graph[relation].edge_index)
        if with_attributes:
            add([relation, "edge_attr"], getattr(graph[relation], "edge_attr", None))
    return digest.hexdigest()


def load_container(path, root):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    current = path
    for _ in range(3):
        if not isinstance(obj, (str, Path)):
            break
        raw = Path(obj)
        candidates = [raw]
        if not raw.is_absolute():
            candidates += [(base / raw).resolve() for base in (current.parent, current.parent.parent, Path.cwd())]
        target = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if target is None:
            break
        if root not in target.parents:
            raise ValueError(f"Path reference leaves selected data directory: {target}")
        current = target
        obj = torch.load(current, map_location="cpu", weights_only=False)
    if isinstance(obj, HeteroData):
        return [obj]
    if isinstance(obj, dict) and "graphs" in obj:
        obj = obj["graphs"]
    if not isinstance(obj, list) or not all(isinstance(g, HeteroData) for g in obj):
        raise TypeError(f"Unsupported graph container: {type(obj).__name__}")
    return obj


def overlap(connection, column, left, right):
    # Count affected samples and unique shared input signatures, not all pairs.
    query = f"""SELECT COUNT(*), COALESCE(SUM(a.n),0), COALESCE(SUM(b.n),0)
        FROM (SELECT {column} AS h,COUNT(*) AS n FROM graphs WHERE part=? GROUP BY {column}) a
        JOIN (SELECT {column} AS h,COUNT(*) AS n FROM graphs WHERE part=? GROUP BY {column}) b ON a.h=b.h"""
    groups, left_rows, right_rows = connection.execute(query, (left, right)).fetchone()
    return {"shared_input_groups": groups, "left_affected_samples": left_rows,
            "right_affected_samples": right_rows}


def audit(args):
    root = args.trusted_data.resolve()
    if not root.is_dir():
        raise ValueError("trusted-data must be an existing graph directory")
    record = json.loads(args.split.read_text(encoding="utf-8"))
    splits = record["split_indices"]
    membership = {}
    for name in ("train", "val", "test"):
        for index in splits[name]:
            if not isinstance(index, int) or index < 0 or index in membership:
                raise ValueError("Invalid, repeated, or cross-split index")
            membership[index] = name
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.output / "graph_manifest.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE graphs (idx INTEGER PRIMARY KEY, path TEXT, offset INTEGER, label INTEGER, part TEXT, edge_hash TEXT, node_hash TEXT)")
    counts = {name: Counter() for name in ("train", "val", "test", "unassigned")}
    shapes, issues = Counter(), Counter()
    skipped = []
    index = 0
    started = time.perf_counter()
    try:
        files = sorted(root.rglob("*.pt"))
        for file_number, path in enumerate(files, 1):
            try:
                graphs = load_container(path, root)
            except TypeError as exc:
                skipped.append({"path": str(path.relative_to(root)), "reason": str(exc)})
                continue
            for offset, graph in enumerate(graphs):
                label_tensor = getattr(graph, "y", None)
                if label_tensor is None or label_tensor.numel() != 1:
                    raise ValueError(f"Missing or nonscalar graph label: {path}")
                label = int(label_tensor.item())
                part = membership.get(index, "unassigned")
                counts[part][str(label)] += 1
                for node in graph.node_types:
                    x = graph[node].x
                    shapes[f"{node}.x width={x.size(-1)} dtype={x.dtype}"] += 1
                    if not torch.isfinite(x).all():
                        issues[f"{node}: nonfinite x"] += 1
                for relation, dim in RELATION_DIMS.items():
                    if relation not in graph.edge_types:
                        issues[f"missing relation {relation}"] += 1
                        continue
                    store = graph[relation]
                    attr = getattr(store, "edge_attr", None)
                    edges = store.edge_index.size(1)
                    if attr is None:
                        issues[f"missing attributes {relation}"] += 1
                    elif edges == 0 and attr.numel() == 0:
                        shapes[f"empty {relation} attr_shape={tuple(attr.shape)}"] += 1
                    elif attr.shape != (edges, dim) or not torch.isfinite(attr).all():
                        issues[f"invalid attributes {relation}"] += 1
                connection.execute("INSERT INTO graphs VALUES (?,?,?,?,?,?,?)", (
                    index, str(path.relative_to(root)), offset, label, part,
                    fingerprint(graph, True), fingerprint(graph, False)))
                index += 1
            if file_number % 2000 == 0:
                connection.commit()
                print(f"files={file_number}/{len(files)} graphs={index} seconds={time.perf_counter()-started:.1f}", flush=True)
        connection.commit()
        checks = {"total_matches_record": index == record.get("total_graphs"),
                  "all_indices_in_range": all(i < index for i in membership),
                  "complete_partition": len(membership) == index,
                  "label_counts_match_record": all(dict(counts[name]) ==
                     {str(k): int(v) for k, v in record.get("split_label_counts", {}).get(name, {}).items()}
                     for name in ("train", "val", "test"))}
        if not all(checks.values()):
            raise ValueError(f"Current loader reconstruction does not match split metadata: {checks}")
        results = {}
        for column in ("edge_hash", "node_hash"):
            connection.execute(f"CREATE INDEX idx_{column} ON graphs({column},part)")
            results[column] = {f"{a}/{b}": overlap(connection, column, a, b)
                              for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
            results[column]["within_split"] = {}
            for part in ("train", "val", "test"):
                rows, unique = connection.execute(
                    f"SELECT COUNT(*),COUNT(DISTINCT {column}) FROM graphs WHERE part=?", (part,)).fetchone()
                results[column]["within_split"][part] = {"samples": rows, "unique_inputs": unique,
                                                          "duplicate_rows": rows - unique}
            query = f"SELECT {column} FROM graphs GROUP BY {column} HAVING COUNT(DISTINCT label)>1 LIMIT 10"
            results[column]["conflicting_label_group_examples"] = [row[0] for row in connection.execute(query)]
        connection.commit()
        connection.close()
        report = {"status": "completed", "exit_code": 0, "data_root": str(root),
            "split_path": str(args.split.resolve()), "split_sha256": file_hash(args.split),
            "script_sha256": file_hash(Path(__file__)), "torch": torch.__version__,
            "ordering": "sorted recursive *.pt, graph order within each container; unsupported containers skipped",
            "historical_order_verified": False,
            "limitation": "Matching counts/labels is necessary but cannot prove historical graph ordering; require original manifest/code provenance. Exact input hashes cannot establish flow independence.",
            "definitions": {"edge_hash": "node features + topology + edge attributes; excludes label/graph_index/sample_id",
                            "node_hash": "node features + topology; excludes edge attributes and label",
                            "equality": "exact tensor dtype/shape/bytes; no normalization or isomorphism matching"},
            "graph_count": index, "split_label_counts": {k: dict(v) for k,v in counts.items()},
            "metadata_checks": checks, "input_shapes": dict(shapes), "graph_issues": dict(issues),
            "skipped_files": skipped, "overlap_under_current_order": results,
            "manifest_sha256": file_hash(database), "elapsed_seconds": time.perf_counter()-started}
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"graph_count": index, "metadata_checks": checks,
                          "overlap_under_current_order": results}, indent=2), flush=True)
        return report
    except Exception as exc:
        connection.close()
        (args.output / "status.json").write_text(json.dumps({"status": "failed", "exit_code": 1,
            "graphs_processed": index, "error": str(exc)}, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trusted-data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())
