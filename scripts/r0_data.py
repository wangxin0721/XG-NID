"""Freeze original-source train/test graph membership; never repartition inputs.

The filename mode recognizes ONLY official data_N.pt / data_test_N.pt.
Other containers require explicit JSONL locations: path, container_offset,
partition (train/test). Membership provenance must be supplied as a file.
This freezes current evidence, not proof of the authors' historical samples.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xgnid.edge_experiments import validate_graph
from audit_graph_split import fingerprint, load_container
from build_grouped_split import LABELS, sha


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def table4_counts():
    return {"train": {str(i): 20000 for i in range(8)},
            "test": {str(i): (1090 if i == 1 else 467 if i == 7 else 4000) for i in range(8)}}


def location(root, row):
    relative = Path(row["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or root not in path.parents:
        raise ValueError("Graph location leaves trusted directory")
    if type(row["container_offset"]) is not int or row["container_offset"] < 0:
        raise ValueError("Invalid container offset")
    return path


def check(graph):
    validate_graph(graph)
    if graph.node_types != ["flow", "packet"]:
        raise ValueError("Official pooling requires node insertion order flow, packet")
    if graph["flow"].x.shape != (1, 82) or graph["packet"].x.size(1) != 1500:
        raise ValueError("Expected Flow82/Packet1500")
    if not 1 <= graph["packet"].x.size(0) <= 20:
        raise ValueError("Expected 1..20 packets")
    if any(graph[node].x.dtype != torch.float32 for node in graph.node_types):
        raise ValueError("Expected float32 features")
    if graph.y.numel() != 1 or graph.y.dtype != torch.long or graph.y.item() not in range(8):
        raise ValueError("Invalid class label")
    n = graph["packet"].x.size(0)
    contain = graph["flow", "contain", "packet"].edge_index
    links = graph["packet", "link", "packet"].edge_index
    if contain.size(1) != n or set(map(tuple, contain.t().tolist())) != {(0, i) for i in range(n)}:
        raise ValueError("Contain topology differs from official builder")
    expected_links = {(i, i + 1) for i in range(n - 1)} | {(i + 1, i) for i in range(n - 1)}
    if links.size(1) != 2 * (n - 1) or set(map(tuple, links.t().tolist())) != expected_links:
        raise ValueError("Expected official ToUndirected consecutive Packet links")


def prepare(args):
    root = args.trusted_data.resolve()
    evidence = args.membership_evidence.read_text(encoding="utf-8")
    if not evidence.strip() or not root.is_dir():
        raise ValueError("Need data directory and nonempty membership provenance")
    if args.locations:
        locations = [json.loads(line) for line in args.locations.read_text(encoding="utf-8").splitlines()]
    else:
        locations = []
        for path in sorted(root.rglob("*.pt")):
            match = re.fullmatch(r"data_(test_)?(\d+)\.pt", path.name)
            if match:
                locations.append({"path": path.relative_to(root).as_posix(), "container_offset": 0,
                                  "partition": "test" if match[1] else "train"})
    if not locations:
        raise ValueError("No official filenames; supply --locations, never infer from counts")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "status.json", {"status": "running"})
    try:
        counts = {part: Counter() for part in ("train", "test")}
        seen, inputs, nodes = set(), {}, {}
        with (args.output / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for index, source in enumerate(locations):
                part = source["partition"]
                if part not in counts:
                    raise ValueError("Only source train/test membership is accepted")
                path = location(root, source)
                key = (str(path), source["container_offset"])
                if key in seen:
                    raise ValueError("Repeated graph location (including cross-partition)")
                seen.add(key)
                graphs = load_container(path, root)
                if not args.locations and len(graphs) != 1:
                    raise ValueError("Multi-graph container requires explicit --locations")
                graph = graphs[source["container_offset"]]
                check(graph)
                label = graph.y.item()
                edge, node = fingerprint(graph, True), fingerprint(graph, False)
                for groups, digest in ((inputs, edge), (nodes, node)):
                    if digest not in groups:
                        groups[digest] = {"train": 0, "test": 0, "labels": set()}
                    groups[digest][part] += 1
                    groups[digest]["labels"].add(label)
                row = {"manifest_index": index, "sample_id": f"r0:{index}:{edge}",
                       "path": path.relative_to(root).as_posix(), "container_offset": source["container_offset"],
                       "partition": part, "label": label, "edge_hash": edge, "node_hash": node}
                stream.write(json.dumps(row) + "\n")
                counts[part][str(label)] += 1
                if (index + 1) % 2000 == 0:
                    print(f"R0 manifest {index + 1}/{len(locations)}", flush=True)
        if any(set(c) != set(map(str, range(8))) for c in counts.values()):
            raise ValueError("Both partitions must cover all eight classes")
        expected = table4_counts()
        def overlap(groups):
            shared = [v for v in groups.values() if v["train"] and v["test"]]
            return {"shared_groups": len(shared), "affected_test_samples": sum(v["test"] for v in shared),
                    "within_partition_duplicate_samples": {p: sum(max(0, v[p] - 1) for v in groups.values()) for p in counts},
                    "conflicting_label_groups": sum(len(v["labels"]) > 1 for v in groups.values())}
        (args.output / "membership_evidence.txt").write_text(evidence, encoding="utf-8")
        meta = {"schema": "xgnid.r0.manifest.v1", "status": "completed", "claim": "protocol_reimplementation",
                "historical_author_sample_identity_verified": False,
                "source_data_root": str(root), "labels": LABELS,
                "sample_manifest_sha256": sha(args.output / "samples.jsonl"),
                "membership_evidence_sha256": sha(args.output / "membership_evidence.txt"),
                "partition_label_counts": {p: dict(c) for p, c in counts.items()},
                "table4_counts_match": counts == expected,
                "membership_mode": "explicit_locations" if args.locations else "official_filenames",
                "locations_sha256": sha(args.locations) if args.locations else None,
                "edge_overlap": overlap(inputs), "node_overlap": overlap(nodes)}
        meta["manifest_id"] = identity(meta)
        write(args.output / "manifest.json", meta)
        write(args.output / "status.json", {"status": "completed", "exit_code": 0})
        return meta
    except BaseException as exc:
        write(args.output / "status.json", {"status": "failed", "exit_code": 1, "error": str(exc)})
        raise


class R0Graphs(Dataset):
    def __init__(self, directory, partition, data_root=None, limit_per_class=None):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        meta = dict(self.meta)
        manifest_id = meta.pop("manifest_id")
        if meta.get("schema") != "xgnid.r0.manifest.v1" or meta.get("status") != "completed" or identity(meta) != manifest_id:
            raise ValueError("Invalid R0 manifest identity")
        for name, field in (("samples.jsonl", "sample_manifest_sha256"), ("membership_evidence.txt", "membership_evidence_sha256")):
            if sha(self.directory / name) != meta[field]:
                raise ValueError("R0 manifest/evidence SHA256 mismatch")
        self.root = Path(data_root or meta["source_data_root"]).resolve()
        self.rows = []
        counts = {p: Counter() for p in ("train", "test")}
        selected, seen = Counter(), set()
        for index, line in enumerate((self.directory / "samples.jsonl").read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            path = location(self.root, row)
            key = (path, row["container_offset"])
            if row["manifest_index"] != index or key in seen:
                raise ValueError("R0 index/duplicate location mismatch")
            seen.add(key)
            counts[row["partition"]][str(row["label"])] += 1
            if row["partition"] == partition and (limit_per_class is None or selected[row["label"]] < limit_per_class):
                self.rows.append(row)
                selected[row["label"]] += 1
        if {p: dict(c) for p, c in counts.items()} != meta["partition_label_counts"]:
            raise ValueError("R0 counts mismatch")
        if set(selected) != set(range(8)):
            raise ValueError("Selected partition misses classes")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        graph = load_container(location(self.root, row), self.root)[row["container_offset"]]
        check(graph)
        if (fingerprint(graph, True), fingerprint(graph, False), graph.y.item()) != (row["edge_hash"], row["node_hash"], row["label"]):
            raise ValueError("Graph input/label changed after R0 freeze")
        graph = graph.clone()
        relation = ("packet", "link", "packet")
        if graph[relation].edge_attr.ndim == 1:
            graph[relation].edge_attr = graph[relation].edge_attr.unsqueeze(-1)
        graph.sample_id = row["sample_id"]
        return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-data", type=Path, required=True)
    parser.add_argument("--membership-evidence", type=Path, required=True)
    parser.add_argument("--locations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2))
