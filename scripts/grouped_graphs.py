"""Verified, explicit graph locations from the frozen grouped-input manifest."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_graph_split import fingerprint, load_container
from build_grouped_split import sha


class FrozenGraphs:
    def __init__(self, split_dir, expected_split_id=None, data_root=None):
        self.directory = Path(split_dir).resolve()
        self.metadata = json.loads((self.directory / "split.json").read_text(encoding="utf-8"))
        meta = self.metadata
        if meta.get("schema") != "xgnid.grouped-input.v1" or meta.get("status") != "completed":
            raise ValueError("Expected completed grouped-input.v1 benchmark")
        if expected_split_id is not None and meta["split_id"] != expected_split_id:
            raise ValueError("Wrong split ID")
        manifest = (self.directory / meta["sample_manifest"]).resolve()
        if manifest.parent != self.directory or sha(manifest) != meta["sample_manifest_sha256"]:
            raise ValueError("Sample manifest location or SHA256 mismatch")
        identity = {"protocol": meta["protocol"], "samples_sha256": meta["sample_manifest_sha256"],
                    "source_manifest_sha256": meta["source_manifest_sha256"]}
        calculated = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        required_checks = ("no_cross_partition_node_hash", "no_cross_partition_edge_hash",
                           "no_repeated_edge_hash", "all_groups_assigned_once")
        if calculated != meta["split_id"] or not all(meta.get("checks", {}).get(k) is True for k in required_checks):
            raise ValueError("Split identity/invariants mismatch")
        self.root = Path(data_root or meta["source_data_root"]).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Graph directory does not exist: {self.root}")
        self.rows = []
        groups, edges = {}, set()
        counts = {part: Counter() for part in ("train", "val", "test")}
        with manifest.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                part, label = row["partition"], row["label"]
                if part not in counts or label not in range(8) or row["manifest_index"] != len(self.rows):
                    raise ValueError("Invalid sample partition, label or index")
                node, edge = row["node_hash"], row["edge_hash"]
                if row["sample_id"] != "edge_sha256:" + edge or row["group_id"] != "node_sha256:" + node:
                    raise ValueError("Sample/group identity mismatch")
                if edge in edges or groups.get(node, (part, label)) != (part, label):
                    raise ValueError("Repeated Edge input or cross-partition input group")
                if Path(row["path"]).is_absolute() or ".." in Path(row["path"]).parts or row["container_offset"] < 0:
                    raise ValueError("Invalid graph location")
                groups[node] = (part, label)
                edges.add(edge)
                counts[part][str(label)] += 1
                self.rows.append(row)
        if len(self.rows) != meta["selected_graph_count"] or len(groups) != meta["independent_input_groups"]:
            raise ValueError("Manifest sample/group count mismatch")
        for part, values in counts.items():
            if dict(values) != meta["partition_label_counts"][part] or sum(values.values()) != meta["partition_sizes"][part]:
                raise ValueError("Manifest partition label counts mismatch")

    def training_candidates(self, per_class=4):
        counts, by_label = Counter(), {label: [] for label in range(8)}
        for row in self.rows:
            label = row["label"]
            if row["partition"] == "train" and counts[label] < per_class:
                by_label[label].append(row)
                counts[label] += 1
        if set(counts) != set(range(8)):
            raise ValueError("Training candidates do not cover all eight classes")
        # Round-robin avoids a batch containing only the first class in a manifest.
        return [by_label[label][i] for i in range(per_class) for label in range(8) if i < len(by_label[label])]

    def load(self, row):
        path = (self.root / row["path"]).resolve()
        if self.root not in path.parents:
            raise ValueError("Resolved graph location leaves selected directory")
        graphs = load_container(path, self.root)
        graph = graphs[row["container_offset"]]
        if fingerprint(graph, True) != row["edge_hash"] or fingerprint(graph, False) != row["node_hash"]:
            raise ValueError(f"Actual graph differs from frozen inputs: {row['path']}")
        if graph.y.numel() != 1 or graph.y.item() != row["label"]:
            raise ValueError(f"Actual graph label differs from manifest: {row['path']}")
        if graph["flow"].x.shape != (1, 82) or graph["packet"].x.size(1) != 1500:
            raise ValueError("Expected unchanged legacy Flow82/Packet1500 features")
        # A local view adapter makes scalar and matrix shards batch-compatible.
        # Fingerprints were checked before this; the source graph is not saved.
        graph = graph.clone()
        relation = ("packet", "link", "packet")
        if graph[relation].edge_attr.ndim == 1:
            graph[relation].edge_attr = graph[relation].edge_attr.unsqueeze(-1)
        graph.sample_id = row["sample_id"]
        graph.manifest_index = row["manifest_index"]
        return graph
