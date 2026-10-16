"""Freeze a new input-grouped benchmark from a read-only graph audit database.

No torch dependency, graph loading, graph mutation, or historical split reuse.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import traceback


LABELS = ["Benign", "WebBased", "Spoofing", "Recon", "Mirai", "Dos", "DDos", "BruteForce"]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args):
    if not 0 < args.val_ratio < 1 or not 0 < args.test_ratio < 1 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Validation and test ratios must be positive and leave a training partition")
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    if audit.get("status") != "completed" or audit.get("exit_code") != 0:
        raise ValueError("Source graph audit did not complete")
    for key in ("total_matches_record", "all_indices_in_range", "complete_partition", "label_counts_match_record"):
        if audit.get("metadata_checks", {}).get(key) is not True:
            raise ValueError(f"Source graph audit metadata check failed: {key}")
    manifest_hash = sha(args.manifest)
    if manifest_hash != audit["manifest_sha256"]:
        raise ValueError("Source SQLite SHA256 does not match its audit report")
    connection = sqlite3.connect(args.manifest.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        total = connection.execute("SELECT COUNT(*) FROM graphs").fetchone()[0]
        if total != audit["graph_count"]:
            raise ValueError("Source graph count changed")
        conflicts = connection.execute(
            "SELECT node_hash FROM graphs GROUP BY node_hash HAVING COUNT(DISTINCT label)>1 LIMIT 1").fetchone()
        if conflicts:
            raise ValueError(f"Conflicting labels in No Edge input group: {conflicts[0]}")
        groups = {}
        for group_id, label in connection.execute("SELECT node_hash,MIN(label) FROM graphs GROUP BY node_hash ORDER BY node_hash"):
            groups.setdefault(label, []).append(group_id)
        if set(groups) != set(range(8)):
            raise ValueError(f"Expected eight labels, got {sorted(groups)}")
        assignment, group_counts = {}, {}
        for label, ids in sorted(groups.items()):
            random.Random(args.seed + label).shuffle(ids)
            n_val, n_test = max(1, int(len(ids) * args.val_ratio)), max(1, int(len(ids) * args.test_ratio))
            if len(ids) <= n_val + n_test:
                raise ValueError(f"Too few independent groups for class {label}")
            parts = {"val": ids[:n_val], "test": ids[n_val:n_val + n_test], "train": ids[n_val + n_test:]}
            group_counts[str(label)] = {part: len(values) for part, values in parts.items()}
            for part, values in parts.items():
                assignment.update({group_id: part for group_id in values})
        # One representative per exact Edge input. Its complete No Edge group
        # has already been assigned, preserving attribute-only variants together.
        query = """SELECT g.idx,g.path,g.offset,g.label,g.edge_hash,g.node_hash,g.part
            FROM graphs g JOIN (SELECT MIN(idx) AS representative FROM graphs GROUP BY edge_hash) chosen
            ON g.idx=chosen.representative ORDER BY g.label,g.node_hash,g.edge_hash"""
        selected = connection.execute(query).fetchall()
    finally:
        connection.close()
    if sha(args.manifest) != manifest_hash:
        raise ValueError("Source audit manifest changed while reading")
    for _, path, offset, _, _, _, _ in selected:
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or offset < 0:
            raise ValueError(f"Unsafe or invalid graph location: {path}:{offset}")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        samples = args.output / "samples.jsonl"
        sample_counts = {part: Counter() for part in ("train", "val", "test")}
        unique_edges, seen_groups = set(), {}
        with samples.open("x", encoding="utf-8") as stream:
            for new_index, (old_index, path, offset, label, edge_hash, node_hash, old_part) in enumerate(selected):
                part = assignment[node_hash]
                if edge_hash in unique_edges or seen_groups.get(node_hash, part) != part:
                    raise ValueError("Duplicate selected sample or cross-partition input group")
                unique_edges.add(edge_hash)
                seen_groups[node_hash] = part
                entry = {"manifest_index": new_index, "sample_id": "edge_sha256:" + edge_hash,
                         "group_id": "node_sha256:" + node_hash, "partition": part, "label": label,
                         "path": path, "container_offset": offset,
                         "edge_hash": edge_hash, "node_hash": node_hash,
                         "audit_index": old_index, "audit_partition": old_part}
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
                sample_counts[part][str(label)] += 1
        protocol = {"schema": "xgnid.grouped-input.v1", "seed": args.seed,
            "grouping": "No Edge input SHA256 (nodes + topology), excluding label",
            "deduplication": "One representative per Edge input SHA256; preserve edge-only variants",
            "representative": "minimum source audit index within each exact Edge input group",
            "ratios_by_group": {"train": 1 - args.val_ratio - args.test_ratio,
                                "val": args.val_ratio, "test": args.test_ratio},
            "rounding": "floor val/test groups per class, minimum one; remainder train",
            "label_mapping": dict(enumerate(LABELS)),
            "train_balancing": False, "class_weighting": False,
            "evaluation_unit": "one unique exact Edge input; edge-only variants in one No Edge group stay in one partition",
            "scope": "new benchmark over pooled existing graphs; not historical split reproduction",
            "flow_independence": "unverified: no original flow/capture identity; input grouping is not flow grouping"}
        identity = {"protocol": protocol, "samples_sha256": sha(samples), "source_manifest_sha256": manifest_hash}
        split_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        report = {"schema": protocol["schema"], "status": "completed", "exit_code": 0,
            "split_id": split_id, "protocol": protocol, "source_data_root": audit["data_root"],
            "source_manifest": str(args.manifest.resolve()), "source_manifest_sha256": manifest_hash,
            "source_audit_report_sha256": sha(args.audit_report),
            "builder_sha256": sha(Path(__file__)), "source_graph_count": total,
            "selected_graph_count": len(selected), "omitted_exact_edge_duplicate_rows": total - len(selected),
            "independent_input_groups": len(assignment),
            "retained_edge_only_variants": len(selected) - len(assignment),
            "sample_manifest": samples.name, "sample_manifest_sha256": identity["samples_sha256"],
            "partition_sizes": {part: sum(counts.values()) for part, counts in sample_counts.items()},
            "partition_label_counts": {part: dict(sorted(counts.items())) for part, counts in sample_counts.items()},
            "per_class_group_counts": group_counts,
            "checks": {"no_cross_partition_node_hash": True, "no_cross_partition_edge_hash": True,
                       "no_repeated_edge_hash": True, "all_groups_assigned_once": len(seen_groups) == len(assignment)},
            "graph_changes": "none; original 82/1500 features and topology retained; old graphs and splits untouched",
            "model_readiness": "not established by this tool; server model forward/backward/optimizer smoke still required"}
        if not all(report["checks"].values()):
            raise ValueError("New partition invariant failed")
        (args.output / "split.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (args.output / "status.json").write_text(json.dumps({"status": "completed", "exit_code": 0}), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("split_id", "selected_graph_count", "independent_input_groups",
            "retained_edge_only_variants", "partition_sizes", "partition_label_counts", "checks")}, indent=2), flush=True)
        return report
    except Exception:
        (args.output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        (args.output / "status.json").write_text(json.dumps({"status": "failed", "exit_code": 1}), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    build(parser.parse_args())
