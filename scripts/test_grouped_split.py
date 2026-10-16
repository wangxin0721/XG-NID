"""Test metadata-only benchmark partitioning; no real graphs or training."""
from argparse import Namespace
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from build_grouped_split import build, sha


class GroupedSplitChecks(unittest.TestCase):
    def fixture(self, root):
        database = root / "audit.sqlite"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE graphs (idx INTEGER PRIMARY KEY,path TEXT,offset INTEGER,label INTEGER,part TEXT,edge_hash TEXT,node_hash TEXT)")
        index = 0
        for label in range(8):
            for group in range(3):
                node = hashlib.sha256(f"node:{label}:{group}".encode()).hexdigest()
                edge = hashlib.sha256(f"edge:{label}:{group}".encode()).hexdigest()
                connection.execute("INSERT INTO graphs VALUES (?,?,?,?,?,?,?)", (index, f"data_{index}.pt", 0, label, "train", edge, node))
                index += 1
                if label == 0 and group == 0:
                    # One exact copy, plus an informative attribute-only variant.
                    for key in (edge, hashlib.sha256(b"attribute variant").hexdigest()):
                        connection.execute("INSERT INTO graphs VALUES (?,?,?,?,?,?,?)", (index, f"data_{index}.pt", 0, label, "test", key, node))
                        index += 1
        connection.commit()
        connection.close()
        report = root / "report.json"
        report.write_text(json.dumps({"status": "completed", "exit_code": 0,
            "manifest_sha256": sha(database), "graph_count": index, "data_root": "/old/graphs",
            "metadata_checks": {k: True for k in ("total_matches_record", "all_indices_in_range", "complete_partition", "label_counts_match_record")}}))
        return Namespace(manifest=database, audit_report=report, output=root / "split_a",
                         seed=42, val_ratio=0.1, test_ratio=0.1)

    def test_reproducible_disjoint_groups_and_preserved_edge_variants(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root, prefix="grouped_split_test_") as temporary:
            root = Path(temporary)
            self.assertEqual(root.resolve().parent, temporary_root.resolve())
            args = self.fixture(root)
            original = sha(args.manifest)
            with redirect_stdout(io.StringIO()):
                a = build(args)
                args.output = root / "split_b"
                b = build(args)
            self.assertEqual(a["split_id"], b["split_id"])
            self.assertEqual(a["selected_graph_count"], 25)
            self.assertEqual(a["omitted_exact_edge_duplicate_rows"], 1)
            self.assertEqual(a["independent_input_groups"], 24)
            self.assertEqual(a["retained_edge_only_variants"], 1)
            self.assertEqual(sha(args.manifest), original)
            records = [json.loads(line) for line in (args.output / "samples.jsonl").read_text().splitlines()]
            partitions = defaultdict(set)
            for row in records:
                partitions[row["group_id"]].add(row["partition"])
            self.assertTrue(all(len(parts) == 1 for parts in partitions.values()))
            self.assertEqual(len({row["sample_id"] for row in records}), 25)
            with self.assertRaises(FileExistsError):
                build(args)

    def test_stale_manifest_rejected(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root, prefix="grouped_split_test_") as temporary:
            root = Path(temporary)
            self.assertEqual(root.resolve().parent, temporary_root.resolve())
            args = self.fixture(root)
            with args.manifest.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                build(args)
            self.assertFalse(args.output.exists())

    def test_conflicting_group_labels_rejected(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root, prefix="grouped_split_test_") as temporary:
            root = Path(temporary)
            self.assertEqual(root.resolve().parent, temporary_root.resolve())
            args = self.fixture(root)
            connection = sqlite3.connect(args.manifest)
            connection.execute("UPDATE graphs SET label=7 WHERE idx=0")
            connection.commit()
            connection.close()
            report = json.loads(args.audit_report.read_text())
            report["manifest_sha256"] = sha(args.manifest)
            args.audit_report.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Conflicting labels"):
                build(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
