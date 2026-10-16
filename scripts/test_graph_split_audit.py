"""Regression checks on tiny in-memory fixtures; no training or real-data scan."""
from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch_geometric.data import HeteroData

from audit_graph_split import audit, file_hash, fingerprint


def graph(label=0, value=1.):
    data = HeteroData()
    data["flow"].x = torch.tensor([[value, 2.]])
    data["packet"].x = torch.tensor([[1., 2.], [3., 4.]])
    data["flow", "contain", "packet"].edge_index = torch.tensor([[0, 0], [0, 1]])
    data["packet", "rev_contain", "flow"].edge_index = torch.tensor([[0, 1], [0, 0]])
    data["packet", "link", "packet"].edge_index = torch.tensor([[0], [1]])
    for relation in data.edge_types:
        width = 1 if relation[1] == "link" else 4
        data[relation].edge_attr = torch.ones((data[relation].edge_index.size(1), width))
    data.y = torch.tensor([label])
    return data


class GraphAuditChecks(unittest.TestCase):
    def test_hash_excludes_label_and_identifies_edge_only_changes(self):
        a, b = graph(), graph(label=1)
        self.assertEqual(fingerprint(a, True), fingerprint(b, True))
        b["packet", "link", "packet"].edge_attr *= 2
        self.assertNotEqual(fingerprint(a, True), fingerprint(b, True))
        self.assertEqual(fingerprint(a, False), fingerprint(b, False))
        b["packet"].x[0, 0] += 1
        self.assertNotEqual(fingerprint(a, False), fingerprint(b, False))

    def test_stream_finds_content_overlap_without_index_overlap(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root, prefix="graph_audit_test_") as temporary:
            root = Path(temporary)
            self.assertEqual(root.resolve().parent, temporary_root.resolve())
            data = root / "graphs"
            data.mkdir()
            for i, g in enumerate((graph(), graph(label=1, value=2.), graph())):
                torch.save([g], data / f"data_{i}.pt")
            split = root / "split.json"
            split.write_text(json.dumps({"total_graphs": 3,
                "split_indices": {"train": [0], "val": [1], "test": [2]},
                "split_label_counts": {"train": {"0": 1}, "val": {"1": 1}, "test": {"0": 1}}}))
            before = {p.name: file_hash(p) for p in data.iterdir()}
            args = Namespace(trusted_data=data, split=split, output=root / "result")
            with redirect_stdout(io.StringIO()):
                report = audit(args)
            overlap = report["overlap_under_current_order"]["edge_hash"]["train/test"]
            self.assertEqual(overlap, {"shared_input_groups": 1, "left_affected_samples": 1, "right_affected_samples": 1})
            self.assertFalse(report["historical_order_verified"])
            self.assertEqual(before, {p.name: file_hash(p) for p in data.iterdir()})
            with self.assertRaises(FileExistsError):
                audit(args)


if __name__ == "__main__":
    unittest.main()
