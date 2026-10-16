"""Tiny CPU fixture integration for the device smoke; no real-data experiment."""
from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch_geometric.data import HeteroData

from audit_graph_split import audit, load_container
from build_grouped_split import build, sha
from grouped_graphs import FrozenGraphs
from smoke_grouped_gpu import run


def fixture(root):
    data_root = root / "graphs"
    data_root.mkdir()
    source_split = {"train": [], "val": [], "test": []}
    for label in range(8):
        for group in range(3):
            index = label * 3 + group
            graph = HeteroData()
            graph["flow"].x = torch.full((1, 82), (index + 1) / 25.)
            payload = torch.arange(1500).float().remainder(256) / 16
            graph["packet"].x = torch.stack([payload, payload.flip(0), payload.roll(7)])
            contain = ("flow", "contain", "packet")
            reverse = ("packet", "rev_contain", "flow")
            link = ("packet", "link", "packet")
            graph[contain].edge_index = torch.tensor([[0, 0, 0], [0, 1, 2]])
            graph[reverse].edge_index = graph[contain].edge_index.flip(0)
            graph[contain].edge_attr = torch.tensor([[0., 64, 44, 36], [1., 128, 108, 100], [0., 192, 172, 164]])
            graph[reverse].edge_attr = graph[contain].edge_attr.clone()
            graph[link].edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
            graph[link].edge_attr = torch.tensor([.1, .1, 2., 2.])
            graph.y = torch.tensor(label)
            torch.save([graph], data_root / f"data_{index:03d}.pt")
            source_split[("train", "val", "test")[group]].append(index)
    source = root / "old_split.json"
    source.write_text(json.dumps({"total_graphs": 24, "split_indices": source_split,
        "split_label_counts": {part: {str(label): 1 for label in range(8)} for part in source_split}}))
    with redirect_stdout(io.StringIO()):
        audit(Namespace(trusted_data=data_root, split=source, output=root / "audit"))
        metadata = build(Namespace(manifest=root / "audit/graph_manifest.sqlite", audit_report=root / "audit/report.json",
                                  output=root / "benchmark", seed=42, val_ratio=.1, test_ratio=.1))
    return metadata


class GroupedGPUChecks(unittest.TestCase):
    def test_full_cpu_fixture_smoke_and_graph_tamper_detection(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root, prefix="grouped_smoke_test_") as temporary:
            root = Path(temporary)
            self.assertEqual(root.resolve().parent, temporary_root.resolve())
            meta = fixture(root)
            before = {p.name: sha(p) for p in (root / "graphs").iterdir()}
            args = Namespace(output=root / "smoke", device="cpu", split_dir=root / "benchmark",
                expected_split_id=meta["split_id"], data_root=None, seed=42,
                candidates_per_class=4, batch_size=16, experiments=["E0", "E1", "E2", "E3"])
            with redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertEqual(report["device"], "cpu")
            self.assertTrue(all(report["paired_initial_states_identical"].values()))
            self.assertEqual(set(report["models"]), set(args.experiments))
            self.assertTrue(all(m["parameter_update_verified"] for m in report["models"].values()))
            self.assertEqual(before, {p.name: sha(p) for p in (root / "graphs").iterdir()})
            frozen = FrozenGraphs(root / "benchmark", meta["split_id"])
            row = frozen.training_candidates()[0]
            path = frozen.root / row["path"]
            graphs = load_container(path, frozen.root)
            graphs[0]["flow"].x[0, 0] += 1
            torch.save(graphs, path)
            with self.assertRaisesRegex(ValueError, "differs from frozen"):
                frozen.load(row)

    def test_manifest_and_split_identity_mismatch_rejected(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root, prefix="grouped_smoke_test_") as temporary:
            root = Path(temporary)
            self.assertEqual(root.resolve().parent, temporary_root.resolve())
            meta = fixture(root)
            with self.assertRaisesRegex(ValueError, "Wrong split ID"):
                FrozenGraphs(root / "benchmark", "wrong")
            with (root / "benchmark/samples.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                FrozenGraphs(root / "benchmark", meta["split_id"])


if __name__ == "__main__":
    unittest.main()
