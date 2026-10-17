"""Synthetic CPU integration checks, never a real-data reproduction result."""
from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import HeteroData, Batch
from torch_geometric.transforms import ToUndirected

from r0_data import prepare, R0Graphs, identity, write
from run_r0_edge import ROOT, run, configuration, verify_smoke
from build_grouped_split import sha
from xgnid.edge_experiments import build_experiment


def fixture(root):
    data = root / "graphs"
    data.mkdir()
    generator = torch.Generator().manual_seed(73)
    for index in range(24):
        graph = HeteroData()
        graph["flow"].x = torch.rand(1, 82, generator=generator)
        graph["packet"].x = torch.rand(4, 1500, generator=generator)
        graph["flow", "contain", "packet"].edge_index = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]])
        graph["flow", "contain", "packet"].edge_attr = torch.rand(4, 4, generator=generator) * 100
        graph["packet", "link", "packet"].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        graph["packet", "link", "packet"].edge_attr = torch.tensor([.1, 1., 10.])
        graph.y = torch.tensor(index % 8)
        name = f"data_{index}.pt" if index < 16 else f"data_test_{index - 16}.pt"
        torch.save(ToUndirected()(graph), data / name)
    evidence = root / "provenance.txt"
    evidence.write_text("Synthetic fixture; source filenames deliberately assigned; no historical identity.")
    args = Namespace(trusted_data=data, membership_evidence=evidence, locations=None, output=root / "manifest")
    meta = prepare(args)
    return data, args, meta


class R0Checks(unittest.TestCase):
    def test_smoke_authorization_requires_cuda_identity_and_checkpoint(self):
        # Synthetic authorization records test validation only, not GPU execution.
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="r0_gate_") as folder:
            root = Path(folder)
            checkpoint = root / "last.pt"
            checkpoint.write_bytes(b"fixture checkpoint, not a trained model")
            binding = {"manifest_id": "fixture", "config_id": "fixture", "code": {"fixture": "v1"},
                       "runtime": {"device_type": "cuda", "torch": "fixture"}}
            path = root / "summary.json"
            summary = {"status": "completed", "exit_code": 0, "mode": "smoke", "binding": binding,
                       "optimizer_step_passed": True, "epochs_completed": 2,
                       "last_checkpoint_sha256": sha(checkpoint)}
            write(path, summary)
            verify_smoke(path, binding)
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                verify_smoke(path, {**binding, "runtime": {"device_type": "cpu"}})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_smoke(path, {**binding, "manifest_id": "other"})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_smoke(path, {**binding, "code": {"fixture": "v2"}})
            summary["epochs_completed"] = 1
            write(path, summary)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                verify_smoke(path, binding)
            summary["epochs_completed"] = 2
            write(path, summary)
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                verify_smoke(path, binding)

    def test_synthetic_training_and_guards(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="r0_test_") as folder:
            root = Path(folder)
            data, prep, meta = fixture(root)
            self.assertFalse(meta["historical_author_sample_identity_verified"])
            self.assertFalse(meta["table4_counts_match"])
            with self.assertRaises(FileExistsError):
                prepare(prep)
            args = Namespace(config=ROOT / "configs/r0_edge.json", mode="smoke", device="cpu", workers=0,
                manifest_dir=root / "manifest", data_root=None, output=root / "smoke", smoke_report=None)
            loaded = []
            original = R0Graphs.__getitem__
            def track(dataset, index):
                loaded.append(dataset.rows[index]["partition"])
                return original(dataset, index)
            with redirect_stdout(io.StringIO()), patch.object(R0Graphs, "__getitem__", track):
                result = run(args)
            self.assertEqual(set(loaded), {"train"})
            self.assertEqual(result["epochs_completed"], 2)
            self.assertTrue(result["optimizer_step_passed"])
            self.assertEqual(len(result["gat_attribute_shapes"]), 6)
            self.assertIsNone(result["test"])
            self.assertTrue((args.output / "last.pt").exists())
            with self.assertRaises(FileExistsError):
                run(args)
            args.mode, args.output, args.smoke_report = "full", root / "full", root / "smoke/summary.json"
            with self.assertRaisesRegex(ValueError, "Table 4"):
                run(args)
            # A changed summary flag cannot override the actual per-class counts.
            meta["table4_counts_match"] = True
            meta.pop("manifest_id")
            meta["manifest_id"] = identity(meta)
            write(root / "manifest/manifest.json", meta)
            with self.assertRaisesRegex(ValueError, "Table 4"):
                run(args)
            self.assertFalse(args.output.exists())
            # Mutated source tensors are detected at load, before training.
            dataset = R0Graphs(root / "manifest", "train")
            row = dataset.rows[0]
            graph_path = data / row["path"]
            graph = torch.load(graph_path, weights_only=False)
            graph["flow"].x += 1
            torch.save(graph, graph_path)
            with self.assertRaisesRegex(ValueError, "changed after R0 freeze"):
                dataset[0]
            samples = root / "manifest/samples.jsonl"
            samples.write_text(samples.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                R0Graphs(root / "manifest", "train")

    def test_frozen_e3_computation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="r0_semantics_") as folder:
            root = Path(folder)
            fixture(root)
            dataset = R0Graphs(root / "manifest", "train")
            batch = Batch.from_data_list([dataset[i] for i in range(8)])
            torch.manual_seed(42)
            model = build_experiment("E3", dataset[0])
            model.eval()
            with torch.no_grad():
                scores = model(batch)
                hf, hp, zf, zp = model._branch_outputs(batch)
                self.assertEqual(tuple(hf.shape), (8, 64))
                self.assertEqual(tuple(hp.shape), (8, 64))
                self.assertEqual(tuple(zf.shape), (8, 8))
                confidence = [model._confidence_from_logits(z) for z in (zf, zp)]
                margins = [model._margin_from_logits(z) for z in (zf, zp)]
                agreement = torch.nn.functional.cosine_similarity(zf, zp, dim=-1)
                raw = model.gate(zf, zp, *confidence, *margins, agreement)
                weights = .9 * raw + .1 * torch.stack(confidence, dim=-1)
                weights /= weights.sum(-1, keepdim=True).clamp_min(1e-8)
                expected = torch.nn.functional.log_softmax(model.shared_head(weights[:, :1] * zf + weights[:, 1:] * zp), dim=-1)
                torch.testing.assert_close(scores, expected)
                torch.testing.assert_close(weights, model._last_cache["gate_weights"])
                model.use_gate = False
                expected_equal = torch.nn.functional.log_softmax(model.shared_head(.5 * zf + .5 * zp), dim=-1)
                torch.testing.assert_close(model(batch), expected_equal)

    def test_reject_unreviewed_config(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="r0_config_") as folder:
            config = json.loads((ROOT / "configs/r0_edge.json").read_text())
            config["lr"] = .001
            path = Path(folder) / "bad.json"
            write(path, config)
            with self.assertRaisesRegex(ValueError, "differs from reviewed"):
                configuration(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
