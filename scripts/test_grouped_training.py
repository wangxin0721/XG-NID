"""Tiny CPU training-loop integration, using artificial fixtures only."""
from argparse import Namespace
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from grouped_graphs import FrozenGraphs
from smoke_grouped_gpu import run as smoke
from test_grouped_gpu_smoke import fixture
from train_grouped import TrainBatches, run


class GroupedTrainingChecks(unittest.TestCase):
    def test_singleton_batches_merge_without_dropping_samples(self):
        for size in (2, 3, 16, 17, 32, 33):
            sampler = TrainBatches(size, 16, 42)
            batches = list(sampler)
            indices = [index for batch in batches for index in batch]
            self.assertEqual(len(batches), len(sampler))
            self.assertEqual(sorted(indices), list(range(size)))
            self.assertTrue(all(len(batch) >= 2 for batch in batches))

    def test_pilot_then_full_and_baseline_guards(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root, prefix="grouped_training_test_") as temporary:
            root = Path(temporary)
            self.assertEqual(root.resolve().parent, temporary_root.resolve())
            meta = fixture(root)
            smoke_args = Namespace(output=root / "smoke", device="cpu", split_dir=root / "benchmark",
                expected_split_id=meta["split_id"], data_root=None, seed=42,
                candidates_per_class=4, batch_size=16, experiments=["E0", "E1", "E2", "E3"])
            args = Namespace(output=root / "pilot", device="cpu", split_dir=root / "benchmark",
                expected_split_id=meta["split_id"], data_root=None, seed=42,
                smoke_report=root / "smoke/smoke.json", pilot_report=None, baseline_report=None,
                fusion_control_report=None, epochs=2, mode="pilot", batch_size=4, lr=.001,
                patience=12, workers=0, pilot_train_per_class=32, pilot_val_per_class=16,
                experiments=["E0", "E1"])
            loaded_parts = []
            original_load = FrozenGraphs.load

            def track_load(instance, row):
                loaded_parts.append(row["partition"])
                return original_load(instance, row)

            with redirect_stdout(io.StringIO()):
                smoke(smoke_args)
                with patch.object(FrozenGraphs, "load", track_load):
                    pilot = run(args)
            self.assertEqual(set(loaded_parts), {"train", "val"})
            self.assertTrue(all(metrics["test"] is None for metrics in pilot["models"].values()))
            orders = []
            for name in ("E0", "E1"):
                self.assertTrue((root / "pilot" / name / "best.pt").exists())
                self.assertTrue((root / "pilot" / name / "last.pt").exists())
                epochs = [json.loads(line) for line in (root / "pilot" / name / "epochs.jsonl").read_text().splitlines()]
                orders.append([entry["train_order_sha256"] for entry in epochs])
                self.assertTrue(all(entry["train_samples"] == 8 for entry in epochs))
            self.assertEqual(orders[0], orders[1])
            args.mode, args.epochs, args.experiments = "full", 1, ["E0"]
            args.output = root / "missing_pilot"
            with self.assertRaisesRegex(ValueError, "pilot-report"):
                run(args)
            args.pilot_report = root / "pilot/summary.json"
            args.output = root / "full_e0"
            with redirect_stdout(io.StringIO()):
                full = run(args)
            self.assertEqual(full["models"]["E0"]["test"]["samples"], 8)
            with (root / "full_e0/E0/predictions.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            args.experiments, args.output = ["E1"], root / "missing_baseline"
            with self.assertRaisesRegex(ValueError, "baseline-report"):
                run(args)
            args.baseline_report = root / "full_e0/summary.json"
            args.epochs, args.output = 2, root / "mismatched_budget"
            with self.assertRaisesRegex(ValueError, "controls/budget"):
                run(args)
            args.epochs, args.experiments, args.output = 1, ["E3"], root / "missing_fusion_control"
            with self.assertRaisesRegex(ValueError, "fusion-control-report"):
                run(args)


if __name__ == "__main__":
    unittest.main()
