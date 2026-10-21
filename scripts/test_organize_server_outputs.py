"""Metadata/fixture tests, standard library only; no experiment files changed."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

import organize_server_outputs as entry


class OrganizationTests(unittest.TestCase):
    def test_screenshot_groups(self):
        expected = {
            "e0_batch_full_b512_s42_shared_v1": "01_batch_comparisons",
            "e3_batch_full_b512_s42_shared_v2": "01_batch_comparisons",
            "grouped_full_E4_s42_v1": "02_grouped_E0_E4",
            "e3_gate_screen_E3a_b512_s42_v2": "03_gate_studies",
            "e3_gate_probe_E3a_b512_s42_v2": "03_gate_studies",
            "e3_gate_cache_analysis_b512_s42_v1": "04_gate_diagnostics",
            "e3_gate_diagnostic_b512_s42_v1": "04_gate_diagnostics",
            "r0_validation_full_s42_v1": "05_R0_references",
            "r0a_v2_cleanup_receipt.json": "05_R0_references",
            "grouped_gpu_smoke_E4_s42_v1": "06_probes_and_checks",
            "e2_e3_preflight_20261019T103529458611Z": "06_probes_and_checks",
            "e3_restart_snapshot_20261020T103302548182Z": "07_restart_snapshots",
            "GNN4ID_edge": "08_legacy",
            "unknown_file": "99_other",
        }
        for name, category in expected.items():
            with self.subTest(name=name):
                self.assertEqual(entry.classify(name), category)

    def test_inventory_pairs_sidecars_and_keeps_failed_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            (outputs / "grouped_full_E3_s42_v1").mkdir()
            for name in ("grouped_full_E3_s42_v1.log", "grouped_full_E3_s42_v1.restart.json",
                         "e3_batch_full_failed.log", "e3_batch_full_failed.pid"):
                (outputs / name).write_text("retained", encoding="utf-8")
            (outputs / entry.VIEW).mkdir()
            groups = entry.inventory(outputs)
            self.assertEqual(len(groups), 2)
            self.assertEqual(sorted(len(g["files"]) for g in groups), [2, 3])
            self.assertIn("未核实", entry.status_for(groups[0]))
            self.assertEqual(len(list(outputs.iterdir())), 6)

    def test_refuse_custom_workspace_and_unmanaged_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "outputs").mkdir()
            path = repo / entry.WORKSPACE
            path.write_text("custom workspace", encoding="utf-8")
            with self.assertRaises(ValueError):
                entry.preflight(repo, [])
            self.assertEqual(path.read_text(encoding="utf-8"), "custom workspace")
            path.unlink()
            (repo / "outputs" / entry.VIEW).mkdir()
            with self.assertRaises(OSError):
                entry.preflight(repo, [])

    def test_status_does_not_infer_process_liveness(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            run = outputs / "grouped_full_E0_s42_v1"
            run.mkdir()
            (run / "status.json").write_text(json.dumps({"status": "running", "pid": 123}), encoding="utf-8")
            value = entry.status_for(entry.inventory(outputs)[0])
            self.assertIn("running", value)
            self.assertIn("非进程检查", value)

    def test_python39_and_no_ml_imports(self):
        tree = ast.parse(Path(entry.__file__).read_text(encoding="utf-8"), feature_version=(3, 9))
        imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(imports.intersection({"torch", "numpy", "pandas", "train_grouped"}))

    @unittest.skipUnless(os.name == "posix", "Link fixture requires POSIX")
    def test_apply_idempotent_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            outputs = repo / "outputs"
            outputs.mkdir()
            run = outputs / "e0_batch_full_b512_s42_shared_v1"
            run.mkdir()
            metric = run / "metrics.json"
            metric.write_bytes(b'{"test": "unchanged"}')
            log = outputs / (run.name + ".log")
            log.write_bytes(b"epoch 1\n")
            before = metric.read_bytes(), log.read_bytes()
            groups = entry.inventory(outputs)
            self.assertEqual(entry.apply(repo, groups), 2)
            self.assertEqual(entry.apply(repo, entry.inventory(outputs)), 0)
            link = outputs / entry.VIEW / "01_batch_comparisons" / run.name / run.name
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), run)
            self.assertEqual((metric.read_bytes(), log.read_bytes()), before)


if __name__ == "__main__":
    unittest.main()
