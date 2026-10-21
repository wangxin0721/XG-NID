"""Standard-library checks only; never import torch or deserialize checkpoints."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import run_e0_batch512 as entry


ROOT = Path(__file__).resolve().parents[1]


class E0ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = entry.read(ROOT / "configs/e0_batch512.json")

    def test_registered_controls_reject_drift(self):
        entry.check_config(self.config, 512)
        for key, value in (("lr", .016), ("seed", 43), ("class_weighting", True)):
            changed = deepcopy(self.config)
            changed["reference_controls"][key] = value
            with self.assertRaises(ValueError):
                entry.check_config(changed, 512)
        with self.assertRaises(ValueError):
            entry.check_config(self.config, 16)

    def test_real_reference_bytes_and_source_identity(self):
        checks = entry.metadata_checks(ROOT, self.config)
        self.assertEqual(checks["reference_e0_initial_state_sha256"],
                         "906e125925a8a2cc53a526602db0f047f76b1e41b8087c893f5cf1af41f64a6b")
        self.assertEqual(len(checks["sources"]), 9)
        self.assertNotIn("torch", sys.modules)

    def test_training_update_and_epoch_loop_preserved(self):
        original = ast.parse((ROOT / "scripts/run_e2_batch_comparison.py").read_text(encoding="utf-8"))
        current = ast.parse((ROOT / "scripts/run_e0_batch512.py").read_text(encoding="utf-8"), feature_version=(3, 9))
        def function(tree, name):
            return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        self.assertEqual(ast.dump(function(original, "update")), ast.dump(function(current, "update")))
        def epoch_loop(tree):
            return next(node for node in ast.walk(function(tree, "full"))
                        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch")
        expected = ast.parse(ast.unparse(epoch_loop(original)).replace("E2 batch=", "E0 batch="))
        actual = ast.parse(ast.unparse(epoch_loop(current)))
        self.assertEqual(ast.dump(expected), ast.dump(actual))


if __name__ == "__main__":
    unittest.main()
