"""Standard-library regression checks; no ML imports or model execution."""
import ast
import copy
from pathlib import Path
import subprocess
import sys
import unittest

from run_e3_gate_ablation_server import compare_initialization

ROOT = Path(__file__).resolve().parents[1]


class InitializationComparisonChecks(unittest.TestCase):
    def certificate(self):
        return {"variant": "E3a", "forward_ast_sha256": "forward-identity", "removed_statements": 2,
                "initialization_changed_tensors": [], "original_initial_state_sha256": "original-state",
                "variant_initial_state_sha256": "variant-state", "actual_initial_weight_mean": [.49, .51]}

    def test_accepts_roundoff_in_diagnostic_means_and_records_difference(self):
        current, saved = self.certificate(), self.certificate()
        current["actual_initial_weight_mean"] = [.49000003, .50999997]
        result = compare_initialization(current, saved)
        self.assertTrue(result["identity_fields_match_exactly"])
        self.assertGreater(result["mean_max_abs_difference"], 0)
        self.assertLess(result["mean_max_abs_difference"], 1e-5)

    def test_all_identity_fields_still_require_exact_match(self):
        for key in ("variant", "forward_ast_sha256", "removed_statements", "initialization_changed_tensors",
                    "original_initial_state_sha256", "variant_initial_state_sha256"):
            current, saved = self.certificate(), self.certificate()
            current[key] = "changed-identity"
            with self.assertRaisesRegex(ValueError, key):
                compare_initialization(current, saved)

    def test_rejects_large_or_invalid_mean_differences(self):
        for values in ([.491, .509], [float("nan"), .5], [1.1, -.1], [.5], [.6, .6], [True, False]):
            current, saved = self.certificate(), self.certificate()
            current["actual_initial_weight_mean"] = values
            with self.assertRaises(ValueError):
                compare_initialization(current, saved)
        current, saved = self.certificate(), self.certificate()
        current["extra"] = True
        with self.assertRaises(ValueError):
            compare_initialization(current, saved)

    def test_model_probe_update_and_epoch_loop_unchanged_from_delivered_version(self):
        old = subprocess.check_output(["git", "show", "fda02b1:scripts/run_e3_gate_ablation_server.py"], cwd=ROOT).decode("utf-8")
        new = (ROOT / "scripts/run_e3_gate_ablation_server.py").read_text(encoding="utf-8")
        old_nodes = {n.name: n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)}
        new_nodes = {n.name: n for n in ast.parse(new, feature_version=(3, 9)).body if isinstance(n, ast.FunctionDef)}
        for name in ("new_model", "update", "probe", "diagnostic_receipt", "parent_receipt", "main"):
            self.assertEqual(ast.dump(old_nodes[name], include_attributes=False), ast.dump(new_nodes[name], include_attributes=False), name)
        def epoch_loop(node):
            return next(n for n in ast.walk(node) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "epoch")
        self.assertEqual(ast.dump(epoch_loop(old_nodes["screen"]), include_attributes=False),
                         ast.dump(epoch_loop(new_nodes["screen"]), include_attributes=False))
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
