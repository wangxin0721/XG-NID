"""Standard-library checks only: no torch, model state load or graph execution."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest

from e3_gate_study import (check_diagnostic, check_spec, choose_alpha, matrix_metrics,
                           pure_gate_ast, read, reference_metadata)

ROOT = Path(__file__).resolve().parents[1]


class StudyChecks(unittest.TestCase):
    def test_real_config_and_reject_unregistered_training_changes(self):
        spec = read(ROOT / "configs/e3_gate_study.json")
        check_spec(spec)
        for key, value in (("lr", .01), ("batch_size", 1024), ("auxiliary_losses", True)):
            changed = copy.deepcopy(spec)
            changed["controls"][key] = value
            with self.assertRaises(ValueError):
                check_spec(changed)
        changed = copy.deepcopy(spec)
        changed["diagnostics"]["partition"] = "test"
        with self.assertRaises(ValueError):
            check_spec(changed)

    def test_real_frozen_ast_changes_exactly_the_two_mix_statements(self):
        source = (ROOT / "src/xgnid/model.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DualBranchLogitGatedHeteroGNN")
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_graph_logits")
        function_source = ast.unparse(function)
        edited = pure_gate_ast(function_source)
        modified_source = ast.unparse(edited)
        self.assertIn("self.gate(", modified_source)
        self.assertIn("flow_conf", modified_source)
        self.assertIn("self.shared_head(fused_logits)", modified_source)
        self.assertNotIn("0.9 * gate_weights", modified_source)
        self.assertNotIn("gate_weights.sum(", modified_source)
        # Reinsert exactly the removed statements at the original positions;
        # the resulting AST must be identical to the entire original function.
        original_else = next(n for n in ast.walk(ast.parse(function_source)) if isinstance(n, ast.If)
                             and any(isinstance(x, ast.Assign) and "0.9 * gate_weights" in ast.unparse(x) for x in n.orelse))
        edited_if = next(n for n in ast.walk(edited) if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(original_else.test))
        edited_if.orelse = copy.deepcopy(original_else.orelse)
        self.assertEqual(ast.dump(edited.body[0], include_attributes=False), ast.dump(ast.parse(function_source).body[0], include_attributes=False))
        with self.assertRaises(ValueError):
            pure_gate_ast(modified_source)

    def test_alpha_ties_follow_registered_rule(self):
        candidates = {i: {"macro_f1": .9} for i in range(21)}
        self.assertEqual(choose_alpha(candidates), 10)
        candidates[9]["macro_f1"] = candidates[11]["macro_f1"] = .91
        self.assertEqual(choose_alpha(candidates), 9)
        candidates[0]["macro_f1"] = .92
        self.assertEqual(choose_alpha(candidates), 0)
        with self.assertRaises(ValueError):
            choose_alpha({0: {"macro_f1": .99}})

    def test_diagnostic_gate_rejects_wrong_partition_reference_or_failed_replay(self):
        reference = {"identity": "guard-test-only"}
        report = {"status": "completed", "exit_code": 0, "mode": "diagnose", "partition": "val",
                  "reference": reference, "test_graphs_loaded": 0, "test_evaluated": False,
                  "models": {name: {"saved_validation_reproduced": True, "model_state_unchanged": True, "samples": 13446}
                             for name in ("E2", "E3")}}
        check_diagnostic(report, reference)
        for key, value in (("partition", "test"), ("test_graphs_loaded", 1), ("test_evaluated", True),
                           ("reference", {}), ("status", "failed")):
            changed = copy.deepcopy(report)
            changed[key] = value
            with self.assertRaises(ValueError):
                check_diagnostic(changed, reference)
        changed = copy.deepcopy(report)
        changed["models"]["E3"]["saved_validation_reproduced"] = False
        with self.assertRaises(ValueError):
            check_diagnostic(changed, reference)

    def test_actual_local_reference_bindings_and_checkpoint_file_bytes(self):
        # Metadata, text and SHA256 only; never deserialize the .pt files.
        spec = read(ROOT / "configs/e3_gate_study.json")
        if not (ROOT / spec["references"]["E3"] / "summary.json").is_file():
            self.skipTest("Local server artifact copies not present")
        result = reference_metadata(ROOT, spec)
        self.assertEqual(set(result["references"]), {"E2", "E3"})
        self.assertEqual(len(result["references"]["E3"]["sources"]), 9)

    def test_metrics_reconstruction_from_real_validation_matrices(self):
        for name, directory in (("E2", "e2_batch_full_b512_s42_shared_v1"), ("E3", "e3_batch_full_b512_s42_shared_v3")):
            path = ROOT / "outputs" / directory / name / "metrics.json"
            if not path.is_file():
                self.skipTest("Local server artifact copies not present")
            saved = read(path)["best_validation"]
            rebuilt = matrix_metrics(saved["confusion_matrix"])
            self.assertEqual(rebuilt["samples"], 13446)
            for key in ("accuracy", "macro_f1", "weighted_f1"):
                self.assertAlmostEqual(saved[key], rebuilt[key], places=12)

    def test_WebBased_focus_uses_real_validation_confusion_and_class_head_metrics(self):
        from diagnose_e3_gate_server import webbased_focus
        labels = ["Benign", "WebBased", "Spoofing", "Recon", "Mirai", "Dos", "DDos", "BruteForce"]
        for name, directory in (("E2", "e2_batch_full_b512_s42_shared_v1"), ("E3", "e3_batch_full_b512_s42_shared_v3")):
            path = ROOT / "outputs" / directory / name / "metrics.json"
            if not path.is_file():
                self.skipTest("Local server artifact copies not present")
            saved = read(path)["best_validation"]
            result = webbased_focus(matrix_metrics(saved["confusion_matrix"]), labels)
            self.assertEqual(result["partition"], "val")
            self.assertEqual(result["support"], 544)
            self.assertAlmostEqual(result["precision"], saved["per_class"]["WebBased"]["precision"], places=12)
            self.assertAlmostEqual(result["recall"], saved["per_class"]["WebBased"]["recall"], places=12)
            self.assertAlmostEqual(result["f1"], saved["per_class"]["WebBased"]["f1-score"], places=12)

    def test_entry_sources_preserve_no_test_loaders_and_linux_model_boundary(self):
        for name in ("diagnose_e3_gate_server.py", "run_e3_gate_ablation_server.py"):
            tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"), feature_version=(3, 9))
            self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) and
                                 (getattr(n, "module", "") or "").startswith("torch") for n in tree.body))
            source = ast.unparse(tree)
            self.assertIn("os.name == 'posix'", source)
            self.assertNotIn("core.select_rows(frozen, 'test')", source)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
