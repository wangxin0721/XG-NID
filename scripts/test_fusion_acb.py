"""Standard-library provenance/control checks, never tensor/model execution."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest

from fusion_acb import VARIANTS, check_config, validate_probe
from e3_gate_study import pure_gate_ast
import run_fusion_acb_server

ROOT = Path(__file__).resolve().parents[1]


class AcbChecks(unittest.TestCase):
    def test_import_and_py39_syntax(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('torch_geometric', sys.modules)
        for file in ('fusion_acb.py', 'run_fusion_acb_server.py'):
            ast.parse((ROOT/'scripts'/file).read_text(encoding='utf-8'), feature_version=(3, 9))

    def test_frozen_design_rejects_coordinated_changes(self):
        config = json.loads((ROOT/'configs/fusion_acb_execution_20261026.json').read_text(encoding='utf-8'))
        design = json.loads((ROOT/'configs/fusion_acb_study_20261023.json').read_text(encoding='utf-8'))
        check_config(config, design)
        for key, value in (('lr', .01), ('batch_size', 1024), ('patience', 20)):
            changed = copy.deepcopy(design)
            changed['controls'][key] = value
            execution = copy.deepcopy(config)
            execution['design'] = changed
            with self.assertRaises(ValueError):
                check_config(execution, changed)
        for key, value in (('seed', 43), ('test_evaluated', True), ('probe_updates', 1)):
            execution = copy.deepcopy(config)
            execution[key] = value
            with self.assertRaises(ValueError):
                check_config(execution, design)

    def valid_probe(self):
        status = {'status': 'completed', 'exit_code': 0, 'mode': 'probe'}
        binding = {'source': 'same', 'seed': 42}
        variants = {v: {'initialization': {'common_initial_state_sha256': 'common',
            'training_start_rng_sha256': 'rng'}, 'optimizer_update_verified': True,
            'auxiliary_gradient_paths_verified': True, 'fusion_parameter_update_verified': v != 'A_equal',
            'updates': [{}, {}, {}]} for v in VARIANTS}
        return {**status, 'binding': binding, 'test_evaluated': False, 'variants': variants}, status, binding

    def test_probe_rejects_unpaired_states_rng_and_missing_updates(self):
        report, status, binding = self.valid_probe()
        validate_probe(report, status, binding)
        for field in ('common_initial_state_sha256', 'training_start_rng_sha256'):
            changed = copy.deepcopy(report)
            changed['variants']['C_global']['initialization'][field] = 'different'
            with self.assertRaises(ValueError):
                validate_probe(changed, status, binding)
        for field in ('optimizer_update_verified', 'auxiliary_gradient_paths_verified', 'fusion_parameter_update_verified'):
            changed = copy.deepcopy(report)
            changed['variants']['B_dynamic'][field] = False
            with self.assertRaises(ValueError):
                validate_probe(changed, status, binding)
        changed = copy.deepcopy(report)
        del changed['variants']['C_global']
        with self.assertRaises(ValueError):
            validate_probe(changed, status, binding)

    def test_probe_rejects_failed_status_stale_binding_or_test(self):
        report, status, binding = self.valid_probe()
        with self.assertRaises(ValueError):
            validate_probe(report, {**status, 'exit_code': 1}, binding)
        with self.assertRaises(ValueError):
            validate_probe(report, status, {'source': 'changed'})
        with self.assertRaises(ValueError):
            validate_probe({**report, 'test_evaluated': True}, status, binding)

    def test_loss_ast_supervises_shared_head_not_raw_z(self):
        tree = ast.parse((ROOT/'scripts/fusion_acb.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'losses')
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        ce = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'cross_entropy']
        self.assertEqual(len(ce), 2)
        self.assertEqual({ast.unparse(n.args[0]) for n in ce},
            {"model.shared_head(cache['flow_logits'])", "model.shared_head(cache['packet_logits'])"})
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and n.func.attr in ('detach', 'loss') for n in calls))
        total = next(n.value for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'total')
        # Evaluate only the scalar arithmetic AST, not a model or ML mock.
        value = eval(compile(ast.Expression(total), '<loss arithmetic>', 'eval'),
                     {'__builtins__': {}}, {'main': 2., 'flow': 4., 'packet': 6.})
        self.assertEqual(value, 2.25)

    def test_existing_forward_has_nonlinear_shared_head_and_no_proxy_after_transform(self):
        tree = ast.parse((ROOT/'src/xgnid/model.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DualBranchLogitGatedHeteroGNN')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_graph_logits')
        transformed = ast.unparse(pure_gate_ast(ast.unparse(fn)))
        self.assertIn('self.shared_head(fused_logits)', transformed)
        self.assertIn('elif not self.use_gate:', transformed)
        self.assertNotIn('0.9 * gate_weights', transformed)

    def test_queue_uses_subprocess_errors_and_no_test_loader(self):
        tree = ast.parse((ROOT/'scripts/run_fusion_acb_server.py').read_text(encoding='utf-8'))
        queue = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'queue')
        calls = [n for n in ast.walk(queue) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'subprocess.run']
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(k.arg == 'check' and isinstance(k.value, ast.Constant) and k.value.value is True for k in calls[0].keywords))
        source = ast.unparse(tree)
        self.assertNotIn("select_rows(frozen, 'test')", source)
        self.assertNotIn('screening_round_gpu_hour_ceiling', source)


if __name__ == '__main__':
    unittest.main()
