"""No-torch config, stability-gate, namespace, grouping and report checks."""
import copy
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch

import run_flow_stats_layernorm as ln
from summarize_flow_stats_layernorm import summarize

ROOT = Path(__file__).resolve().parents[1]


class Checks(unittest.TestCase):
    def config(self):
        c = ln.read(ROOT / 'configs/flow_stats_layernorm_s42_v1.json')
        base = ln.read(ROOT / 'configs' / c['parent_config'])
        return c, base

    def test_only_layernorm_changes_registered_model_and_training(self):
        self.assertNotIn('torch', sys.modules)
        c, base = self.config()
        ln.check_config(c, base)
        self.assertEqual(ln.sha(ROOT / 'configs' / c['parent_config']), c['parent_config_sha256'])
        for key, value in [('encoder_mode', 'eval'), ('stats_lr', 1e-5), ('head_lr', 1e-4),
                           ('seed', 43), ('epochs', 12), ('start', 'failed_last_checkpoint'),
                           ('test_evaluated', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                altered = copy.deepcopy(c)
                altered[key] = value
                ln.check_config(altered, base)
        altered = copy.deepcopy(c)
        altered['probe']['ratio_max_ceiling'] = 10.
        with self.assertRaises(ValueError):
            ln.check_config(altered, base)

    def test_stability_gate_blocks_old_scale_and_nonfinite_or_functional_damage(self):
        safe = {'ratio_after': {'median': .01, 'p95': .02, 'max': .03},
                'eval_nll_increase': .001, 'prediction_change_fraction': 0.}
        self.assertEqual(ln.stability_violations(safe, ln.PROBE), [])
        for measure, value in [('median', 5.37535), ('p95', 11.3605), ('max', 49.7222), ('max', float('nan'))]:
            changed = copy.deepcopy(safe)
            changed['ratio_after'][measure] = value
            self.assertTrue(ln.stability_violations(changed, ln.PROBE))
        for name, value in [('eval_nll_increase', .101), ('prediction_change_fraction', .051)]:
            changed = copy.deepcopy(safe)
            changed[name] = value
            self.assertTrue(ln.stability_violations(changed, ln.PROBE))
        at_bound = {'ratio_after': {'median': .1, 'p95': .25, 'max': 1.},
                    'eval_nll_increase': .1, 'prediction_change_fraction': .05}
        self.assertEqual(ln.stability_violations(at_bound, ln.PROBE), [])

    def test_optimizer_includes_layernorm_and_new_last_index_without_duplicate_parameters(self):
        class P:
            requires_grad = False
            def requires_grad_(self, value):
                self.requires_grad = value
        named = [(n, P()) for n in ('flow_encoder.x', 'packet_encoder.x', 'shared_head.x',
            'joint_gate.x', 'flow_stats_mlp.0.weight', 'flow_stats_mlp.1.weight',
            'flow_stats_mlp.1.bias', 'flow_stats_mlp.3.weight', 'flow_stats_mlp.3.bias', 'gate.old')]
        model = SimpleNamespace(named_parameters=lambda **kw: iter(named),
                                parameters=lambda: (p for _, p in named))
        c, _ = self.config()
        groups = ln.groups(model, ln.VARIANT, c)
        self.assertEqual([len(g['params']) for g in groups], [2, 1, 1, 5])
        self.assertTrue(dict(named)['flow_stats_mlp.1.weight'].requires_grad)
        self.assertFalse(dict(named)['gate.old'].requires_grad)
        self.assertEqual(ln.parent.VARIANT, 'J2_flow_stats')

    def test_isolated_runtime_preserves_parent_and_roundtrip_uses_new_builder(self):
        entry = ln.runtime_entry(Path('unused'))
        ns = entry.__globals__
        self.assertEqual(ns['VARIANT'], ln.VARIANT)
        self.assertIs(ns['build'], ln.build)
        self.assertIs(ns['roundtrip_probe'].__globals__, ns)
        self.assertIs(ns['joint'].build, ln.parent.joint.build)
        self.assertIs(ns['joint'].train, ln.monitored_train)
        self.assertIsNot(ln.parent.run.__globals__, ns)
        self.assertIs(ln.parent.run.__globals__['build'], ln.parent.build)

    def test_monitored_trainer_forwards_reconstruction_and_metrics_without_extra_forwards(self):
        class Value:
            def detach(self): return self
            def cpu(self): return self
        class Model:
            _flow_stats_diagnostics = {k: Value() for k in ('flow_weights', 'flow_graph_norm',
                'flow_residual_norm', 'residual_to_graph_norm')}
            hook = None
            def register_forward_hook(self, function):
                self.hook = function
                return SimpleNamespace(remove=lambda: setattr(self, 'hook', None))
        class Core:
            calls = 0
            def evaluate(self, model, *args, **kwargs):
                self.calls += 1
                model.hook(model, None, None)
                return {'macro_f1': .97}, []
        def surrogate(core, args, c, data, device, m, variant, binding):
            # These globals must be the callbacks injected by parent.run.
            built, grouped = build(), groups()
            for exported in (False, False, True):
                measured, _ = core.evaluate(m, None, None, export_rows=exported)
            return built, grouped, measured
        with tempfile.TemporaryDirectory() as directory:
            core, model = Core(), Model()
            ns = dict(ln.monitored_train.__globals__, build=lambda: 'new-architecture', groups=lambda: 'new-groups')
            adapted = FunctionType(ln.monitored_train.__code__, ns)
            with patch.object(ln.parent.joint, 'train', surrogate), \
                 patch.object(ln.parent, 'tensor_summary', return_value={'mean': 1.}):
                result = adapted(core, SimpleNamespace(output=Path(directory)), {}, None, None, model, ln.VARIANT, {})
            self.assertEqual(result[:2], ('new-architecture', 'new-groups'))
            self.assertIn('flow_stats_diagnostics', result[2])
            self.assertEqual(core.calls, 3)
            self.assertIsNone(model.hook)
            history = [json.loads(s) for s in (Path(directory) / 'epoch_scale_diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual([x['epoch'] for x in history], [0, 1, None])

    def test_report_preserves_threshold_and_names_independent_candidate(self):
        c, _ = self.config()
        original = ln.read(ROOT / 'configs/n0_seed42_reference_20261029.json')['model']
        m = {**original, 'selected_epoch_zero': False, 'epochs': 17, 'training_evidence': {},
             'last_validation': {'flow_stats_diagnostics': {'residual_to_graph_norm': {'mean': .05}}}}
        count = {'counts': {'repaired': 0, 'regressed': 0}, 'net_correct': 0}
        r = {'status': 'completed', 'test_evaluated': False, 'protocol': c, 'N0': m, 'frozen_V2': m,
             'reference_joint': {'J0': m, 'J2': m}, 'candidate': m, 'stats_off': m['best_validation'],
             'stats_on_minus_off_pp': 0., 'comparisons': {'lower_lr_J2': count}, 'stage_A': {'elapsed_seconds': None},
             'preflight_wall_seconds': .1, 'post_training_diagnostics_wall_seconds': .1, 'attribution': 'Candidate screen.'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ln.write(root / 'summary.json', r)
            ln.write(root / 'probe.json', {'status': 'passed', 'bounds': ln.PROBE, 'updates': [{}, {}, {}]})
            for _ in range(2):
                text = summarize(root)
            self.assertIn('| J2_flow_stats_ln |', text)
            a = ln.read(root / 'analysis.json')
            self.assertEqual(a['rows'][-1]['model'], ln.VARIANT)
            self.assertEqual(a['last_validation_scale']['residual_to_graph_norm']['mean'], .05)
            self.assertEqual(a['budget_screening']['threshold_pp'], .1)
            with (root / 'per_class.csv').open(newline='') as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows), 49)
            self.assertEqual(sum(row[0] == ln.VARIANT for row in rows), 8)

    def test_rejected_probe_exits_with_zero_formal_updates_and_no_success_report(self):
        c, _ = self.config()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new_diagnostic_output'
            args = SimpleNamespace(output=output)
            meta = (c, None, None, None, None, None, None, {'registered': True})
            def reject(*args):
                raise ln.ProbeRejected('fixed batch exceeded registered bound')
            with patch.object(ln, 'runtime_entry', return_value=reject):
                with self.assertRaises(SystemExit) as error, redirect_stdout(io.StringIO()):
                    ln.execute(args, meta)
            self.assertEqual(error.exception.code, 2)
            status = ln.read(output / 'status.json')
            self.assertEqual(status['status'], 'preflight_rejected')
            self.assertEqual(status['formal_training_updates'], 0)
            self.assertFalse(status['formal_training_started'])
            self.assertFalse((output / 'summary.json').exists())
            self.assertFalse((output / ln.VARIANT).exists())


if __name__ == '__main__':
    unittest.main()
