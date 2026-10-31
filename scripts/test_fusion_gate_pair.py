"""No-torch tests of the paired-control contracts and report integrity."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_fusion_gate_pair as pair
from summarize_fusion_gate_pair import analyze, prediction_transitions, summarize

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Benign', 'WebBased', 'Spoofing', 'Recon', 'Mirai', 'Dos', 'DDos', 'BruteForce']


class Checks(unittest.TestCase):
    def config(self):
        return pair.read(ROOT / 'configs/fusion_gate_pair_s42_v1.json')

    def test_scope_uses_new_head_protocol_not_joint_learning_rates(self):
        self.assertNotIn('torch', sys.modules)
        c = self.config()
        pair.check_config(c)
        self.assertEqual((c['epochs'], c['patience'], c['include_epoch_zero'], c['lr']), (100, 20, False, .001))
        for key, value in [('seed', 43), ('width', 64), ('epochs', 30), ('lr', 1e-5),
                           ('include_epoch_zero', True), ('statistics_residual', True), ('automatic_followup', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                pair.check_config({**c, key: value})
        old = pair.read(ROOT / 'configs/joint_gate_lrscale01_20261030.json')
        self.assertEqual((old['epochs'], old['patience']), (30, 12))

    def test_architecture_and_adapter_do_not_change_historical_implementation(self):
        equal, gate = pair.architecture('equal'), pair.architecture('dynamic')
        self.assertEqual((equal['trainable_parameters'], gate['trainable_parameters']), (5480, 7144))
        self.assertEqual(equal['unused_frozen_gate_parameters'], 1664)
        self.assertEqual(gate['gate'], [69, 16, 32])
        self.assertEqual(equal['shared_head'], [32, 32, 8])
        original = pair.parent.train.__globals__['architecture']
        adapted = pair.adapted(pair.parent.train, 'equal')
        self.assertEqual(adapted.__globals__['architecture'](32), equal)
        self.assertIs(adapted.__globals__['optimizer'], pair.optimizer)
        self.assertIs(pair.parent.train.__globals__['architecture'], original)
        self.assertEqual(original(32)['trainable_parameters'], 7144)
        with self.assertRaises(ValueError):
            pair.architecture('dynamic', 64)

    def test_optimizer_excludes_unused_gate_and_frozen_encoders_without_omissions(self):
        class P:
            def __init__(self, active, count):
                self.requires_grad, self.count = active, count
            def numel(self):
                return self.count
        encoder, public, gate = P(False, 99), P(True, 5480), P(False, 1664)
        fusion_params = [public, gate]
        graph = SimpleNamespace(flow_encoder=SimpleNamespace(parameters=lambda: iter([encoder])),
            packet_encoder=SimpleNamespace(parameters=lambda: iter([])),
            fusion=SimpleNamespace(parameters=lambda: iter(fusion_params), fusion_mode='equal'),
            parameters=lambda: iter([encoder, public, gate]))
        core = SimpleNamespace(torch=SimpleNamespace(optim=SimpleNamespace(Adam=lambda ps, **kw: (list(ps), kw))))
        params, kw = pair.optimizer(core, graph, self.config())
        self.assertEqual(params, [public])
        self.assertEqual(kw, {'lr': .001, 'weight_decay': 0.})
        fusion_params.append(public)
        with self.assertRaises(ValueError):
            pair.optimizer(core, graph, self.config())
        fusion_params.pop()
        gate.requires_grad = True
        with self.assertRaises(ValueError):
            pair.optimizer(core, graph, self.config())
        graph.fusion.fusion_mode = 'dynamic'
        self.assertEqual(pair.optimizer(core, graph, self.config())[0], [public, gate])
        encoder.requires_grad = True
        with self.assertRaises(ValueError):
            pair.optimizer(core, graph, self.config())

    def test_transition_destinations_distinguish_repairs_regressions_and_wrong_switches(self):
        equal = [{'sample_id': str(i), 'label': y, 'prediction': p} for i, (y, p) in enumerate([(1, 3), (2, 2), (1, 0), (7, 7)])]
        gate = [{**r, 'prediction': p} for r, p in zip(equal, [1, 0, 3, 7])]
        t = prediction_transitions(equal, gate, LABELS)
        by_kind = {r['kind']: r for r in t}
        self.assertEqual(by_kind['repaired']['true_class'], 'WebBased')
        self.assertEqual(by_kind['repaired']['equal_prediction'], 'Recon')
        self.assertEqual(by_kind['regressed']['true_class'], 'Spoofing')
        self.assertEqual(by_kind['both_wrong_changed']['gate_prediction'], 'Recon')
        with self.assertRaises(ValueError):
            prediction_transitions(equal, gate[::-1], LABELS)

    def fixture(self):
        cm = pair.read(ROOT / 'configs/n0_seed42_reference_20261029.json')['model']['best_validation']['confusion_matrix']
        base = pair.parent.frozen.matrix_report(cm, LABELS)
        wider_cm = copy.deepcopy(cm)
        wider_cm[1][3] -= 1
        wider_cm[1][1] += 1
        better = pair.parent.frozen.matrix_report(wider_cm, LABELS)
        def arm(mode, validation):
            return {'architecture': pair.architecture(mode), 'epoch_zero': base,
                'best_validation': validation, 'last_validation': base,
                'best_epoch': 1, 'epochs': 21, 'optimizer_updates': 4431, 'orders': ['same'] * 21,
                'training_start_rng_sha256': 'same_rng', 'training_wall_seconds': 1.,
                'training_evidence': {}, 'epoch_zero_eligible': False, 'selected_epoch_zero': False}
        return {'status': 'completed', 'exit_code': 0, 'test_evaluated': False, 'automatic_followup': False,
            'tensor_preflight_passed': True, 'protocol': self.config(),
            'models': {'D32-Equal': arm('equal', base), 'D32-Gate': arm('dynamic', better)},
            'historical_D32': arm('dynamic', better),
            'Gate_versus_Equal': {'counts': {'repaired': 1, 'regressed': 0}, 'net_correct': 1},
            'Gate_versus_historical_D32': {'counts': {'repaired': 0, 'regressed': 0}, 'net_correct': 0},
            'prediction_transitions': [{'kind': 'repaired', 'true_class': 'WebBased', 'equal_prediction': 'Recon',
                                        'gate_prediction': 'WebBased', 'count': 1}],
            'common_epochs_order_checked': 21, 'initialization': {'public_initialization_identical': True},
            'graph_replays': {'D32-Equal': {'samples': 13446}, 'D32-Gate': {'samples': 13446}},
            'encoder_parameters_and_BN_unchanged': True, 'source_artifacts_unchanged': True,
            'shared_cache_unchanged': True, 'shared_cache_reuse_wall_seconds': .1,
            'preflight_including_cache_wall_seconds': 2., 'discarded_probe_updates': 6,
            'execution_comparison': {'common_protocol_differences': {}}}

    def test_report_checks_metrics_orders_transitions_and_historical_rerun(self):
        r = self.fixture()
        a = analyze(r)
        self.assertGreater(a['Gate_minus_Equal_pp'], 0.)
        self.assertFalse(a['budget_screen_passed'])
        self.assertFalse(a['historical_rerun_review_required'])
        for damage in ('order', 'transition', 'metrics'):
            bad = copy.deepcopy(r)
            if damage == 'order':
                bad['models']['D32-Gate']['orders'][0] = 'different'
            elif damage == 'transition':
                bad['prediction_transitions'][0]['count'] = 2
            else:
                bad['models']['D32-Gate']['best_validation']['macro_f1'] += .001
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                analyze(bad)
        changed = copy.deepcopy(r)
        perfect = pair.parent.frozen.matrix_report([[sum(row) if i == j else 0 for j in range(8)]
                                                  for i, row in enumerate(r['historical_D32']['best_validation']['confusion_matrix'])], LABELS)
        changed['historical_D32']['best_validation'] = perfect
        self.assertTrue(analyze(changed)['gate_contribution_interpretation_pending_execution_review'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair.write(root / 'summary.json', r)
            text = summarize(root)
            self.assertIn('Gate-Equal:', text)
            self.assertIn('32', text)
            self.assertIn('last_validation confusion matrix', text)
            self.assertIn('1,664 unused frozen gate parameters', text)
            self.assertTrue((root / 'analysis.json').exists())
            with (root / 'per_class.csv').open(newline='') as stream:
                self.assertEqual(len(list(csv.reader(stream))), 33)
            with (root / 'prediction_transitions.csv').open(newline='') as stream:
                self.assertEqual(len(list(csv.reader(stream))), 2)

    def source_fixture(self, root):
        """Integrity fixture, deliberately not a tensor/checkpoint execution test."""
        c = self.config()
        (root / 'configs').mkdir()
        original = ROOT / 'configs' / c['source_protocol']
        (root / 'configs' / c['source_protocol']).write_bytes(original.read_bytes())
        source = root / c['source_directory']
        (source / 'D32').mkdir(parents=True)
        fake = self.fixture()
        m = fake['historical_D32']
        m['architecture'] = pair.parent.architecture(32)
        m['checkpoint_sha256'] = {}
        for filename in ('best.pt', 'last.pt', 'epoch_zero.pt'):
            (source / 'D32' / filename).write_bytes(b'integrity fixture ' + filename.encode())
            m['checkpoint_sha256'][filename] = pair.sha(source / 'D32' / filename)
        (source / 'D32_preflight.pt').write_bytes(b'clean fixture')
        (source / 'D32/validation_predictions.jsonl.gz').write_bytes(b'prediction fixture')
        c['source_D32_best_sha256'] = m['checkpoint_sha256']['best.pt']
        c['source_D32_macro_f1'] = m['best_validation']['macro_f1']
        encoder = {'encoder_sha256': 'same_encoder'}
        receipts = {}
        for part in ('train', 'val'):
            path = source / (part + '_encoder64.pt')
            path.write_bytes(b'cache fixture ' + part.encode())
            c['source_cache_sha256'][part] = pair.sha(path)
            receipts[part] = {'sha256': pair.sha(path), 'metadata': {'source': encoder, 'embedding_dim': 64}}
        receipts['val']['N0_replayed_validation'] = m['best_validation']
        base_c = pair.read(original)
        b = {'base_identity': 'validated', 'encoder_source': encoder, 'caches': receipts, 'cache_tensor_sha256': {}}
        s = {**fake, 'binding': b, 'protocol': base_c, 'models': {'D32': m},
             'frozen_encoder_unchanged': True}
        init = {'streams': pair.parent.initialization_streams(42), 'encoder_sha256': 'same_encoder', 'all_heads_fresh': True}
        rounds = {'D32': {'strict_reload': True, 'checkpoint_sha256': pair.sha(source / 'D32_preflight.pt')}}
        for filename, obj in [('status.json', {'status': 'completed', 'exit_code': 0}), ('summary.json', s),
                ('binding.json', b), ('config.json', base_c), ('gpu.json', {'physical_index': 1}), ('initialization.json', init),
                ('preflight_roundtrips.json', rounds), ('cache_receipts.json', receipts), ('D32/metrics.json', m)]:
            pair.write(source / filename, obj)
        history = [{'epoch': n, 'val_macro_f1': m['best_validation']['macro_f1'], 'optimizer_updates': 211,
                    'best_epoch': 1, 'stale_epochs': n-1, 'train_order_sha256': 'same'} for n in range(1, 22)]
        (source / 'D32/epochs.jsonl').write_text('\n'.join(json.dumps(x) for x in history), encoding='utf-8')
        cfg = root / 'configs/new_pair.json'
        pair.write(cfg, c)
        base = (base_c, {}, {'model': {'best_validation': m['best_validation']}}, root, {}, {'base_identity': 'validated'})
        return SimpleNamespace(repo=root, config=cfg), base, source

    def test_metadata_preserves_source_and_rejects_changed_cache_or_start(self):
        for damage in (None, 'cache', 'clean', 'best', 'binding'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                args, base, source = self.source_fixture(Path(directory))
                if damage in ('cache', 'clean', 'best'):
                    name = {'cache': 'train_encoder64.pt', 'clean': 'D32_preflight.pt', 'best': 'D32/best.pt'}[damage]
                    (source / name).write_bytes(b'changed')
                elif damage == 'binding':
                    b = pair.read(source / 'binding.json')
                    pair.write(source / 'binding.json', {**b, 'unknown_change': True})
                with patch.object(pair.parent, 'metadata', return_value=base):
                    if damage:
                        with self.assertRaises(ValueError):
                            pair.metadata(args)
                    else:
                        meta = pair.metadata(args)
                        pair.assert_source_unchanged(meta[-1])
                        (source / 'D32/epoch_zero.pt').write_bytes(b'changed after metadata')
                        with self.assertRaises(ValueError):
                            pair.assert_source_unchanged(meta[-1])


if __name__ == '__main__':
    unittest.main()
