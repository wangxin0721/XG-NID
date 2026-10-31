"""No-torch checks of scope, cache identity, selection and report contracts."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_fusion_dimension_pair as pair
from summarize_fusion_dimension_pair import analyze, summarize

ROOT = Path(__file__).resolve().parents[1]


class Checks(unittest.TestCase):
    def config(self):
        return pair.read(ROOT / 'configs/fusion_dimension_pair_s42_v1.json')

    def test_scope_budget_epoch_zero_override_are_only_for_this_pair(self):
        self.assertNotIn('torch', sys.modules)
        c = self.config()
        pair.check_config(c)
        self.assertEqual((c['epochs'], c['patience'], c['include_epoch_zero']), (100, 20, False))
        old = pair.read(ROOT / 'configs/frozen_fusion_20261029.json')
        joint = pair.read(ROOT / 'configs/joint_gate_lrscale01_20261030.json')
        self.assertEqual((old['epochs'], old['patience'], old['include_epoch_zero']), (30, 12, True))
        self.assertEqual((joint['epochs'], joint['patience']), (30, 12))
        for key, value in [('epochs', 30), ('patience', 12), ('include_epoch_zero', True), ('seed', 43),
                           ('encoder_mode', 'train'), ('statistics_residual', True), ('test_evaluated', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                changed = copy.deepcopy(c)
                changed[key] = value
                pair.check_config(changed)

    def test_named_initialization_streams_are_fixed_and_dimensions_are_explicit(self):
        self.assertEqual(pair.initialization_streams(42), pair.initialization_streams(42))
        streams = pair.initialization_streams(42)
        self.assertEqual(len(set(streams.values())), 6)
        self.assertTrue(all(0 <= x < 2**63 for x in streams.values()))
        with self.assertRaises(ValueError):
            pair.initialization_streams(44)
        self.assertEqual(pair.architecture(8)['gate'], [21, 16, 8])
        self.assertEqual(pair.architecture(32)['gate'], [69, 16, 32])
        self.assertEqual(pair.architecture(8)['trainable_parameters'], 2080)
        self.assertEqual(pair.architecture(32)['trainable_parameters'], 7144)
        with self.assertRaises(ValueError):
            pair.architecture(64)

    def test_equal_start_uses_elementwise_error_not_exact_reduced_mean(self):
        observed = {'mean': [.5000000596046448] * 8, 'min': [.5] * 8, 'max': [.5] * 8,
                    'max_abs_deviation_from_half': 0.}
        # Regression: the old assertion rejected this valid all-half tensor.
        self.assertFalse(all(v == .5 for key in ('mean', 'min', 'max') for v in observed[key]))
        receipt = pair.validate_equal_start(observed)
        self.assertEqual(receipt['max_abs_error'], 0.)
        self.assertEqual(receipt['observed_mean'], observed['mean'])
        for error in (1e-4, float('nan'), float('inf')):
            bad = {**observed, 'mean': [.5] * 8, 'max_abs_deviation_from_half': error}
            with self.subTest(error=error), self.assertRaises(ValueError):
                pair.validate_equal_start(bad)

    def test_old_projected_cache_or_wrong_checkpoint_or_order_cannot_be_reused(self):
        source = {'encoder_sha256': 'same', 'N0_best_sha256': 'same'}
        rows = [{'sample_id': 'a'}, {'sample_id': 'b'}]
        h = pair.cache_header('train', source, rows)
        pair.validate_cache_header(h, 'train', source, rows)
        for bad in ({**h, 'embedding_dim': 8}, {**h, 'schema': 'old'},
                    {**h, 'source': {'encoder_sha256': 'other'}}, {**h, 'partition': 'val'}):
            with self.assertRaises(ValueError):
                pair.validate_cache_header(bad, 'train', source, rows)
        with self.assertRaises(ValueError):
            pair.validate_cache_header(h, 'train', source, rows[::-1])
        with self.assertRaises(ValueError):
            pair.cache_header('test', source, rows)

    def history(self, scores):
        best, selected, stale = float('-inf'), None, 0
        history = []
        for epoch, score in enumerate(scores, 1):
            best, stale, improved = pair.pair.selection_state(score, best, stale)
            if improved:
                selected = epoch
            history.append({'epoch': epoch, 'val_macro_f1': score, 'optimizer_updates': 211,
                            'best_epoch': selected, 'stale_epochs': stale})
        metrics = {'epochs': len(history), 'best_epoch': selected, 'best_validation': {'macro_f1': best},
                   'optimizer_updates': 211 * len(history), 'selected_epoch_zero': False}
        return history, metrics

    def test_first_trained_epoch_is_selected_even_below_epoch_zero_and_patience_is20(self):
        c = self.config()
        history, m = self.history([.60] * 21)
        m['epoch_zero'] = {'macro_f1': .90}  # deliberately higher; excluded by request
        pair.check_history(history, m, c)
        self.assertEqual(m['best_epoch'], 1)
        self.assertEqual(history[-1]['stale_epochs'], 20)
        short, short_m = self.history([.60] * 13)
        with self.assertRaises(ValueError):
            pair.check_history(short, short_m, c)
        changed = copy.deepcopy(m)
        changed.update(best_epoch=0, selected_epoch_zero=True)
        with self.assertRaises(ValueError):
            pair.check_history(history, changed, c)
        full, full_m = self.history([.5 + epoch / 1000 for epoch in range(1, 101)])
        pair.check_history(full, full_m, c)
        self.assertEqual(full_m['best_epoch'], 100)

    def test_optimizer_trains_all_fresh_heads_and_excludes_encoder_ids(self):
        class P:
            def __init__(self, active): self.requires_grad = active
        encoder, head_a, head_b = P(False), P(True), P(True)
        named = [head_a, head_b]
        graph = SimpleNamespace(flow_encoder=SimpleNamespace(parameters=lambda: iter([encoder])),
            packet_encoder=SimpleNamespace(parameters=lambda: iter([])),
            fusion=SimpleNamespace(parameters=lambda: iter(named)), parameters=lambda: iter([encoder, head_a, head_b]))
        core = SimpleNamespace(torch=SimpleNamespace(optim=SimpleNamespace(Adam=lambda p, **k: (list(p), k))))
        result = pair.optimizer(core, graph, self.config())
        self.assertEqual(result[0], [head_a, head_b])
        self.assertEqual(result[1]['lr'], .001)
        named.append(encoder)
        with self.assertRaises(ValueError):
            pair.optimizer(core, graph, self.config())
        named[:] = [head_a, head_a, head_b]
        with self.assertRaises(ValueError):
            pair.optimizer(core, graph, self.config())

    def fixture(self, directory):
        repo = Path(directory)
        cfgdir = repo / 'configs'
        cfgdir.mkdir()
        c = self.config()
        (cfgdir / c['historical_J2']['config']).write_bytes((ROOT / 'configs' / c['historical_J2']['config']).read_bytes())
        ref = pair.read(ROOT / 'configs/n0_seed42_reference_20261029.json')
        root = repo / c['historical_J2']['directory']
        (root / 'J2').mkdir(parents=True)
        # Source-integrity fixtures only; these are not loadable model weights.
        (root / 'J2/best.pt').write_bytes(b'integrity fixture')
        c['historical_J2']['best_sha256'] = pair.sha(root / 'J2/best.pt')
        c['historical_J2']['macro_f1'] = ref['model']['best_validation']['macro_f1']
        m = {'best_validation': ref['model']['best_validation'], 'checkpoint_sha256': c['historical_J2']['best_sha256']}
        summary = {'status': 'completed', 'test_evaluated': False, 'N0': ref['model'], 'joint': {'J2': m},
            'binding': {'joint_config_sha256': c['historical_J2']['config_sha256'],
                        'joint_source_sha256': pair.sha(ROOT / 'scripts/run_joint_gate.py')}}
        pair.write(root / 'status.json', {'status': 'completed', 'exit_code': 0})
        pair.write(root / 'summary.json', summary)
        pair.write(root / 'J2/metrics.json', m)
        config = cfgdir / 'pair.json'
        pair.write(config, c)
        args = SimpleNamespace(repo=repo, config=config)
        parent = ({'split_directory': 'data/same'}, ref, repo / ref['directory'], {'N0_validated': True})
        return args, parent, root

    def test_metadata_rejects_changed_reference_without_torch_or_unrelated_queue(self):
        for damage in (None, 'weights', 'running', 'wrong_N0'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                args, frozen_reference, root = self.fixture(directory)
                if damage == 'weights':
                    (root / 'J2/best.pt').write_bytes(b'changed')
                if damage == 'running':
                    pair.write(root / 'status.json', {'status': 'running'})
                if damage == 'wrong_N0':
                    s = pair.read(root / 'summary.json')
                    s['N0'] = {'other_seed': 44}
                    pair.write(root / 'summary.json', s)
                with patch.object(pair.frozen, 'metadata', return_value=frozen_reference) as check:
                    if damage is None:
                        meta = pair.metadata(args)
                        self.assertEqual(meta[0]['epochs'], 100)
                        self.assertEqual(check.call_count, 1)
                    else:
                        with self.assertRaises(ValueError):
                            pair.metadata(args)

    def test_report_separates_paired_gain_from_historical_score_and_epoch_zero(self):
        original = pair.read(ROOT / 'configs/n0_seed42_reference_20261029.json')['model']
        labels = ['Benign', 'WebBased', 'Spoofing', 'Recon', 'Mirai', 'Dos', 'DDos', 'BruteForce']
        base = pair.frozen.matrix_report(original['best_validation']['confusion_matrix'], labels)
        cm = copy.deepcopy(base['confusion_matrix'])
        cm[1][3] -= 1
        cm[1][1] += 1
        wider = pair.frozen.matrix_report(cm, labels)
        def arm(width, validation):
            return {'architecture': pair.architecture(width), 'epoch_zero': base, 'best_validation': validation,
                    'last_validation': validation, 'best_epoch': 1, 'epochs': 21,
                    'optimizer_updates': 4431, 'training_wall_seconds': .1, 'training_evidence': {}}
        h = {'best_validation': wider, 'best_epoch': 9, 'epochs': 21, 'updates': 4431, 'joint_wall_seconds': 1.}
        counts = {'counts': {'repaired': 1, 'regressed': 0}, 'net_correct': 1}
        r = {'status': 'completed', 'test_evaluated': False, 'protocol': self.config(),
            'models': {'D8': arm(8, base), 'D32': arm(32, wider)}, 'historical_J2': h, 'N0_source': original,
            'D32_minus_D8_pp': 100 * (wider['macro_f1'] - base['macro_f1']), 'D32_minus_historical_J2_pp': 0.,
            'D32_versus_D8': counts, 'D32_versus_N0': counts, 'shared_cache_wall_seconds': 1.,
            'common_epochs_order_checked': 21, 'frozen_encoder_unchanged': True, 'attribution': 'No isolated capacity claim.'}
        a = analyze(r)
        self.assertTrue(a['D32_above_paired_D8'])
        self.assertFalse(a['D32_above_historical_J2'])
        self.assertFalse(a['epoch_zero_eligible'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair.write(root / 'summary.json', r)
            text = summarize(root)
            self.assertIn('max 100 epochs / patience 20', text)
            self.assertIn('excluded from selection', text)
            with (root / 'per_class.csv').open(newline='') as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows), 33)
            r.update(cache_reused=True, recovery_source='/source/unchanged',
                     original_cache_extraction_wall_seconds=2.)
            pair.write(root / 'summary.json', r)
            recovered_text = summarize(root)
            self.assertIn('cache reuse/load/validate seconds', recovered_text)
            self.assertIn('New recovery probe updates: 0', recovered_text)
            self.assertTrue(pair.read(root / 'analysis.json')['cache_reused'])


if __name__ == '__main__':
    unittest.main()
