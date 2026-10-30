"""CPU-only provenance, selection, grouping and analysis checks; no tensor claims."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import run_flow_stats_residual as screen
from summarize_flow_stats_residual import analyze, summarize

ROOT = Path(__file__).resolve().parents[1]


class Parameter:
    def __init__(self):
        self.requires_grad = False

    def requires_grad_(self, value):
        self.requires_grad = value


class Checks(unittest.TestCase):
    def config(self):
        c = screen.read(ROOT / 'configs/flow_stats_residual_s42_v1.json')
        ref = screen.read(ROOT / 'configs' / c['reference_config'])
        return c, ref

    def test_exact_historical_control_and_scope(self):
        self.assertNotIn('torch', sys.modules)
        c, ref = self.config()
        screen.check_config(c, ref)
        self.assertEqual(screen.sha(ROOT / 'configs' / c['reference_config']), c['reference_config_sha256'])
        self.assertEqual(screen.sha(ROOT / 'scripts/run_joint_gate.py'), c['reference_source_sha256'])
        for key, value in [('encoder_mode', 'eval'), ('encoder_lr', 1e-5), ('training_jobs', 2),
                           ('start', 'joint_best'), ('test_evaluated', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                changed = copy.deepcopy(c)
                changed[key] = value
                screen.check_config(changed, ref)
        changed = copy.deepcopy(c)
        changed['stats_path']['normalization'] = 'LayerNorm'
        with self.assertRaises(ValueError):
            screen.check_config(changed, ref)

    def test_optimizer_has_stats_and_gate_and_deduplicates_shared_encoder(self):
        c, _ = self.config()
        a, b, gate, stats, old = [Parameter() for _ in range(5)]
        named = [('flow_encoder.shared', a), ('packet_encoder.shared', a),
                 ('flow_classifier.weight', b), ('shared_head.shared', b),
                 ('joint_gate.0.weight', gate), ('flow_stats_mlp.0.weight', stats),
                 ('gate.net.0.weight', old)]
        model = SimpleNamespace(named_parameters=lambda **kw: iter(named),
                                parameters=lambda: iter([a, b, gate, stats, old]))
        groups = screen.groups(model, screen.VARIANT, c)
        self.assertEqual([(g['name'], g['lr'], len(g['params'])) for g in groups],
                         [('encoder', 1e-6, 1), ('head', 1e-5, 1), ('gate', 1e-5, 1), ('stats', 1e-4, 1)])
        self.assertFalse(old.requires_grad)
        named.append(('flow_stats_mlp.conflict', a))
        with self.assertRaises(ValueError):
            screen.groups(model, screen.VARIANT, c)

    def fixture(self, root):
        c, ref = self.config()
        root = Path(root)
        m0 = screen.read(ROOT / 'configs/n0_seed42_reference_20261029.json')['model']['best_validation']
        folder = root / c['reference_directory']
        folder.mkdir(parents=True)
        screen.write(folder / 'status.json', {'status': 'completed', 'exit_code': 0})
        receipt = {'best_sha256': ref['stage_A_expected_best_sha256'], 'reused': True}
        parent = {'same_parent': True}
        refs = {'42': {'metrics': {'best_validation': m0}}}
        report = {'metrics': {'best_validation': m0}}
        summary = {'status': 'completed', 'test_evaluated': False,
                   'binding': {'joint_config_sha256': c['reference_config_sha256'],
                               'joint_source_sha256': c['reference_source_sha256'],
                               'parent': parent, 'stage_A_best_sha256': receipt['best_sha256'],
                               'stage_A_receipt': receipt},
                   'N0': refs['42']['metrics'], 'frozen_V2': report['metrics'], 'joint': {}}
        for name in ('J0', 'J2'):
            p = folder / name
            p.mkdir()
            # These are file-integrity fixtures, NOT loadable tensor checkpoints.
            (p / 'best.pt').write_bytes(b'unit test integrity fixture')
            (p / 'validation_predictions.jsonl.gz').write_bytes(b'unit test prediction fixture')
            history = []
            for epoch in range(1, 22):
                history.append({'epoch': epoch, 'optimizer_updates': 211,
                    'val_macro_f1': m0['macro_f1'] if epoch >= 9 else .95,
                    'best_epoch': 9 if epoch >= 9 else 0,
                    'stale_epochs': epoch - 9 if epoch >= 9 else epoch,
                    'train_order_sha256': str(epoch)})
            (p / 'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in history), encoding='utf-8')
            m = {'epoch_zero': {**m0, 'macro_f1': .96}, 'best_validation': m0,
                 'last_validation': m0, 'epochs': 21, 'updates': 4431, 'best_epoch': 9,
                 'orders': [r['train_order_sha256'] for r in history], 'selected_epoch_zero': False,
                 'training_evidence': screen.joint.trained_evidence(.96, history),
                 'checkpoint_sha256': screen.sha(p / 'best.pt')}
            screen.write(p / 'metrics.json', m)
            summary['joint'][name] = m
        # V2 start equality uses CM, as in real old records.
        c['reference_J2_best_sha256'] = summary['joint']['J2']['checkpoint_sha256']
        screen.write(folder / 'summary.json', summary)
        return root, c, ref, parent, refs, report, receipt, folder

    def test_reference_integrity_and_reuse_reject_stale_or_incomplete_artifacts(self):
        for damage in (None, 'checkpoint', 'incomplete', 'wrong_start', 'wrong_order', 'wrong_selection'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                repo, c, ref, parent, refs, report, receipt, folder = self.fixture(directory)
                if damage == 'checkpoint':
                    (folder / 'J2/best.pt').write_bytes(b'stale checkpoint')
                if damage == 'incomplete':
                    screen.write(folder / 'status.json', {'status': 'running'})
                if damage in ('wrong_start', 'wrong_order', 'wrong_selection'):
                    s = screen.read(folder / 'summary.json')
                    if damage == 'wrong_start':
                        s['binding']['stage_A_best_sha256'] = 'wrong'
                    if damage == 'wrong_order':
                        s['joint']['J2']['orders'][0] = 'wrong'
                    if damage == 'wrong_selection':
                        s['joint']['J2']['best_epoch'] = 0
                    screen.write(folder / 'summary.json', s)
                def check():
                    return screen.reference_receipt(repo, c, ref, parent, refs, report, receipt,
                                                    ROOT / 'configs' / c['reference_config'])
                if damage is None:
                    _, _, saved = check()
                    self.assertTrue(saved['reused'])
                else:
                    with self.assertRaises(ValueError):
                        check()

    def test_data_order_guard_runs_before_yielding_first_batch(self):
        class Sampler:
            def __init__(self, *args):
                self.last_order_sha256 = None
            def __iter__(self):
                self.last_order_sha256 = 'correct'
                yield [0, 1]
                yield [2, 3]
        good = screen.checked_sampler(SimpleNamespace(TrainBatches=Sampler), ['correct'])(4, 2, 42)
        self.assertEqual(list(good), [[0, 1], [2, 3]])
        bad = screen.checked_sampler(SimpleNamespace(TrainBatches=Sampler), ['other'])(4, 2, 42)
        with self.assertRaises(ValueError):
            next(iter(bad))

    def test_screen_uses_stronger_j2_not_just_v2_or_epoch_zero_floor(self):
        c, _ = self.config()
        def m(score):
            return {'best_validation': {'macro_f1': score, 'per_class': {}}, 'best_epoch': 0}
        data = {'status': 'completed', 'test_evaluated': False, 'protocol': c,
                'N0': m(.9699623354), 'frozen_V2': m(.9709366725),
                'reference_joint': {'J0': m(.9702289873), 'J2': m(.9712071203)},
                'candidate': m(.9709366725), 'stats_on_minus_off_pp': 0.,
                'comparisons': {}, 'stage_A': {}, 'attribution': 'no component claim'}
        self.assertFalse(analyze(data)['budget_screening']['score_threshold_passed'])
        data['candidate'] = m(.97195)  # beats V2 by >.10pp, not J2
        self.assertFalse(analyze(data)['budget_screening']['score_threshold_passed'])
        data['candidate'] = m(.97225)
        self.assertTrue(analyze(data)['budget_screening']['score_threshold_passed'])

    def test_report_includes_full_classes_and_unknown_historical_time(self):
        c, _ = self.config()
        m = screen.read(ROOT / 'configs/n0_seed42_reference_20261029.json')['model']
        model = {**m, 'selected_epoch_zero': False, 'training_evidence': {}, 'epochs': 17}
        counts = {'counts': {'repaired': 0, 'regressed': 0}, 'net_correct': 0}
        r = {'status': 'completed', 'test_evaluated': False, 'protocol': c,
             'N0': model, 'frozen_V2': model, 'reference_joint': {'J0': model, 'J2': model},
             'candidate': model, 'stats_off': model['best_validation'], 'stats_on_minus_off_pp': 0.,
             'comparisons': {'lower_lr_J2': counts}, 'stage_A': {'elapsed_seconds': None},
             'preflight_wall_seconds': 0.1, 'post_training_diagnostics_wall_seconds': 0.1,
             'attribution': 'No isolated component attribution.'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screen.write(root / 'summary.json', r)
            text = summarize(root)
            self.assertIn('WebBased', text)
            self.assertIn('Spoofing', text)
            self.assertIn('None', text)
            self.assertEqual(len((root / 'per_class.csv').read_text().splitlines()), 49)
            self.assertFalse(screen.read(root / 'analysis.json')['budget_screening']['score_threshold_passed'])


if __name__ == '__main__':
    unittest.main()
