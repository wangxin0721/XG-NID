"""No-torch checks: mature starts, grouping, epoch0 selection and attribution."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_joint_fusion32 as joint
from summarize_joint_fusion32 import analyze, changed_samples, summarize
import test_fusion_gate_pair as previous_tests

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Benign', 'WebBased', 'Spoofing', 'Recon', 'Mirai', 'Dos', 'DDos', 'BruteForce']


class Checks(unittest.TestCase):
    def config(self):
        return joint.read(ROOT/'configs/joint_fusion32_s42_v1.json')

    def test_scope_mature_starts_and_joint_budget_are_local_to_new_config(self):
        self.assertNotIn('torch', sys.modules)
        c = self.config()
        joint.check_config(c)
        self.assertEqual((c['epochs'], c['patience'], c['include_epoch_zero']), (30, 12, True))
        self.assertEqual([c['variants'][n]['epoch'] for n in joint.VARIANTS], [72, 62])
        self.assertNotEqual(c['variants']['J32-Equal']['macro_f1'], c['variants']['J32-Gate']['macro_f1'])
        old = joint.read(ROOT/'configs/fusion_gate_pair_s42_v1.json')
        self.assertEqual((old['epochs'], old['patience'], old['include_epoch_zero'], old['lr']), (100, 20, False, .001))
        for key, value in [('encoder_mode', 'train'), ('inputs', 'cached_64D'), ('epochs', 100),
                           ('include_epoch_zero', False), ('seed', 43), ('warmup_training', True),
                           ('head_reinitialization', True), ('automatic_followup', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                joint.check_config({**c, key: value})

    def test_parameter_groups_deduplicate_aliases_and_reject_unknown_or_conflicting_roles(self):
        class P:
            def __init__(self, count): self.count, self.requires_grad = count, False
            def numel(self): return self.count
            def requires_grad_(self, value): self.requires_grad = value
        enc, head, gate = P(42), P(5480), P(1664)
        names = [('flow_encoder.w', enc), ('packet_encoder.shared', enc),
                 ('fusion.flow_projection.w', head), ('fusion.gate.0.w', gate)]
        model = SimpleNamespace(named_parameters=lambda **kw: iter(names), parameters=lambda: iter([enc, head, gate]))
        groups = joint.groups(model, 'equal', self.config())
        self.assertEqual([g['name'] for g in groups], ['encoder', 'head'])
        self.assertTrue(enc.requires_grad)
        self.assertFalse(gate.requires_grad)
        groups = joint.groups(model, 'dynamic', self.config())
        self.assertEqual([g['lr'] for g in groups], [1e-6, 1e-5, 1e-5])
        self.assertEqual(len(groups[0]['params']), 1)
        self.assertTrue(gate.requires_grad)
        names.append(('fusion.shared_head.alias', enc))
        with self.assertRaises(ValueError): joint.groups(model, 'dynamic', self.config())
        names.pop()
        names.append(('legacy_unrecognized.w', P(5)))
        with self.assertRaises(ValueError): joint.groups(model, 'dynamic', self.config())
        self.assertEqual(joint.group_name('fusion.packet_projection.weight', 'equal'), 'head')
        with self.assertRaises(ValueError): joint.group_name('shared_head.old_prefix', 'equal')

    def test_encoder_eval_is_independent_of_gradients_and_buffers_are_checked(self):
        param = SimpleNamespace(requires_grad=True)
        child = SimpleNamespace(training=False)
        encoder = SimpleNamespace(modules=lambda: iter([child]), parameters=lambda: iter([param]))
        model = SimpleNamespace(flow_encoder=encoder, packet_encoder=encoder,
            named_buffers=lambda: iter([('flow_encoder.bn.running_mean', 'buffer')]))
        core = SimpleNamespace(state_hash=lambda values: str(list(values)))
        expected = core.state_hash(joint.buffer_values(model))
        joint.assert_modes_buffers(core, model, expected)
        child.training = True
        with self.assertRaises(ValueError): joint.assert_modes_buffers(core, model, expected)
        child.training = False
        param.requires_grad = False
        with self.assertRaises(ValueError): joint.assert_modes_buffers(core, model, expected)
        param.requires_grad = True
        with self.assertRaises(ValueError): joint.assert_modes_buffers(core, model, 'changed_BN')

    def history(self, scores, start=.97):
        best, selected, stale = start, 0, 0
        history = []
        for epoch, value in enumerate(scores, 1):
            best, stale, improved = joint.pair.selection_state(value, best, stale)
            if improved: selected = epoch
            history.append({'epoch': epoch, 'val_macro_f1': value, 'best_epoch': selected,
                'stale_epochs': stale, 'optimizer_updates': 211, 'train_order_sha256': str(epoch)})
        return history, {'epoch_zero': {'macro_f1': start}, 'best_validation': {'macro_f1': best},
            'best_epoch': selected, 'selected_epoch_zero': selected == 0, 'epochs': len(scores),
            'optimizer_updates': len(scores)*211, 'orders': [str(x) for x in range(1, len(scores)+1)]}

    def test_epoch_zero_is_retained_on_ties_or_degradation_with_patience12(self):
        history, result = self.history([.97]*12)
        joint.check_history(history, result, self.config())
        self.assertTrue(result['selected_epoch_zero'])
        history, result = self.history([.971]+[.9705]*12)
        joint.check_history(history, result, self.config())
        self.assertEqual(result['best_epoch'], 1)
        history, result = self.history([.96]*11)
        with self.assertRaises(ValueError): joint.check_history(history, result, self.config())
        history, result = self.history([.97+i*.00001 for i in range(1, 31)])
        joint.check_history(history, result, self.config())
        self.assertEqual(result['best_epoch'], 30)

    def test_changed_samples_export_ids_and_correct_transition_types(self):
        ref = [{'sample_id': str(i), 'label': 1, 'prediction': p} for i, p in enumerate([3, 1, 0, 1])]
        cand = [{**a, 'prediction': p} for a, p in zip(ref, [1, 3, 2, 1])]
        changes = changed_samples(ref, cand, LABELS)
        self.assertEqual([r['change'] for r in changes], ['repaired', 'regressed', 'both_wrong_changed'])
        self.assertEqual(changes[0]['sample_id'], '0')
        self.assertEqual(changes[0]['old_class'], 'Recon')
        with self.assertRaises(ValueError): changed_samples(ref, cand[::-1], LABELS)
        with self.assertRaises(ValueError): changed_samples([ref[0], ref[0]], cand[:2], LABELS)

    def fixture(self, improved=False):
        source = previous_tests.Checks().fixture()['models']
        result = {}
        for name, (oldname, mode) in joint.VARIANTS.items():
            zero = source[oldname]['best_validation']
            best = copy.deepcopy(zero)
            if improved and mode == 'dynamic':
                cm = copy.deepcopy(best['confusion_matrix'])
                cm[1][3] -= 1
                cm[1][1] += 1
                best = joint.dim.frozen.matrix_report(cm, LABELS)
            result[name] = {'epoch_zero': zero, 'best_validation': best, 'last_validation': zero,
                'epoch_zero_eligible': True, 'selected_epoch_zero': not (improved and mode == 'dynamic'),
                'best_epoch': 1 if improved and mode == 'dynamic' else 0, 'epochs': 12,
                'optimizer_updates': 2532, 'orders': ['order']*12, 'training_start_rng_sha256': 'rng',
                'training_evidence': {}, 'training_wall_seconds': 1., 'parameter_groups': [], 'joint_trainable_parameters': 99}
        sources = {'D32-Equal': source['D32-Equal'], 'D32-Gate': source['D32-Gate']}
        counts = {n: {'counts': {'repaired': int(improved and n == 'J32-Gate'), 'regressed': 0},
                       'net_correct': int(improved and n == 'J32-Gate')} for n in joint.VARIANTS}
        transitions = {n: ([{'kind': 'repaired', 'true_class': 'WebBased', 'equal_prediction': 'Recon',
                            'gate_prediction': 'WebBased', 'count': 1}] if improved and n == 'J32-Gate' else []) for n in joint.VARIANTS}
        return {'status': 'completed', 'exit_code': 0, 'tensor_preflight_passed': True,
            'encoder_BN_buffers_unchanged': True, 'encoder_parameters_trainable': True,
            'raw_graph_training': True, 'frozen_cache_used_for_training': False,
            'source_artifacts_unchanged': True, 'test_evaluated': False, 'automatic_followup': False,
            'protocol': self.config(), 'models': result, 'source_models': sources,
            'own_start_changes': counts, 'own_start_transitions': transitions,
            'Gate_versus_Equal': {'counts': {'repaired': 1, 'regressed': 0}, 'net_correct': 1},
            'Gate_versus_Equal_transitions': [], 'common_epochs_order_checked': 12,
            'graph_replays': {}, 'discarded_probe_updates': 6, 'preflight_wall_seconds': 2.}

    def test_report_separates_inherited_advantage_from_new_joint_gain(self):
        r = self.fixture()
        a = analyze(r)
        self.assertGreater(a['initial_Gate_minus_Equal_pp'], 0)
        self.assertEqual(a['Gate_gain_from_own_start_pp'], 0.)
        self.assertEqual(a['gap_change_from_joint_training_pp'], 0.)
        self.assertFalse(a['budget_screen_passed'])
        self.assertTrue(a['Gate_selected_epoch_zero'])
        improved = analyze(self.fixture(True))
        self.assertGreater(improved['Gate_gain_from_own_start_pp'], 0.)
        self.assertAlmostEqual(improved['Gate_gain_from_own_start_pp'], improved['gap_change_from_joint_training_pp'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joint.write(root/'summary.json', r)
            text = summarize(root)
            self.assertIn('Inherited Gate-Equal gap', text)
            self.assertIn('Matching the starting score alone is not evidence', text)
            self.assertTrue((root/'analysis.json').exists())
            with (root/'per_class.csv').open(newline='') as stream:
                self.assertEqual(len(list(csv.reader(stream))), 65)
        bad = copy.deepcopy(r)
        bad['models']['J32-Gate']['epoch_zero'] = bad['models']['J32-Equal']['epoch_zero']
        with self.assertRaises(ValueError): analyze(bad)

    def test_budget_reference_uses_stronger_new_equal_not_just_old_gate(self):
        r = self.fixture(True)
        # Both improve, Equal becomes stronger: the budget target must move with it.
        cm = copy.deepcopy(r['models']['J32-Gate']['best_validation']['confusion_matrix'])
        cm[1][3] -= 2
        cm[1][1] += 2
        higher = joint.dim.frozen.matrix_report(cm, LABELS)
        r['models']['J32-Equal'].update(best_validation=higher, best_epoch=1, selected_epoch_zero=False)
        a = analyze(r)
        self.assertEqual(a['stronger_reference_macro_f1'], higher['macro_f1'])
        self.assertTrue(a['Equal_overtook_Gate'])
        self.assertFalse(a['budget_screen_passed'])

    def source_fixture(self, root):
        c = self.config()
        (root/'configs').mkdir()
        protocol = joint.read(ROOT/'configs'/c['source_protocol'])
        (root/'configs'/c['source_protocol']).write_bytes((ROOT/'configs'/c['source_protocol']).read_bytes())
        source = root/c['source_directory']
        source.mkdir(parents=True)
        s = previous_tests.Checks().fixture()
        b = {'validated_previous_reference': True}
        s.update(binding=b, protocol=protocol, source_artifacts_unchanged=True)
        for name, (old, mode) in joint.VARIANTS.items():
            folder = source/old
            folder.mkdir()
            m = s['models'][old]
            for filename in ('best.pt', 'last.pt', 'validation_predictions.jsonl.gz'):
                (folder/filename).write_bytes(b'integrity fixture '+filename.encode())
            m.update(checkpoint_sha256={n: joint.sha(folder/n) for n in ('best.pt', 'last.pt')},
                     predictions_sha256=joint.sha(folder/'validation_predictions.jsonl.gz'))
            c['variants'][name].update(epoch=m['best_epoch'], macro_f1=m['best_validation']['macro_f1'],
                                      best_sha256=m['checkpoint_sha256']['best.pt'])
            replay = {'strict_state_load': True, 'confusion_matrix_reproduced': True}
            s['graph_replays'][old] = replay
            for filename, obj in [('metrics.json', m), ('config.json', {'binding': b, 'protocol': protocol,
                    'variant': old, 'architecture': joint.control.architecture(mode)}), ('graph_replay.json', replay)]:
                joint.write(folder/filename, obj)
            (folder/'epochs.jsonl').write_text('integrity fixture', encoding='utf-8')
        for filename, obj in [('status.json', {'status': 'completed', 'exit_code': 0}), ('summary.json', s),
                ('binding.json', b), ('config.json', protocol), ('gpu.json', {}), ('initialization.json', {}),
                ('probes.json', {}), ('preflight_roundtrips.json', {})]:
            joint.write(source/filename, obj)
        cfg = root/'configs/joint.json'
        joint.write(cfg, c)
        previous = (protocol, {}, None, {}, b)
        return SimpleNamespace(repo=root, config=cfg), previous, source

    def test_source_metadata_rejects_changed_weights_binding_or_missing_replay(self):
        for damage in (None, 'weights', 'binding', 'replay'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                args, previous, source = self.source_fixture(Path(directory))
                if damage == 'weights': (source/'D32-Gate/best.pt').write_bytes(b'changed')
                elif damage == 'binding': joint.write(source/'binding.json', {'changed': True})
                elif damage == 'replay': joint.write(source/'D32-Equal/graph_replay.json', {})
                with patch.object(joint.control, 'metadata', return_value=previous):
                    if damage:
                        with self.assertRaises(ValueError): joint.metadata(args)
                    else:
                        meta = joint.metadata(args)
                        joint.control.assert_source_unchanged(meta[-1])
                        (source/'D32-Equal/last.pt').write_bytes(b'changed after metadata')
                        with self.assertRaises(ValueError): joint.control.assert_source_unchanged(meta[-1])


if __name__ == '__main__':
    unittest.main()
