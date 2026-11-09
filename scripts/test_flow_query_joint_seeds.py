"""CPU checks for probability exports, paired evidence, source matching and audit guards."""
import contextlib
import copy
import io
import math
import statistics
import tempfile
import unittest
from pathlib import Path

import torch

import run_flow_query_joint_seeds as run
import test_flow_query_joint_packet2 as fixtures


def predictions(values):
    records = []
    for y, p in enumerate(values):
        probs = [.01]*8
        probs[p] = .93
        records.append({'sample_id': 'sample'+str(y), 'label': y, 'prediction': p,
                        'probabilities': probs, 'log_probabilities': [math.log(v) for v in probs]})
    return records


def tiny_config(epochs=3):
    return {**fixtures.config(seed=43, epochs=epochs), 'encoder_lr': 1e-6, 'workers': 0,
            'new_seeds': [43, 44], 'aggregate_seeds': [42, 43, 44], 'budget_threshold_pp': .1,
            'focus_classes': ['WebBased', 'Spoofing', 'Recon'], 'JCL_Multi_reference_macro_f1': .9752460910749855,
            'source_sha256': '7815875a111d3e94ceefd0e9ef6cb64dedaf6ff80723645f1ec035b376af31ff'}


class SeedChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')
        self.temporary = run.raw.ROOT/'tmp'/'flow_query_seed_checks'
        self.temporary.mkdir(parents=True, exist_ok=True)

    def test_new_stage_does_not_inherit_seed42_resume_or_deferred_seed_fields(self):
        old = {'seed': 42, 'resume_directory': 'old', 'resume_seed42_from_epoch': 150,
               'additional_seeds_deferred': [43, 44], 'variants': {'Mean-Query': 'packet_mean'}, 'encoder_lr': 1e-6}
        study = {'new_seeds': [43, 44], 'variants': run.core.JOINT}
        config = run.stage_config(old, study)
        self.assertEqual(config['variants'], run.core.JOINT)
        self.assertEqual(config['encoder_lr'], 1e-6)
        self.assertFalse({'seed', 'resume_directory', 'resume_seed42_from_epoch', 'additional_seeds_deferred'} & config.keys())
        self.assertEqual(old['seed'], 42)

    def test_probability_vectors_ids_and_labels_are_verified(self):
        good = predictions(list(range(8)))
        self.assertEqual(run.validate_predictions(good)['macro_f1'], 1.)
        for mutate in ('duplicate', 'wrong_order', 'normalization', 'argmax', 'logs'):
            with self.subTest(mutate=mutate):
                rows = copy.deepcopy(good)
                if mutate == 'duplicate':
                    rows[1]['sample_id'] = rows[0]['sample_id']
                elif mutate == 'wrong_order':
                    rows.reverse()
                elif mutate == 'normalization':
                    rows[0]['probabilities'][0] = .1
                elif mutate == 'argmax':
                    rows[0]['prediction'] = 1
                else:
                    rows[0]['log_probabilities'][0] -= .2
                with self.assertRaises(ValueError):
                    run.validate_predictions(rows, good)

    def test_sample_evidence_distinguishes_repairs_from_other_wrong_predictions(self):
        mean = predictions([2, 3, 2, 1, 4, 5, 6, 7])
        flow = predictions([1, 1, 0, 3, 4, 5, 6, 7])
        with tempfile.TemporaryDirectory(dir=self.temporary) as tmp:
            folder = Path(tmp)
            result = run.paired_evidence(mean, flow, folder, 43)
            self.assertEqual(result['comparison'], {'counts': {'repaired': 2, 'regressed': 1,
                              'both_correct': 4, 'both_wrong': 1}, 'net_correct': 1})
            self.assertEqual(result['WebBased_Recon_repaired'], 2)
            self.assertEqual(result['WebBased_Recon_regressed'], 0)
            self.assertEqual(result['WebBased_Recon_share_of_all_repairs'], 1.)
            self.assertEqual(result['per_class']['Spoofing']['net_correct'], -1)
            rows = run.core.read_predictions(folder/'paired_validation.jsonl.gz')
            changed = run.core.read_predictions(folder/'changed_samples.jsonl.gz')
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(changed), 4)
            self.assertEqual(changed[0]['change'], 'both_wrong_changed')
            self.assertEqual(changed[0]['mean_probabilities'], mean[0]['probabilities'])
            self.assertEqual(changed[0]['flow_probabilities'], flow[0]['probabilities'])
            expected = run.raw.compare(mean, flow)[1]
            self.assertEqual([{key: row[key] for key in expected[0]} for row in changed], expected)

    def test_paired_input_mismatch_is_rejected(self):
        mean, flow = predictions(list(range(8))), predictions(list(range(8)))
        flow[2]['label'] = 1
        with tempfile.TemporaryDirectory(dir=self.temporary) as tmp:
            with self.assertRaisesRegex(ValueError, 'IDs/labels/order'):
                run.paired_evidence(mean, flow, Path(tmp), 44)

    def test_three_seed_stats_keep_original_failure_and_separate_new_seeds(self):
        pairs = {}
        deltas = [.09863128176158531, .12, -.04]
        for seed, delta in zip((42, 43, 44), deltas):
            pairs[str(seed)] = {'models': {name: {'best_validation': {'macro_f1': .97+(delta/100 if name == 'Joint-Flow' else 0)}}
                                          for name in run.NAMES},
                                'analysis': {'Joint_Flow_minus_Joint_Mean_pp': delta,
                                    'error_types': [{'true_class': 'WebBased', 'wrong_prediction': 'Recon',
                                                     'repaired': 2, 'regressed': 1}]}}
        result = run.summarize(pairs, tiny_config())
        self.assertAlmostEqual(result['paired_delta_mean_pp'], statistics.mean(deltas))
        self.assertAlmostEqual(result['paired_delta_sample_std_pp'], statistics.stdev(deltas))
        self.assertAlmostEqual(result['new_seeds43_44']['mean_pp'], .04)
        self.assertAlmostEqual(result['new_seeds43_44']['sample_std_pp'], statistics.stdev(deltas[1:]))
        self.assertFalse(result['all_three_positive'])
        self.assertFalse(result['seed42_budget_passed'])
        self.assertFalse(result['statistical_significance_claimed'])
        self.assertEqual(set(result['error_type_recurrence']['WebBased->Recon']), {'42', '43', '44'})
        del pairs['42']
        with self.assertRaises(ValueError):
            run.summarize(pairs, tiny_config())

    def test_new_seeds_share_mature_encoders_and_match_within_each_pair(self):
        source = run.raw.OperatorClassifier('GAT', seed=42).state_dict()
        keys = []
        for seed in (43, 44):
            models = run.build_models(source, seed, self.device)
            a, b = [models[name] for name in run.NAMES]
            run.core.assert_tree_equal(a.state_dict(), b.state_dict())
            for name, value in source.items():
                self.assertTrue(torch.equal(value, a.state_dict()[name]))
            self.assertEqual(sum(p.numel() for p in a.parameters() if p.requires_grad), 25384)
            self.assertTrue(all(not module.training for e in (a.flow_encoder, a.packet_encoder) for module in e.modules()))
            keys.append(a.packet_pool.key.weight.detach().clone())
        self.assertFalse(torch.equal(*keys))
        a = run.epoch_orders(8, 4, 43, 3)
        self.assertEqual(a, run.epoch_orders(8, 4, 43, 3))
        self.assertNotEqual(a, run.epoch_orders(8, 4, 44, 3))

    def test_raw_probability_replay_keeps_state_and_matches_original_evaluation(self):
        model, _, _, graphs = fixtures.joint_fixture('flow')
        data = fixtures.TinyGraphs(graphs)
        rows = [{'sample_id': g.sample_id, 'label': int(g.y), 'partition': 'val'} for g in graphs]
        c = tiny_config()
        original, original_rows = run.core.joint_evaluate(model, run.core.graph_loader(data, rows, 'val', c), rows, self.device)
        before = run.raw.state_digest(model.state_dict())
        measured, replay = run.probability_replay(model, run.core.graph_loader(data, rows, 'val', c), rows, self.device)
        self.assertEqual(measured, original)
        self.assertEqual(run.core.parent.identities(replay), run.core.parent.identities(original_rows))
        self.assertEqual(run.raw.state_digest(model.state_dict()), before)
        self.assertEqual(len(replay[0]['probabilities']), 8)
        self.assertEqual(len(replay[0]['flow_weights']), 32)
        self.assertGreater(len(replay[0]['packet_attention']), 0)

    def test_selected_checkpoint_audit_rejects_wrong_epoch_or_adam_steps(self):
        model, _, _, graphs = fixtures.joint_fixture()
        data, c = fixtures.TinyGraphs(graphs), tiny_config()
        rows = {part: [{'sample_id': g.sample_id, 'label': int(g.y), 'partition': part} for g in graphs]
                for part in ('train', 'val')}
        with tempfile.TemporaryDirectory(dir=self.temporary) as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp)
            result, _ = run.core.fit_joint('Joint-Flow', model, data, rows, self.device, c, {}, folder,
                                          run.epoch_orders(8, 4, 43, 3))
            source = folder/'Joint-Flow'
            saved = torch.load(source/'best.pt', map_location='cpu', weights_only=False)
            history = run.core.read_history(source/'epochs.jsonl')
            run.checkpoint_check(saved, model, result, history, 'best.pt', 'Joint-Flow', 2)
            bad = copy.deepcopy(saved)
            bad['epoch'] += 1
            with self.assertRaisesRegex(ValueError, 'epoch/metrics'):
                run.checkpoint_check(bad, model, result, history, 'best.pt', 'Joint-Flow', 2)
            bad = copy.deepcopy(saved)
            bad['config']['variant'] = 'Joint-Mean'
            with self.assertRaisesRegex(ValueError, 'variant'):
                run.checkpoint_check(bad, model, result, history, 'best.pt', 'Joint-Flow', 2)
            last = torch.load(source/'last.pt', map_location='cpu', weights_only=False)
            run.checkpoint_check(last, model, result, history, 'last.pt', 'Joint-Flow', 2)
            bad = copy.deepcopy(last)
            next(iter(bad['optimizer_state']['state'].values()))['step'] += 1
            with self.assertRaisesRegex(ValueError, 'Adam steps'):
                run.checkpoint_check(bad, model, result, history, 'last.pt', 'Joint-Flow', 2)


if __name__ == '__main__':
    unittest.main()
