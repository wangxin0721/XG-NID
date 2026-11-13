"""Verify matched fusion features, training and top-two evidence."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_fusion_decision_diagnosis as run
from test_flow_query_joint_packet2 import TinyGraphs, joint_fixture


class FusionDecisionChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_frozen_cache_preserves_scores_and_refreshes_different_bindings(self):
        base, _, _, graphs = joint_fixture()
        model = run.joint.JointCrossLayerQueryClassifier(base.flow_encoder, base.packet_encoder,
                                                        base.fusion, 'flow').eval()
        rows = [{'sample_id': graph.sample_id, 'label': graph.y.item(), 'partition': 'val'} for graph in graphs]
        config = {'seed': 42, 'workers': 0, 'eval_batch_size': 4}
        device = torch.device('cpu')
        data = TinyGraphs(graphs)
        batches = run.joint.legacy.loader(data, rows, 'val', config)
        _, original = run.raw.evaluate(model, batches, rows, device, export=True)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            path = output / 'val.pt'
            features, receipt = run.cached_features(path, {'checkpoint': 'a'}, model, data, rows,
                                                    'val', config, device, output)
            self.assertFalse(receipt['reused'])
            with patch.object(run, 'extract', side_effect=AssertionError('Unexpected extraction')):
                cached, receipt = run.cached_features(path, {'checkpoint': 'a'}, model, data, rows,
                                                      'val', config, device, output)
                self.assertTrue(receipt['reused'])
            with patch.object(run, 'extract', return_value=features) as extractor:
                _, receipt = run.cached_features(path, {'checkpoint': 'b'}, model, data, rows,
                                                 'val', config, device, output)
                extractor.assert_called_once()
                self.assertFalse(receipt['reused'])
            for name, values in features.items():
                self.assertTrue(torch.equal(values, cached[name]))
        ranked = run.ranking_records(features['LogProbabilities'], rows)
        self.assertEqual(run.joint.legacy.identities(ranked), run.joint.legacy.identities(original))
        torch.testing.assert_close(features['D32Fused'], features['Alpha'] * features['D32Flow']
                                   + (1 - features['Alpha']) * features['D32Packet'])

    def test_pca_fits_training_only_and_heads_share_budget_and_initialization(self):
        generator = torch.Generator().manual_seed(19)
        train = {key: torch.randn(40, 32, generator=generator) for key in ('D32Flow', 'D32Packet', 'D32Fused')}
        val = {key: value[:8] + 100 for key, value in train.items()}
        (train_views, val_views), pca = run.readouts(train, val, 16)
        for name, spec in pca.items():
            torch.testing.assert_close(spec['mean'], train[name].double().mean(0))
        self.assertTrue(all(x.shape == (8, 32) for x in val_views.values()))
        torch.testing.assert_close(val_views['Equal32'], 0.5 * val['D32Flow'] + 0.5 * val['D32Packet'])
        heads = run.matched_heads(42)
        for kind, count in (('Linear', 264), ('Nonlinear', 1320)):
            selected = [heads[f'{name}_{kind}'] for name in run.READOUTS]
            self.assertEqual(len({run.raw.state_digest(head.state_dict()) for head in selected}), 1)
            self.assertEqual(sum(p.numel() for p in selected[0].parameters()), count)
        examples = [{'sample_id': str(i), 'label': i % 8} for i in range(40)]
        config = {'seed': 42, 'batch_size': 8, 'epochs': 2, 'learning_rate': 0.001, 'weight_decay': 0}
        tr = {name: train_views[name.split('_')[0]] for name in heads}
        va = {name: val_views[name.split('_')[0]] for name in heads}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            predictions, checkpoints, updates = run.transition.fit_probes(
                heads, tr, va, examples, examples[:8], config, torch.device('cpu'), output)
            ledger = run.joint.query.read_history(output / 'epochs.jsonl')
        self.assertEqual(updates, 10)
        self.assertEqual(set(ledger[0]['train_ce']), set(heads))
        self.assertEqual(set(predictions), set(heads))
        self.assertTrue(torch.equal(checkpoints['D32_Linear']['mean'], checkpoints['D32_Nonlinear']['mean']))

    def test_ranking_reports_correct_wrong_other_and_oracle_directions(self):
        scores = torch.full((5, 8), -5.)
        labels = (run.RECON, run.WEB, run.WEB, 0, run.RECON)
        for i, (first, second) in enumerate(((run.WEB, run.RECON), (run.RECON, run.WEB), (run.WEB, run.RECON),
                                             (run.WEB, run.RECON), (run.WEB, 2))):
            scores[i, first], scores[i, second] = 0, -1
        rows = [{'sample_id': str(i), 'label': label} for i, label in enumerate(labels)]
        rank = run.ranking_records(scores, rows)
        with tempfile.TemporaryDirectory() as tmp:
            reference = dict.fromkeys(('C', 'A', 'B37'), rank)
            metrics, effects, coverage, subsets = run.summarize(reference, {}, {'val': rank, 'train': rank}, Path(tmp))
        self.assertEqual(coverage['val']['eligible'], 4)
        self.assertEqual(coverage['val']['eligible_correct'], {'WebBased': 1})
        self.assertEqual(coverage['val']['eligible_other_classes'], {'Benign': 1})
        self.assertEqual(effects['Oracle']['counts']['repaired'], 2)
        self.assertEqual(effects['Oracle']['directions']['Recon -> WebBased']['total_errors'], 1)
        self.assertEqual(metrics['Oracle']['accuracy'], 3 / 5)
        self.assertEqual(len(subsets['common_Recon']), 2)


if __name__ == '__main__':
    unittest.main()
