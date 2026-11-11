"""CPU checks for frozen training, paired budgets and sample-level comparisons."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

import train_flow_query_strength_pair as run
from test_flow_query_joint_packet2 import joint_fixture


class StrengthPairChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_cached_components_and_zero_initialized_arms(self):
        model, _, batch, _ = joint_fixture()
        model.requires_grad_(False).eval()
        model.packet_pool.query.weight.fill_(0.25)
        rows = [{'sample_id': sid, 'label': label} for sid, label in zip(batch.sample_id, batch.y.tolist())]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            features, labels, original = run.extract_features(model, [batch], rows, torch.device('cpu'), Path(tmp), 'val')
        with torch.no_grad():
            old_scores, _, _ = model.forward_with_attention(*model.encode(batch))
        self.assertEqual([row['prediction'] for row in original], old_scores.argmax(-1).tolist())
        self.assertEqual(labels.tolist(), batch.y.tolist())
        self.assertFalse(features.requires_grad)
        self.assertTrue(bool(features[:, 2].ne(0).any()))
        a = run.evaluate(model.fusion, 'global', run.GlobalStrength(), features, rows, 4)
        b = run.evaluate(model.fusion, 'adaptive', run.adaptive_strength(), features, rows, 4)
        self.assertEqual(a, b)
        self.assertTrue(all(row['lambda'] == 0.5 for row in a[1]))

    def test_paired_training_budget_and_best_selection(self):
        model, _, _, _ = joint_fixture()
        model.requires_grad_(False).eval()
        frozen = run.raw.state_digest(model.state_dict())
        run.raw.seed_all(42)
        features = torch.randn(24, 3, 64)
        labels = torch.arange(24).remainder(8)
        val_rows = [{'sample_id': str(i), 'label': i} for i in range(8)]
        config = {'epochs': 3, 'batch_size': 6, 'eval_batch_size': 4, 'seed': 42, 'learning_rate': 0.001}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp)
            best, exports, updates, counts = run.fit_pair(model.fusion, features[:16], labels[:16], features[16:],
                                                        val_rows, config, output)
            ledger = [json.loads(line) for line in (output / 'epochs.jsonl').read_text().splitlines()]
        self.assertEqual(updates, 9)
        self.assertEqual(counts, {'global': 1, 'adaptive': 129})
        self.assertEqual(run.raw.state_digest(model.state_dict()), frozen)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual([r['epoch'] for r in ledger], [0, 1, 2, 3])
        sampler = run.raw.TrainBatches(16, 6, 42)
        for row in ledger[1:]:
            list(sampler)
            self.assertEqual(row['order_sha256'], sampler.order_sha256)
            self.assertEqual((row['samples_per_arm'], row['updates_per_arm']), (16, 3))
        for arm in run.ARMS:
            chosen = max(ledger[1:], key=lambda row: row['validation'][arm]['macro_f1'])
            self.assertEqual(best[arm]['epoch'], chosen['epoch'])
            self.assertEqual(best[arm]['validation'], chosen['validation'][arm])
            self.assertEqual([r['sample_id'] for r in exports[arm]], [r['sample_id'] for r in val_rows])
        self.assertEqual(len({r['lambda'] for r in exports['global']}), 1)
        self.assertGreater(len({r['lambda'] for r in exports['adaptive']}), 1)

    def test_retention_repair_and_comparison_to_stronger_control(self):
        baseline = [{'label': 1, 'prediction': 1}, {'label': 1, 'prediction': 1},
                    {'label': 3, 'prediction': 3}, {'label': 3, 'prediction': 1}]
        learned = [{'label': 1, 'prediction': 1}, {'label': 1, 'prediction': 3},
                   {'label': 3, 'prediction': 1}, {'label': 3, 'prediction': 3}]
        focus = run.compare(baseline, learned)
        self.assertEqual(focus['WebBased']['retained'], 1)
        self.assertEqual(focus['WebBased']['original_correct'], 2)
        self.assertEqual(focus['Recon']['repaired'], 1)
        self.assertEqual(focus['Recon']['regressed'], 1)
        def metrics(f1, web, recon):
            return {'macro_f1': f1, 'per_class': {'WebBased': {'recall': web}, 'Recon': {'recall': recon}}}
        original = metrics(0.97, 0.94, 0.96)
        trained = {'global': {'validation': metrics(0.972, 0.94, 0.96)},
                   'adaptive': {'validation': metrics(0.9725, 0.94, 0.97)}}
        summary = run.summarize(original, trained, 0.1)
        self.assertEqual(summary['stronger_control'], 'global')
        self.assertTrue(summary['adaptive_beats_global'])
        self.assertTrue(summary['webbased_recall_preserved_and_recon_improved'])
        self.assertFalse(summary['macro_f1_budget_met'])


if __name__ == '__main__':
    unittest.main()
