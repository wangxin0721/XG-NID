"""Check margin fitting, correction boundaries and matched training."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_recon_webbased_rerank as run


class RerankChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_rules_preserve_ties_and_allow_other_true_labels_to_flip(self):
        scores = torch.full((4, 8), -8.)
        scores[0, run.WEB], scores[0, run.RECON] = 0, -1
        scores[1, run.RECON], scores[1, run.WEB] = 0, -1
        scores[2, 2], scores[2, run.WEB] = 0, -1
        scores[3, run.WEB], scores[3, run.RECON] = 0, 0
        rows = [{'sample_id': str(i), 'label': y} for i, y in enumerate((run.RECON, 0, 2, run.WEB))]
        records = run.diagnosis.ranking_records(scores, rows)
        self.assertEqual(run.eligible_training_rows(records), [0, 3])
        outputs = run.rerank(records, torch.tensor([1., -2., 100., 0.]))
        self.assertEqual([row['prediction'] for row in outputs], [run.WEB, run.WEB, 2, run.WEB])
        self.assertEqual(outputs[2]['delta'], 0)
        self.assertEqual(outputs[0]['corrected_margin'], 0)

    def test_constant_bias_uses_the_training_bce_optimum(self):
        margin = torch.zeros(4)
        labels = torch.tensor([0., 0., 0., 1.])
        fitted = run.fit_bias(margin, labels)
        self.assertAlmostEqual(fitted['bias'], float(torch.tensor(1 / 3, dtype=torch.float64).log()), places=12)
        self.assertEqual(fitted['fit_samples'], 4)

    def test_preparation_fit_and_archived_reports_use_matched_heads(self):
        rng = torch.Generator().manual_seed(83)
        train = {key: torch.randn((12, 32), generator=rng) for key in ('D32Fused', 'D32Flow', 'D32Packet')}
        val = {key: value[:6] + 2 for key, value in train.items()}
        scores = torch.full((6, 8), -8.)
        scores[:, run.WEB], scores[:, run.RECON] = 0, -1
        rows = [{'sample_id': str(i), 'label': y} for i, y in enumerate(
            (run.RECON, run.WEB, run.WEB, 0, run.RECON, run.RECON))]
        ranking = run.diagnosis.ranking_records(scores, rows)
        selected = list(range(8))
        device = torch.device('cpu')
        a, b, norm = run.prepare_features(train, val, selected, device)
        torch.testing.assert_close(norm['Fused']['mean'], train['D32Fused'][selected].mean(0))
        torch.testing.assert_close(norm['Difference']['mean'],
                                   (train['D32Flow'][selected] - train['D32Packet'][selected]).mean(0))
        config = {'seed': 42, 'batch_size': 4, 'epochs': 2, 'learning_rate': 0.001, 'weight_decay': 0,
                  'budget_reference_pp': .1}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            margin, labels = torch.tensor([-1., 1.] * 4), torch.tensor([1., 0.] * 4)
            best, last, exports, updates = run.fit_heads(a, b, margin, labels, ranking, config, output)
            bias = run.fit_bias(margin, labels)
            metrics, effects, _ = run.summarize(
                dict.fromkeys(('C', 'A', 'B37'), ranking),
                {'Bias': run.rerank(ranking, torch.zeros(6)), **exports}, output)
            run.write_report(output, metrics, effects, best, last, bias, config, updates)
            self.assertTrue(run.raw.read(output / 'preflight.json')['passed'])
            ledger = run.joint.query.read_history(output / 'epochs.jsonl')
            self.assertEqual([r['updates_per_head'] for r in ledger], [0, 2, 2])
            self.assertEqual(updates, 4)
            for name in run.HEADS:
                expected = max(ledger, key=lambda r: r['validation'][name]['macro_f1'])
                self.assertEqual(best[name]['epoch'], expected['epoch'])
                self.assertEqual(last[name]['epoch'], 2)

    def test_cache_reuses_valid_archive_and_extraction_keeps_new_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'outputs' / '.fusion_decision_feature_cache' / ('a' * 64) / 'val.pt'
            old.parent.mkdir(parents=True)
            rows = [{'sample_id': '1', 'label': 1}]
            features = {name: torch.zeros((1, size)) for name, size in run.diagnosis.FEATURES.items()}
            binding = {'device': 'old', 'torch': torch.__version__}
            torch.save({'binding': {**binding, 'partition': 'val'}, 'sample_ids': ['1'], 'labels': [1],
                        'features': features, 'feature_sha256': run.raw.state_digest(features)}, old)
            provenance = {'feature_caches': {'val': {'path': str(old), 'sha256': run.raw.sha(old)}}}
            current = {'device': 'new'}
            args = (root, root / 'outputs' / 'new', provenance, binding, current, None, None,
                    rows, 'val', {}, torch.device('cpu'))
            with patch.object(run.diagnosis, 'cached_features') as extract:
                _, reused = run.load_features(*args)
                extract.assert_not_called()
                self.assertTrue(reused['reused_diagnosis_cache'])
                provenance['feature_caches']['val']['sha256'] = 'different'
                extract.return_value = (features, {'path': 'new'})
                _, rebuilt = run.load_features(*args)
                self.assertFalse(rebuilt['reused_diagnosis_cache'])
                self.assertEqual(extract.call_args.args[1], current)
            self.assertTrue(old.is_file())


if __name__ == '__main__':
    unittest.main()
