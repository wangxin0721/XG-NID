"""CPU regression checks for the fixed-strength validation replay."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch

import replay_flow_query_strength as run
from test_flow_query_joint_packet2 import joint_fixture


class StrengthReplayChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    @torch.no_grad()
    def test_pinned_checkpoint_and_lambda_one_reproduce_original_joint_flow(self):
        model, _, batch, _ = joint_fixture()
        model.packet_pool.query.weight.fill_(0.25)
        features = model.encode(batch)
        original_scores, _, _ = model.forward_with_attention(*features)
        flow, nodes, index, size = features
        readouts = run.packet_readouts(model.packet_pool, *features)
        means = torch.stack([nodes[index == i].mean(dim=0) for i in range(size)])
        torch.testing.assert_close(readouts['0'], means)
        torch.testing.assert_close(readouts['0.5'], (readouts['1'] + means) * 0.5)
        self.assertFalse(torch.allclose(readouts['0.5'], means))
        scores, _ = model.fusion(flow, readouts['1'])
        self.assertTrue(torch.equal(scores, original_scores))

        original = [
            {'sample_id': sid, 'label': label, 'prediction': pred}
            for sid, label, pred in zip(batch.sample_id, batch.y.tolist(), original_scores.argmax(-1).tolist())
        ]
        metrics = run.study.validate_predictions(original, probabilities=False)
        protocol = {'seed': 42, 'split_id': 'test', 'eval_batch_size': 16, 'workers': 0}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            flow_dir, mean_dir = root / 'Joint-Flow', root / 'Joint-Mean'
            flow_dir.mkdir()
            mean_dir.mkdir()
            checkpoint = flow_dir / 'best.pt'
            flow_path = flow_dir / 'validation_predictions.jsonl.gz'
            mean_path = mean_dir / flow_path.name
            torch.save({'epoch': 95, 'model_state': model.state_dict(), 'validation': metrics,
                        'config': {'variant': 'Joint-Flow', 'labels': run.LABELS, 'protocol': protocol}}, checkpoint)
            run.raw.save_predictions(flow_path, original)
            run.raw.save_predictions(mean_path, original)
            source = {'directory': '.', 'best_epoch': 95, 'checkpoint_sha256': run.raw.sha(checkpoint),
                      'joint_flow_predictions_sha256': run.raw.sha(flow_path),
                      'joint_mean_predictions_sha256': run.raw.sha(mean_path)}
            loaded, gold, reference, flow_metrics, mean_metrics = run.load_source(root, 42, source, protocol, original)
            self.assertFalse(any(param.requires_grad for param in loaded.parameters()))
            state = run.raw.state_digest(loaded.state_dict())
            predictions = run.replay(loaded, [batch], original, torch.device('cpu'), root, 42)
            run.verify_baseline(predictions['1'], gold, 42, root)
            self.assertEqual(run.raw.state_digest(loaded.state_dict()), state)
            self.assertEqual(gold, reference)
            self.assertEqual(flow_metrics, mean_metrics)

    def test_baseline_rejects_swapped_predictions_with_identical_metrics(self):
        original = [{'sample_id': 'a', 'label': 1, 'prediction': 1},
                    {'sample_id': 'b', 'label': 1, 'prediction': 3}]
        swapped = [{**original[0], 'prediction': 3}, {**original[1], 'prediction': 1}]
        self.assertEqual(run.study.validate_predictions(swapped, probabilities=False),
                         run.study.validate_predictions(original, probabilities=False))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'differs on 2 predictions'):
                run.verify_baseline(swapped, original, 43, Path(tmp))
            check = run.raw.read(Path(tmp) / 'seed43_lambda1_check.json')
            self.assertEqual([row['sample_id'] for row in check['examples']], ['a', 'b'])

    def test_focus_distinguishes_original_tradeoff_from_other_changes(self):
        def rows(predictions):
            ids = ('web_gain', 'recon_loss', 'wrong', 'web_loss', 'recon_gain')
            labels = (1, 3, 3, 1, 3)
            return [{'sample_id': sid, 'label': label, 'prediction': pred}
                    for sid, label, pred in zip(ids, labels, predictions)]

        predictions = {'1': rows((1, 1, 2, 3, 3)),
                       '0.5': rows((3, 3, 1, 1, 1)),
                       '0': rows((1, 3, 1, 3, 3))}
        reference = rows((3, 3, 2, 1, 1))
        paired = list(run.paired_rows(predictions, reference))
        counts, focus = run.analyze_focus(paired)
        self.assertEqual(counts['0.5']['WebBased'], {'repaired': 1, 'regressed': 1,
                         'joint_mean_losses_recovered': 1, 'joint_mean_gains_lost': 1})
        self.assertEqual(counts['0.5']['Recon'], counts['0.5']['WebBased'])
        self.assertEqual(counts['0']['WebBased']['regressed'], 0)
        self.assertEqual(counts['0']['Recon']['repaired'], 1)
        self.assertEqual(counts['0']['Recon']['joint_mean_losses_recovered'], 1)
        self.assertEqual([row['sample_id'] for row in focus], ['web_gain', 'recon_loss', 'web_loss', 'recon_gain'])
        metrics = {s: run.study.validate_predictions(records, probabilities=False)
                   for s, records in predictions.items()}
        results = {'42': {'best_epoch': 95, 'metrics': metrics, 'focus_counts': counts,
                          'joint_mean_metrics': run.study.validate_predictions(reference, probabilities=False)}}
        evidence = {'WebBased_requires_stronger_unique_ids': 1, 'Recon_recovered_unique_ids': 1}
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'report.md'
            run.write_report(report, results, evidence)
            self.assertIn('| 42 | 0.5 | 1 / 1 | 1 | 1 / 2 | 1 | 1 |', report.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
