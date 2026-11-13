"""Verify same-encoding readouts and paired validation evidence on CPU."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import torch

import replay_flow_query_selective as run
from test_flow_query_joint_packet2 import joint_fixture


class SelectiveReplayChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    @torch.no_grad()
    def test_both_paths_use_one_encoding_and_match_original_endpoints(self):
        backbone, _, batch, _ = joint_fixture()
        model = run.joint.JointCrossLayerQueryClassifier(
            backbone.flow_encoder, backbone.packet_encoder, backbone.fusion, 'flow').eval()
        model.packet_pool.query.weight.fill_(0.1)
        original = model(batch)
        (flow, first, mean), _ = run.joint.legacy.JointCrossLayerClassifier.encode(model, batch)
        expected_off, _ = model.forward_features(flow, first, mean)
        before = run.raw.state_digest(model.state_dict())
        calls = []
        hooks = [module.register_forward_hook(lambda *args: calls.append(1))
                 for module in (model.flow_encoder, model.packet_encoder)]
        try:
            on, off, _ = run.paired_forward(model, batch)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(on, original))
        self.assertTrue(torch.equal(off, expected_off))
        self.assertFalse(torch.equal(on, off))
        self.assertEqual(run.raw.state_digest(model.state_dict()), before)

        rows = [{'sample_id': sid, 'label': label} for sid, label in zip(batch.sample_id, batch.y.view(-1).tolist())]
        archived = [{**row, 'prediction': pred} for row, pred in zip(rows, original.argmax(-1).tolist())]
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            records = run.replay(model, [batch], rows, archived, torch.device('cpu'), output)
            self.assertTrue(run.raw.read(output / 'small_sample_check.json')['passed'])
            self.assertEqual(run.predictions(records, 'on'), archived)
            self.assertEqual(records[0]['off_log_probabilities'], off[0].tolist())

    def test_pair_outcomes_error_directions_and_oracle_correctness(self):
        values = [(1, 3, 1), (3, 1, 0), (3, 3, 1), (1, 1, 1), (2, 0, 2), (3, 1, 3)]
        records = [{'sample_id': str(i), 'label': y, 'on_prediction': p, 'off_prediction': q}
                   for i, (y, p, q) in enumerate(values)]
        metrics, analysis = run.analyze(records, records)
        self.assertEqual(list(analysis['outcomes'].values()), [1, 1, 3, 1])
        self.assertEqual(analysis['oracle_correct_gain'], 3)
        self.assertEqual(metrics['oracle']['accuracy'], 5 / 6)
        self.assertEqual(analysis['only_off_correct_origins']['WebBased'], {'Recon': 1})
        web = analysis['directions']['WebBased -> Recon']
        recon = analysis['directions']['Recon -> WebBased']
        self.assertEqual((web['on'], web['off'], web['oracle']), (1, 0, 0))
        self.assertEqual((recon['on'], recon['off'], recon['oracle']), (2, 1, 1))
        self.assertEqual(recon['original_errors_still_wrong'], 1)
        self.assertEqual(recon['new_errors_from_correct'], 1)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run.write_report(output, metrics, analysis, 0.1, 0.0)
            self.assertEqual(run.raw.read(output / 'metrics.json')['paired_analysis']['next_step'],
                             'check_training_pairs')
            self.assertIn('最多可多判对 3 张图', (output / 'report.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
