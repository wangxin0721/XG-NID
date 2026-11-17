"""Exercise the paired gate, frozen extraction, and selected validation replay on CPU."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

import run_final_d32_gate_ablation as run
from test_packet_transition_readout import fixture


class FinalD32GateAblationChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')

    def test_matched_parameter_initialization_and_gate_conditioning(self):
        model, _ = fixture()
        features = torch.linspace(-1, 1, 5 * 69).reshape(5, 69)
        arms = run.make_arms(model.fusion, features.mean(0, keepdim=True))
        static, dynamic = (arms[name] for name in run.ARMS)
        self.assertEqual(sum(p.numel() for p in static.parameters()), 2984)
        self.assertEqual(run.independent.model_state(static.gate), run.independent.model_state(dynamic.gate))
        self.assertEqual(run.independent.model_state(static.classifier), run.independent.model_state(dynamic.classifier))
        torch.testing.assert_close(static(features)[0], dynamic(features)[0], rtol=0, atol=0)
        with torch.no_grad():
            for arm in arms.values():
                arm.gate[0].weight.fill_(0.01)
                arm.gate[0].bias.zero_()
                arm.gate[-1].weight.fill_(0.2)
        _, static_weights = static(features)
        _, dynamic_weights = dynamic(features)
        torch.testing.assert_close(static_weights, static_weights[:1].expand_as(static_weights))
        self.assertGreater(float((dynamic_weights[0] - dynamic_weights[-1]).abs().max()), 0.01)

    def test_raw_graph_extraction_training_and_selected_replay(self):
        model, graphs = fixture()
        readout = run.transition.make_heads({'seed': 42}, self.device, ('B',))['B']
        with torch.no_grad():
            readout.output.weight.fill_(0.03)
        readout.requires_grad_(False).eval()
        states = {name: run.independent.model_state(module) for name, module in (('C', model), ('B', readout))}
        rows = [{'sample_id': graph.sample_id, 'label': graph.y.item()} for graph in graphs]
        batches = [Batch.from_data_list(graphs[:4]), Batch.from_data_list(graphs[4:])]
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            cache = run.collect(model, readout, batches, rows, self.device, output, 'extract')
            with torch.no_grad():
                expected = []
                for i, batch in enumerate(batches):
                    encoded, _ = run.transition.encode(model, batch, self.device)
                    logits, alpha, _ = model_scores(model, readout, encoded)
                    features = cache.features[4*i:4*i + batch.num_graphs]
                    reconstructed = model.fusion.shared_head(
                        alpha * features[:, :32] + (1 - alpha) * features[:, 32:64]).log_softmax(dim=-1)
                    torch.testing.assert_close(logits, reconstructed, rtol=0, atol=0)
                    expected.extend(logits.argmax(-1).tolist())
                self.assertEqual([r['prediction'] for r in cache.original], expected)
            arms = run.make_arms(model.fusion, cache.features.mean(0, keepdim=True))
            schedule = {**run.SCHEDULE, 'epochs': 2, 'batch_size': 4}
            labels = torch.tensor([row['label'] for row in rows])
            best, last = run.fit(arms, cache.features, labels, cache.features, rows, 4, output, schedule)
            history = [json.loads(line) for line in (output / 'epochs.jsonl').read_text().splitlines()]
            self.assertEqual([row['updates_per_arm'] for row in history], [0, 2, 2])
            checkpoint = output / 'fusion_arms.pt'
            torch.save({'best': best, 'last': last}, checkpoint)
            saved = torch.load(checkpoint, weights_only=True)
            replay = run.collect(model, readout, batches, rows, self.device, output, 'replay')
            exports, _ = run.selected_predictions(arms, saved['best'], replay, rows, {'eval_batch_size': 4}, self.device)
            for name in run.ARMS:
                expected_best = max(history, key=lambda row: row['validation'][name]['macro_f1'])
                self.assertEqual(saved['best'][name]['epoch'], expected_best['epoch'])
            result = run.save_partition(output, 'validation', rows, replay, exports)
            self.assertEqual(result['original_B']['samples'], len(graphs))
            self.assertEqual(result['dynamic']['samples'], len(graphs))
        run.check_frozen(model, readout, {'frozen_c_sha256': states['C'], 'frozen_readout_sha256': states['B']}, {})

    def test_fixed_test_requires_locked_validation_runs(self):
        with TemporaryDirectory() as directory, patch.object(run, 'source') as source:
            root = Path(directory)
            (root / 'seed42').mkdir()
            run.raw.write(root / 'seed42' / 'status.json', {'status': 'running'})
            with self.assertRaisesRegex(ValueError, 'Seed42 validation run is incomplete'):
                run.test_all(root, None, root / 'test')
            source.assert_not_called()


def model_scores(model, readout, features):
    residual, _ = readout(features.packet_mean, features.transitions, features.lengths)
    scores, alpha = model.fusion(features.flow, features.packet + residual)
    return scores, alpha, residual


if __name__ == '__main__':
    unittest.main()
