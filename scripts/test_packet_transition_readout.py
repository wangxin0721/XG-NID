"""Check packed chronological edges, staged gradients, and paired checkpoint replay."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import Batch

import run_packet_transition_readout as run
from test_flow_query_joint_packet2 import TinyGraphs, joint_fixture
from xgnid.model import FLOW_TO_PACKET, PACKET_TO_FLOW, PACKET_TO_PACKET


def fixture():
    base, _, _, graphs = joint_fixture()
    for graph in graphs:
        packet_count = graph['packet'].num_nodes
        contain = graph[FLOW_TO_PACKET]
        contain.edge_index = contain.edge_index.flip(1)
        node = contain.edge_index[1].float()
        contain.edge_attr[:, 0] = node % 2
        contain.edge_attr[:, 3] = node * 10
        graph[PACKET_TO_FLOW].edge_index = contain.edge_index.flip(0)
        graph[PACKET_TO_FLOW].edge_attr = contain.edge_attr
        graph[PACKET_TO_PACKET].edge_attr.fill_(packet_count)
    model = run.joint.JointCrossLayerQueryClassifier(
        base.flow_encoder, base.packet_encoder, base.fusion, 'flow').requires_grad_(False).eval()
    return model, graphs


class TransitionReadoutChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')

    def test_chronological_edges_and_packed_reordering(self):
        model, graphs = fixture()
        batch = Batch.from_data_list(graphs[:3])
        src, dst, metadata, lengths, counts = run.chronological_edges(batch)
        np.testing.assert_array_equal(src, [2, 3, 5])
        np.testing.assert_array_equal(dst, [3, 4, 6])
        np.testing.assert_array_equal(lengths, [0, 2, 1])
        np.testing.assert_array_equal(metadata[:, :4], np.eye(4)[[1, 2, 1]])
        np.testing.assert_allclose(metadata[:, 4], np.log1p([3, 3, 2]))
        np.testing.assert_allclose(metadata[:, 5], np.diff(np.log1p([0, 10, 20]))[[0, 1, 0]])
        self.assertEqual(counts['reverse_links'], 3)
        self.assertEqual(counts['other_links'], 2)
        features, _ = run.encode(model, batch, self.device)
        with torch.no_grad():
            torch.testing.assert_close(run.baseline(model, features), model(batch), rtol=0, atol=0)
        cache = run.FrozenCache([features], batch.y.tolist())
        reordered, labels = cache.batch([2, 0, 1], self.device)
        self.assertEqual(labels.tolist(), [2, 0, 1])
        self.assertEqual(reordered.lengths.tolist(), [1, 0, 2])
        torch.testing.assert_close(reordered.transitions, features.transitions[[2, 0, 1]])
        graphs[1][FLOW_TO_PACKET].edge_attr[1, 0] = 9
        _, _, _, lengths, counts = run.chronological_edges(Batch.from_data_list(graphs[:3]))
        self.assertEqual(lengths.tolist(), [0, 0, 1])
        self.assertEqual(counts['invalid_forward_links'], 2)

    def test_paired_training_selects_and_replays_best(self):
        model, graphs = fixture()
        data = TinyGraphs(graphs)
        rows = {part: [{'sample_id': g.sample_id, 'label': g.y.item(), 'partition': part} for g in graphs]
                for part in ('train', 'val')}
        data.rows = rows['train']
        config = {'seed': 42, 'workers': 0, 'eval_batch_size': 4, 'batch_size': 4,
                  'epochs': 2, 'learning_rate': 0.001, 'budget_reference_pp': 0.10}
        before = run.raw.state_digest(model.state_dict())
        heads = run.make_heads(config, self.device)
        self.assertEqual(sum(p.numel() for p in heads['B'].parameters()), 3200)
        attention = run.check_attention(run.PacketTransitionReadout(), self.device)
        self.assertEqual(attention['single_edge_attention'], 1)
        checks = run.probe(model, data, config, self.device)
        self.assertTrue(checks['passed'])
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            train, train_coverage, _ = run.cache_split(model, data, rows['train'], 'train', config, self.device, output)
            val, val_coverage, reference = run.cache_split(model, data, rows['val'], 'val', config, self.device, output, heads)
            best, last, updates = run.fit(model, heads, train, val, rows['val'], reference, config, self.device, output)
            checkpoint = output / 'transition_readouts.pt'
            torch.save({'best': best, 'last': last}, checkpoint)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            predictions = run.replay_selected(model, saved, data, rows['val'], config, self.device, output)
            self.assertEqual(predictions['A'], reference)
            ledger = [json.loads(line) for line in (output / 'epochs.jsonl').read_text().splitlines()]
            self.assertEqual([row['updates_per_arm'] for row in ledger], [0, 2, 2])
            for arm, item in best.items():
                expected = max(ledger, key=lambda row: row['validation'][arm]['macro_f1'])
                self.assertEqual(item['epoch'], expected['epoch'])
                self.assertEqual(item['prediction_ids_sha256'], run.joint.prediction_digest(predictions[arm]))
            run.summarize(output, rows['val'], predictions, val.features.lengths.tolist(), saved, config, updates,
                          {'val': val_coverage, 'train': train_coverage}, 0)
            self.assertEqual(set(run.raw.read(output / 'metrics.json')['comparisons']), {'B-A', 'C1-A', 'C1-B'})
        self.assertEqual(updates, 4)
        self.assertEqual(run.raw.state_digest(model.state_dict()), before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
